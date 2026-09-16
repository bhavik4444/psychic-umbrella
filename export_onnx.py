"""
Export the trained GTCRN checkpoint to ONNX.

    python export_onnx.py --checkpoint checkpoints/best_model.pt --output gtcrn.onnx

Uses the legacy TorchScript-based exporter (dynamo=False). GTCRN has shape-
derived control flow (e.g. BandTRA's `pad = (-Fq) % num_bands`, and slicing
that depends on the time dimension), which the newer Dynamo/torch.export
exporter tries to symbolically verify for every possible T and fails on —
it collapses the "dynamic" time axis back to whatever value you traced with
and then raises a range-conflict error. The legacy tracer has no such
guard-checking: it traces one concrete execution and simply parameterizes
whichever axes you name in `dynamic_axes`, which is what we want here.

Also patches nn.LayerNorm before the model is built. GTCRN's intra_ln/
inter_ln use a 2-D normalized_shape (nn.LayerNorm((width, numUnits))),
which exports as a single ONNX LayerNormalization node with 2-D scale/bias.
onnx2tf's LayerNormalization -> tf_keras.layers.LayerNormalization
converter only handles a single trailing normalized axis, so it crashes
trying to load a [33, 32] weight into a shape it thinks is (33,). Swapping
in a manual mean/var/normalize implementation makes torch.onnx.export
trace it as plain ReduceMean/Sub/Mul/Sqrt/Div ops instead of a fused
LayerNormalization node, which onnx2tf handles fine regardless of how many
axes are normalized. Parameter names/shapes are unchanged, so it's a
transparent swap for a checkpoint trained with plain nn.LayerNorm.
"""
import argparse
import torch
import torch.nn as nn


class _LayerNormND(nn.Module):
    """onnx2tf-safe replacement for nn.LayerNorm over a multi-dim normalized_shape."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.dims = tuple(range(-len(normalized_shape), 0))
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape, device=device, dtype=dtype)) if bias else None
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=self.dims, keepdim=True)
        var = x.var(dim=self.dims, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


# Patch before model.py is imported/instantiated so every nn.LayerNorm(...)
# call inside GTCRN picks up the ONNX-friendly version at construction time.
nn.LayerNorm = _LayerNormND

from model import build_model_from_config  # noqa: E402  (must come after the patch)


def export_to_onnx(checkpoint_path, onnx_path, opset=17, dummy_time_frames=50):
    ck = torch.load(checkpoint_path, map_location="cpu")
    cfg = ck["config"]

    model = build_model_from_config(cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()  # critical: BatchNorm must use running stats, not batch stats

    n_fft = cfg.get("n_fft", 512)
    n_freqs = n_fft // 2 + 1

    # Shape: (B, F, T, 2) real/imag STFT, matching GTCRN.forward's docstring.
    dummy = torch.randn(1, n_freqs, dummy_time_frames, 2, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["spec"],
        output_names=["enhanced_spec"],
        dynamic_axes={
            "spec": {2: "time"},
            "enhanced_spec": {2: "time"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,   # <-- use the legacy tracer, not torch.export
    )
    print(f"exported ONNX ({n_freqs} freq bins) -> {onnx_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="gtcrn.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.opset)