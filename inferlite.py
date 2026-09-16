"""
Inference for the exported GTCRN-DF .tflite model. No PyTorch dependency —
STFT/ISTFT are reimplemented in numpy to match STFTFrontEnd in train.py
(sqrt-Hann analysis/synthesis window, center=True framing).

    python infer_tflite.py --tflite gtcrn.tflite --input noisy.wav --output clean.wav
    python infer_tflite.py --tflite gtcrn.tflite --input in_dir/ --output out_dir/

IMPORTANT: --sample_rate, --n_fft, --hop_length must match the values the
checkpoint was trained/exported with (see the "config" dict inside the
original .pt checkpoint). --mask_min is NOT adjustable here: it was baked
into the graph as a constant when the model was exported to ONNX/TFLite.
If you need a different mask floor, re-export with a different value and
produce a new .tflite file.

Requires: numpy, soundfile, scipy (for resampling), and either
`tensorflow` or the lighter `tflite_runtime` package.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg")

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter  # lighter on-device alternative


# ===========================================================================
# STFT / ISTFT (numpy re-implementation of STFTFrontEnd, train.py)
# ===========================================================================
def make_sqrt_hann(win_length):
    n = np.arange(win_length)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / win_length)  # periodic Hann
    return np.sqrt(np.clip(hann, 0.0, None)).astype(np.float32)


def stft_np(wav, n_fft, hop_length, window):
    """wav: (N,) float32 -> spec: (F, T, 2) float32, matching torch.stft(
    center=True, pad_mode='reflect')."""
    pad = n_fft // 2
    x = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(x) - n_fft) // hop_length
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    frames = x[idx] * window[None, :]                    # (T, n_fft)
    spec = np.fft.rfft(frames, n=n_fft, axis=-1).T        # (F, T)
    return np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)


def istft_np(spec, n_fft, hop_length, window, length=None):
    """spec: (F, T, 2) -> wav: (N,) float32. Overlap-add with window-energy
    normalisation, matching torch.istft's reconstruction."""
    complex_spec = spec[..., 0] + 1j * spec[..., 1]       # (F, T)
    frames = np.fft.irfft(complex_spec.T, n=n_fft, axis=-1)  # (T, n_fft)
    n_frames = frames.shape[0]

    out_len = (n_frames - 1) * hop_length + n_fft
    out = np.zeros(out_len, dtype=np.float64)
    win_sum = np.zeros(out_len, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop_length
        out[s:s + n_fft] += frames[i] * window
        win_sum[s:s + n_fft] += window ** 2
    win_sum = np.where(win_sum > 1e-11, win_sum, 1.0)
    out = out / win_sum

    pad = n_fft // 2
    out = out[pad: out_len - pad]
    if length is not None:
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
        else:
            out = out[:length]
    return out.astype(np.float32)


def read_audio(path, target_sr):
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


# ===========================================================================
# TFLite model wrapper
# ===========================================================================
class TFLiteGTCRN:
    def __init__(self, tflite_path):
        self.interpreter = Interpreter(model_path=str(tflite_path))
        in_details = self.interpreter.get_input_details()
        out_details = self.interpreter.get_output_details()
        self.input_index = in_details[0]["index"]
        self.output_index = out_details[0]["index"]

    def __call__(self, spec):
        """spec: (1, F, T, 2) float32 numpy -> (1, F, T, 2) float32 numpy."""
        self.interpreter.resize_tensor_input(self.input_index, spec.shape)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_index, spec.astype(np.float32))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)


def enhance(model, wav, n_fft, hop_length, window, chunk_sec=20.0,
            context_sec=3.0, sample_rate=16000, norm_rms=0.1):
    """Same chunking/normalisation strategy as infer.py's enhance()."""
    n = wav.shape[0]
    in_rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    scale = (norm_rms / in_rms) if in_rms > 1e-9 else 1.0
    x = wav * scale

    chunk = int(chunk_sec * sample_rate)
    ctx = int(context_sec * sample_rate)
    out = np.zeros(n, dtype=np.float32)

    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        start = max(0, pos - ctx)
        seg = x[start:end]

        spec = stft_np(seg, n_fft, hop_length, window)[None, ...]   # (1,F,T,2)
        pred = model(spec)
        y = istft_np(pred[0], n_fft, hop_length, window, length=seg.shape[-1])

        out[pos:end] = y[pos - start:]           # drop the priming context
        pos = end

    return (out / scale).astype(np.float32)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tflite", required=True)
    p.add_argument("--input", required=True, help="a wav file or a folder of them")
    p.add_argument("--output", required=True, help="output file or folder")
    p.add_argument("--sample_rate", type=int, default=16000,
                   help="must match the checkpoint the .tflite was exported from")
    p.add_argument("--n_fft", type=int, default=512,
                   help="must match the checkpoint the .tflite was exported from")
    p.add_argument("--hop_length", type=int, default=256,
                   help="must match the checkpoint the .tflite was exported from")
    p.add_argument("--mix_rms", type=float, default=0.1,
                   help="training loudness normalisation target (cfg['mix_rms'] in train.py)")
    p.add_argument("--output_gain_db", type=float, default=0.0)
    p.add_argument("--chunk_sec", type=float, default=20.0)
    p.add_argument("--context_sec", type=float, default=3.0)
    args = p.parse_args()

    model = TFLiteGTCRN(args.tflite)
    window = make_sqrt_hann(args.n_fft)

    print(f"loaded {args.tflite} | {args.sample_rate} Hz, "
          f"n_fft={args.n_fft}, hop={args.hop_length}")

    in_path = Path(args.input)
    out_path = Path(args.output)
    if in_path.is_dir():
        files = sorted(f for f in in_path.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        out_path.mkdir(parents=True, exist_ok=True)
        pairs = [(f, out_path / f.name) for f in files]
    else:
        if out_path.suffix == "":
            out_path.mkdir(parents=True, exist_ok=True)
            pairs = [(in_path, out_path / in_path.name)]
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pairs = [(in_path, out_path)]

    gain = 10.0 ** (args.output_gain_db / 20.0)
    for src, dst in pairs:
        wav = read_audio(src, args.sample_rate)
        y = enhance(model, wav, args.n_fft, args.hop_length, window,
                    args.chunk_sec, args.context_sec, args.sample_rate,
                    args.mix_rms) * gain
        peak = float(np.max(np.abs(y))) + 1e-12
        if peak > 0.999:                        # only ever attenuates
            y = y * (0.999 / peak)
        sf.write(str(dst), y, args.sample_rate)
        print(f"  {src.name} -> {dst}")

    print("done")


if __name__ == "__main__":
    main()