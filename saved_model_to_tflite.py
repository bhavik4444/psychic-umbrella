# saved_model_to_tflite.py
"""
python saved_model_to_tflite.py --saved_model gtcrn_tf_out --output gtcrn.tflite
"""
import argparse
import tensorflow as tf


def convert(saved_model_dir, tflite_path, quantize_dynamic=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # TFLITE_BUILTINS alone is often insufficient for GRU-derived subgraphs;
    # SELECT_TF_OPS lets the converter fall back to native TF kernels for any
    # op onnx2tf couldn't map to a TFLite builtin, so it doesn't silently
    # produce a wrong graph or refuse to convert.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    if quantize_dynamic:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {tflite_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--saved_model", required=True)
    p.add_argument("--output", default="gtcrn.tflite")
    p.add_argument("--quantize_dynamic", action="store_true")
    args = p.parse_args()
    convert(args.saved_model, args.output, args.quantize_dynamic)