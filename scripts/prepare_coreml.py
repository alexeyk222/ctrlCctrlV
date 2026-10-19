#!/usr/bin/env python3
"""Prepare a separate ONNX detector copy for the Core ML MLProgram converter."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


CONVERSION_VERSION = "ppocrv5-det-fp16-v1"


def cached_model(source, cache):
    """Convert once under a process lock; publish only complete model files."""
    import onnxruntime as ort

    source, cache = Path(source), Path(cache)
    with (source / "inference.onnx").open("rb") as model_file:
        source_hash = hashlib.file_digest(model_file, "sha256").hexdigest()
    identity = f"{CONVERSION_VERSION}-{ort.__version__}-{source_hash[:16]}"
    destination = cache / identity
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / (identity + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").exists() and (destination / "inference.onnx").exists():
            return destination / "inference.onnx"
        with tempfile.TemporaryDirectory(prefix="convert-", dir=cache) as temporary:
            temporary = Path(temporary)
            changed = prepare(source, temporary, fp16=True)
            manifest = {"version": CONVERSION_VERSION, "source_sha256": source_hash,
                        "onnxruntime": ort.__version__, "pooling_fixes": changed}
            (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("inference.onnx", "inference.yml", "manifest.json"):
                os.replace(temporary / filename, destination / filename)
    return destination / "inference.onnx"


def prepare(source, destination, fp16=False):
    import onnx
    from onnx import helper

    source, destination = Path(source), Path(destination)
    model = onnx.load(source / "inference.onnx")
    changed = []
    for node in model.graph.node:
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (node.op_type == "MaxPool" and attributes.get("ceil_mode") == 1
                and attributes.get("auto_pad") in (b"SAME_UPPER", b"SAME_LOWER")
                and all(s == 1 for s in attributes.get("strides", [1, 1]))):
            # With stride 1 the output-size expression is already an integer;
            # ceil and floor are identical. Core ML disallows ceil + SAME.
            for attr in node.attribute:
                if attr.name == "ceil_mode":
                    attr.i = 0
            changed.append(node.name)
    onnx.checker.check_model(model)
    destination.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination / "inference.onnx")
    if fp16:
        import onnxruntime as ort
        from onnxruntime.transformers.float16 import convert_float_to_float16

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.optimized_model_filepath = str(destination / "fused.onnx")
        session = ort.InferenceSession(str(destination / "inference.onnx"), opts,
                                       providers=["CPUExecutionProvider"])
        del session
        model = convert_float_to_float16(onnx.load(destination / "fused.onnx"), keep_io_types=True)
        onnx.save(model, destination / "inference.onnx")
    shutil.copy2(source / "inference.yml", destination / "inference.yml")
    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    print(prepare(args.source, args.destination, args.fp16))
