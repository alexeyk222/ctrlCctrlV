#!/usr/bin/env python3
"""Compare identical detector tensors through ONNX CPU and static Core ML."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--compute-units", default="CPUAndNeuralEngine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=Path.home() / ".paddlex/official_models/PP-OCRv5_server_det_onnx/inference.onnx")
    parser.add_argument("--model", type=Path, default=Path("/private/tmp/ctrlcctrlv-bench/det-pool-fix/inference.onnx"))
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    x = np.load(args.input)["x"]
    options = ort.SessionOptions()
    options.log_severity_level = 3
    original = ort.InferenceSession(str(args.original), options, providers=["CPUExecutionProvider"])
    ref = original.run(None, {"x": x})[0]
    del original
    fixed = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])
    repaired = fixed.run(None, {"x": x})[0]
    del fixed
    print("repair_error", float(np.max(np.abs(ref - repaired))), flush=True)
    options = ort.SessionOptions()
    options.log_severity_level = 2
    for name, value in zip(("DynamicDimension.0", "DynamicDimension.1", "DynamicDimension.2"), (x.shape[0], x.shape[2], x.shape[3])):
        options.add_free_dimension_override_by_name(name, value)
    provider = {"ModelFormat": "MLProgram", "MLComputeUnits": args.compute_units,
                "ProfileComputePlan": "1", "RequireStaticInputShapes": "1",
                "ModelCacheDirectory": "/private/tmp/ctrlcctrlv-bench/static-cache/" + "x".join(map(str, x.shape))}
    start = time.perf_counter()
    session = ort.InferenceSession(str(args.model), options,
        providers=[("CoreMLExecutionProvider", provider), "CPUExecutionProvider"])
    load_seconds = time.perf_counter() - start
    times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        actual = session.run(None, {"x": x})[0]
        times.append(time.perf_counter() - start)
    report = {"shape": list(x.shape), "compute_units": args.compute_units,
              "load_seconds": load_seconds, "seconds": times,
              "repair_max_error": float(np.max(np.abs(ref - repaired))),
              "coreml_max_error": float(np.max(np.abs(ref - actual))),
              "coreml_mean_error": float(np.mean(np.abs(ref - actual))),
              "mask_disagreement": float(np.mean((ref > .3) != (actual > .3)))}
    args.output.write_text(json.dumps(report, indent=2))
    np.savez(args.output.with_suffix(".npz"), x=x, ref=ref, actual=actual)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
