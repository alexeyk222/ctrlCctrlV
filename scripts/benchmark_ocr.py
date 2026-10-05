#!/usr/bin/env python3
"""Reproducible local OCR fixtures, stage timings, and provider comparisons."""

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time


def text_errors(expected, actual):
    """Character edit distance, ignoring only line-edge indentation/whitespace."""
    expected = "\n".join(line.strip() for line in expected.splitlines() if line.strip())
    actual = "\n".join(line.strip() for line in actual.splitlines() if line.strip())
    previous = list(range(len(actual) + 1))
    for row, a in enumerate(expected, 1):
        current = [row]
        for col, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (a != b)))
        previous = current
    return {"edit_distance": previous[-1], "reference_characters": len(expected),
            "cer": previous[-1] / max(1, len(expected))}


def fixtures(directory):
    from PIL import Image, ImageDraw, ImageFont

    samples = [
        "def foo(x):", "    return x | 1", "# Привет мир",
        'path = "src/main.py"', "items = [0, 1, 2, 10, 100]",
        "if value != None and count >= 10:",
        "    print(items[0], value)",
        "result = {key: value for key, value in pairs}",
        'message = "Русский текст и English code"',
        "total += (left - right) * 2 / 3",
        "assert len(items) == 5", "return result",
    ]
    cases = [
        ("small", (1100, 250), 26, False, samples[:3]),
        ("large_light", (2880, 1800), 28, False, samples * 3),
        ("large_dark", (2880, 1800), 24, True, samples * 3),
        ("small_font", (2560, 1600), 18, False, samples * 4),
    ]
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, size, font_size, dark, lines in cases:
        path = directory / f"{name}.png"
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
        image = Image.new("RGB", size, "#202124" if dark else "white")
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(lines):
            draw.text((35, 25 + index * (font_size + 13)), line, font=font,
                      fill="#e8eaed" if dark else "#181818")
        image.save(path)
        path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


class TimedRunner:
    def __init__(self, runner, name, events, capture):
        self.runner, self.name, self.events, self.capture = runner, name, events, capture

    def __call__(self, *args, **kwargs):
        import numpy as np

        inputs = kwargs.get("x", args[0] if args else None)
        arrays = inputs if isinstance(inputs, list) else [inputs]
        start = time.perf_counter()
        output = self.runner(*args, **kwargs)
        self.events.append({"stage": self.name, "seconds": time.perf_counter() - start,
                            "shapes": [list(x.shape) for x in arrays]})
        if self.capture and self.name == "det":
            np.savez(self.capture, x=arrays[0], y=output[0])
            self.capture = None
        if self.name == "det":
            import cv2
            mask = (output[0][0, 0] > 0.3).astype("uint8")
            self.events[-1]["contours"] = len(cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])
        return output

    def __getattr__(self, key):
        return getattr(self.runner, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "coreml", "paddle"), default="cpu")
    parser.add_argument("--format", choices=("NeuralNetwork", "MLProgram"), default="MLProgram")
    parser.add_argument("--compute-units", default="CPUAndNeuralEngine")
    parser.add_argument("--max-side", type=int, default=4000)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--case", default="small")
    parser.add_argument("--directory", type=Path, default=Path("/private/tmp/ctrlcctrlv-bench"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--det-model", type=Path)
    parser.add_argument("--static-model", type=Path)
    parser.add_argument("--max-candidates", type=int, default=0)
    args = parser.parse_args()

    paths = fixtures(args.directory)
    with contextlib.redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR
        from paddlex.inference import load_pipeline_config

        config = load_pipeline_config("OCR")
        engine = {"providers": ["CPUExecutionProvider"], "log_severity_level": 3}
        if args.backend == "coreml":
            options = {"ModelFormat": args.format, "MLComputeUnits": args.compute_units,
                       "ProfileComputePlan": "1" if args.profile else "0",
                       "RequireStaticInputShapes": "0",
                       "ModelCacheDirectory": str(args.directory / (args.format + "-cache"))}
            config["SubModules"]["TextDetection"]["engine_config"] = {
                "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [options, {}],
                "log_severity_level": 0 if args.profile else 2,
            }
        kwargs = dict(ocr_version="PP-OCRv5", lang="ru", paddlex_config=config,
                      use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False, text_det_limit_side_len=args.max_side,
                      text_det_limit_type="max")
        if args.det_model:
            kwargs.update(text_detection_model_name="PP-OCRv5_server_det",
                          text_detection_model_dir=str(args.det_model),
                          text_recognition_model_name="eslav_PP-OCRv5_mobile_rec")
        if args.backend != "paddle":
            kwargs.update(engine="onnxruntime", engine_config=engine)
        start = time.perf_counter()
        ocr = PaddleOCR(**kwargs)
        report = {"backend": args.backend, "format": args.format,
                  "compute_units": args.compute_units, "max_side": args.max_side,
                  "load_seconds": time.perf_counter() - start, "runs": []}
        pipeline = ocr.paddlex_pipeline._pipeline
        if args.max_candidates:
            pipeline.text_det_model.post_op.max_candidates = args.max_candidates
        if args.static_model:
            from ocr_coreml import CoreMLDetector
            pipeline.text_det_model.runner = CoreMLDetector(
                pipeline.text_det_model.runner, args.static_model,
                args.directory / "static-cache", compute_units=args.compute_units,
                profile=args.profile)
        events = []
        for name, predictor in [("det", pipeline.text_det_model), ("rec", pipeline.text_rec_model)]:
            predictor.runner = TimedRunner(predictor.runner, name, events, args.capture)
        for path in paths:
            if args.case != "all" and args.case != path.stem:
                continue
            for i in range(args.repeat):
                events.clear()
                start = time.perf_counter()
                result = ocr.predict(str(path))[0].json["res"]
                seconds = time.perf_counter() - start
                run = {"case": path.stem, "iteration": i, "seconds": seconds,
                       "stages": list(events), "texts": result["rec_texts"],
                       "boxes": result["rec_boxes"], "scores": result["rec_scores"]}
                from ocr_paddle import reconstruct
                run["text"] = reconstruct(result["rec_texts"], result["rec_boxes"])
                run["accuracy"] = text_errors(path.with_suffix(".txt").read_text(), run["text"])
                report["runs"].append(run)
                print(path.stem, i, round(seconds, 3), len(result["rec_texts"]), flush=True)
        ocr.close()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "load_seconds": report["load_seconds"],
                      "runs": [{k: r[k] for k in ("case", "iteration", "seconds")} for r in report["runs"]]}))


if __name__ == "__main__":
    main()
