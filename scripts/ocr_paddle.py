#!/usr/bin/env python3
"""PP-OCRv5 screenshot OCR. Text goes to stdout; diagnostics go to stderr."""

import argparse
import contextlib
import os
import platform
import statistics
import sys


def reconstruct(texts, boxes):
    """Restore reading order and approximate monospace spacing from OCR boxes."""
    if len(texts) != len(boxes):
        raise ValueError("PaddleOCR returned a different number of texts and boxes")

    segments = []
    for text, box in zip(texts, boxes):
        if not text:
            continue
        left, top, right, bottom = map(float, box)
        segments.append((left, (top + bottom) / 2, right, bottom - top, text))

    if not segments:
        return ""

    height = statistics.median(s[3] for s in segments)
    width_samples = [(s[2] - s[0]) / len(s[4]) for s in segments if s[4]]
    char_width = statistics.median(width_samples) if width_samples else max(1, height * 0.55)
    left_margin = min(s[0] for s in segments)

    rows = []
    for segment in sorted(segments, key=lambda s: s[1]):
        matching = min(rows, key=lambda row: abs(row[0] - segment[1]), default=None)
        if matching is not None and abs(matching[0] - segment[1]) <= max(1, height * 0.55):
            matching[1].append(segment)
            matching[0] = sum(s[1] for s in matching[1]) / len(matching[1])
        else:
            rows.append([segment[1], [segment]])

    lines = []
    for _, row in sorted(rows, key=lambda r: r[0]):
        row.sort(key=lambda s: s[0])
        leading = max(0, round((row[0][0] - left_margin) / char_width))
        parts = [" " * leading, row[0][4]]
        for previous, current in zip(row, row[1:]):
            gap = max(0, round((current[0] - previous[2]) / char_width))
            parts.extend((" " * gap, current[4]))
        lines.append("".join(parts))
    return "\n".join(lines)


def _onnx_ocr(language, accelerator, max_side):
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        lang=language,
        engine="onnxruntime",
        engine_config={"providers": ["CPUExecutionProvider"], "log_severity_level": 3},
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=max_side,
        text_det_limit_type="max",
    )

    ocr._ctrlcctrlv_backend = "ONNX CPU"
    pipeline = ocr.paddlex_pipeline._pipeline
    # The default 1000-contour cap discarded most lines on a large dark fixture.
    # Keep detection/recognition thresholds unchanged and inspect more contours.
    pipeline.text_det_model.post_op.max_candidates = 10000
    if accelerator == "coreml":
        try:
            from pathlib import Path
            import onnxruntime as ort
            from ocr_coreml import CoreMLDetector
            from prepare_coreml import cached_model

            if "CoreMLExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime has no CoreMLExecutionProvider")
            cache = Path(os.getenv("OCR_COREML_CACHE", "~/.cache/ctrlcctrlv-coreml")).expanduser()
            detector = pipeline.text_det_model
            model = cached_model(detector.runner.model_dir, cache / "models")
            detector.runner = CoreMLDetector(
                detector.runner, model, cache / "sessions" / model.parent.name,
                profile=os.getenv("OCR_COREML_PROFILE") == "1",
            )
            ocr._ctrlcctrlv_detector = detector.runner
            ocr._ctrlcctrlv_backend = "Core ML FP16 detector (CPU + ANE) / ONNX CPU recognizer"
        except Exception as exc:
            print(f"Core ML setup failed; using ONNX CPU: {exc}", file=sys.stderr)
    return ocr


def create_ocr(language, accelerator="auto", max_side=None):
    if max_side is None:
        max_side = int(os.getenv("OCR_MAX_SIDE", "4000"))
    if max_side < 960:
        raise ValueError("OCR_MAX_SIDE must be at least 960")

    if accelerator == "auto":
        accelerator = (
            "coreml"
            if platform.system() == "Darwin" and platform.machine() == "arm64"
            else "onnx"
        )

    if accelerator in {"coreml", "onnx"}:
        ocr = _onnx_ocr(language, accelerator, max_side)
        print(f"PP-OCRv5 backend: {ocr._ctrlcctrlv_backend}", file=sys.stderr)
        return ocr

    if accelerator != "paddle":
        raise ValueError(f"Unknown accelerator: {accelerator}")

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        lang=language,
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_side_len=max_side,
        text_det_limit_type="max",
    )
    ocr._ctrlcctrlv_backend = "PaddlePaddle CPU"
    print(f"PP-OCRv5 backend: {ocr._ctrlcctrlv_backend}", file=sys.stderr)
    return ocr


def recognize(ocr, image):
    pages = []
    for result in ocr.predict(input=image):
        data = result.json["res"]
        pages.append(reconstruct(data["rec_texts"], data["rec_boxes"]))
    return "\n".join(pages)


def backend_status(ocr):
    detector = getattr(ocr, "_ctrlcctrlv_detector", None)
    if detector and detector.last_backend != "not run":
        return detector.last_backend + " detector / ONNX CPU recognizer"
    return getattr(ocr, "_ctrlcctrlv_backend", "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--lang", choices=("ru", "en"), default="ru")
    parser.add_argument(
        "--accelerator",
        choices=("auto", "coreml", "onnx", "paddle"),
        default="auto",
    )
    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = recognize(create_ocr(args.lang, args.accelerator), args.image)
        sys.stdout.write(result + ("\n" if result else ""))
    except Exception as exc:
        print(f"PaddleOCR error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
