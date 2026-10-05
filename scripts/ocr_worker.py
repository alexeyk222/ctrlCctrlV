#!/usr/bin/env python3
"""Keep PP-OCRv5 loaded; exchange one JSON request/response per stdin/stdout line."""

import argparse
import contextlib
import json
import sys
import time

from ocr_paddle import backend_status, create_ocr, recognize


def respond(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(language, accelerator="auto"):
    with contextlib.redirect_stdout(sys.stderr):
        ocr = create_ocr(language, accelerator)
    respond(
        {
            "ready": True,
            "backend": getattr(ocr, "_ctrlcctrlv_backend", "unknown"),
        }
    )

    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            image = request["image"]
            if not isinstance(image, str) or not image:
                raise ValueError("image must be a nonempty path")
            start = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                text = recognize(ocr, image)
            respond({"id": request_id, "text": text, "backend": backend_status(ocr),
                     "seconds": round(time.perf_counter() - start, 3)})
        except Exception as exc:
            respond({"id": request_id, "error": str(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=("ru", "en"), default="ru")
    parser.add_argument(
        "--accelerator",
        choices=("auto", "coreml", "onnx", "paddle"),
        default="auto",
    )
    args = parser.parse_args()
    try:
        serve(args.lang, args.accelerator)
    except Exception as exc:
        print(f"PaddleOCR worker error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
