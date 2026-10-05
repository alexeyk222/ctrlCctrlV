#!/usr/bin/env python3
"""Local VLM OCR on Apple GPU (MLX); persistent JSON lines or single image.

Profiles (see docs/VL_RESULTS.md for the comparison):
  fast      Qwen3.5-4B 4bit: near-exact characters, ~1-13 s, loses code indentation
  accurate  chandra-ocr-2 8bit: keeps code indentation, about twice as slow
  paddle    PaddleOCR-VL-1.6: previous default
"""

import argparse
import contextlib
import html
import json
import re
import sys
import time

LITERAL_PROMPT = ("Transcribe all text in this image exactly as it appears, preserving line "
                  "breaks and code indentation. Output plain text only, no Markdown, no commentary.")
PROFILES = {
    "fast": {"model": "mlx-community/Qwen3.5-4B-MLX-4bit",
             "revision": "32f3e8ecf65426fc3306969496342d504bfa13f3",
             "prompt": LITERAL_PROMPT, "output": "markdown",
             "backend": "Qwen3.5-4B / MLX Apple GPU / 4bit"},
    "accurate": {"model": "mlx-community/chandra-ocr-2-oQ8",
                 "revision": "eafcb4c79468ff6cf8b76ecc3aedbffe0dd82282",
                 "prompt": LITERAL_PROMPT, "output": "html",
                 "backend": "chandra-ocr-2 / MLX Apple GPU / 8bit"},
    "paddle": {"model": "PaddlePaddle/PaddleOCR-VL-1.6",
               "revision": "c5630abae1d940eafe0697512a0325494b02ab42",
               "prompt": "OCR:", "output": "plain",
               "backend": "PaddleOCR-VL-1.6 / MLX Apple GPU / BF16"},
}
# Upscaled images cost prompt tokens; beyond this the gain is not worth the time.
MAX_SIDE = 2400


def load_model(profile):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import load

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal GPU is unavailable")
    mx.set_default_device(mx.gpu)
    # Installation downloads weights explicitly. Screenshot OCR stays offline.
    path = snapshot_download(profile["model"], revision=profile["revision"], local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.model", "*.jinja"])
    return load(path)


def from_markdown(text):
    """Drop Markdown wrappers the model adds; text inside fenced blocks stays literal."""
    lines, fenced = [], False
    for line in text.splitlines():
        if re.fullmatch(r"\s*```[\w+-]*\s*", line):
            fenced = not fenced
            continue
        lines.append(line if fenced else re.sub(r"`([^`\n]+)`", r"\1", line))
    return "\n".join(lines)


def from_html(text):
    """chandra emits layout blocks as HTML; keep <pre> whitespace, one blank line between blocks."""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"</(div|p|pre|h[1-6]|li|tr)>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip("\n")


def prepare_image(image, scale):
    from PIL import Image

    picture = Image.open(image).convert("RGB")
    scale = min(scale, MAX_SIDE / max(picture.size))
    if scale > 1:
        picture = picture.resize((round(picture.width * scale), round(picture.height * scale)),
                                 Image.Resampling.LANCZOS)
    return picture


def recognize(model, processor, image, profile, scale=1.0, max_tokens=8192):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(processor, model.config, profile["prompt"], num_images=1)
    picture = prepare_image(image, scale) if isinstance(image, str) else image
    result = generate(model, processor, prompt, image=[picture],
                      max_tokens=max_tokens, temperature=0.0, verbose=False)
    if result.finish_reason == "length" or result.generation_tokens >= max_tokens:
        raise RuntimeError("OCR output reached its token limit; select a smaller region")
    if profile["output"] == "markdown":
        return from_markdown(result.text)
    if profile["output"] == "html":
        return from_html(result.text)
    return result.text


def respond(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if not args.worker and not args.image:
        parser.error("provide an image or --worker")
    profile = PROFILES[args.profile]
    try:
        with contextlib.redirect_stdout(sys.stderr):
            model, processor = load_model(profile)
        if not args.worker:
            with contextlib.redirect_stdout(sys.stderr):
                text = recognize(model, processor, args.image, profile, args.scale)
            print(text)
            return 0
        respond({"ready": True, "backend": profile["backend"]})
        for line in sys.stdin:
            request_id = None
            try:
                request = json.loads(line)
                request_id = request["id"]
                image = request["image"]
                if not isinstance(image, str) or not image:
                    raise ValueError("image must be a nonempty path")
                scale = float(request.get("scale", 1.0))
                start = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    text = recognize(model, processor, image, profile, scale)
                respond({"id": request_id, "text": text, "backend": profile["backend"],
                         "seconds": round(time.perf_counter() - start, 3)})
            except Exception as exc:
                respond({"id": request_id, "error": str(exc)})
        return 0
    except Exception as exc:
        print(f"VL OCR error ({args.profile}): {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
