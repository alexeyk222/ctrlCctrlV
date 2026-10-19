#!/usr/bin/env python3
"""Compare a local MLX OCR model with a reference, retaining unmodified output.

Metrics (edit distance against the reference):
  nows      all whitespace removed: pure character accuracy
  lines     line-edge whitespace stripped, empty lines dropped: adds line breaks
  exact     only trailing whitespace and empty lines dropped: adds code indentation
Each metric is computed on the raw model output and on a cleaned copy with
Markdown wrappers (code fences, **bold**, headings, inline backticks) removed.
"""

import argparse
import contextlib
import json
from pathlib import Path
import re
import sys
import time


def distance(expected, actual):
    previous = list(range(len(actual) + 1))
    for row, a in enumerate(expected, 1):
        current = [row]
        for col, b in enumerate(actual, 1):
            current.append(min(current[-1] + 1, previous[col] + 1,
                               previous[col - 1] + (a != b)))
        previous = current
    return previous[-1]


def normalize(text, mode):
    if mode == "nows":
        return re.sub(r"\s+", "", text)
    strip = str.strip if mode == "lines" else str.rstrip
    return "\n".join(strip(line) for line in text.splitlines() if line.strip())


def clean_markdown(text):
    text = re.sub(r"^\s*```[\w+-]*\s*$", "", text, flags=re.M)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = text.replace("**", "")
    return text.replace("`", "")


def score(reference, text):
    result = {}
    for variant, candidate in (("raw", text), ("clean", clean_markdown(text))):
        for mode in ("nows", "lines", "exact"):
            expected = normalize(reference, mode)
            result[f"{variant}_{mode}"] = distance(expected, normalize(candidate, mode))
    result["reference_characters_nows"] = len(normalize(reference, "nows"))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="resize the image before OCR (e.g. 2 for Retina-like input)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        import mlx.core as mx
        from PIL import Image

        mx.set_default_device(mx.gpu)
        image = Image.open(args.image).convert("RGB")
        if args.scale != 1.0:
            image = image.resize((round(image.width * args.scale), round(image.height * args.scale)),
                                 Image.Resampling.LANCZOS)
        start = time.perf_counter()
        model, processor = load(args.model)
        load_seconds = time.perf_counter() - start
        prompt = apply_chat_template(processor, model.config, args.prompt, num_images=1)
        runs = []
        reference = args.reference.read_text()
        for iteration in range(args.repeat):
            start = time.perf_counter()
            result = generate(model, processor, prompt, image=[image],
                              max_tokens=args.max_tokens, temperature=0, verbose=False)
            seconds = time.perf_counter() - start
            runs.append({"iteration": iteration, "seconds": round(seconds, 3), "text": result.text,
                         "finish_reason": result.finish_reason,
                         "prompt_tokens": result.prompt_tokens,
                         "tokens": result.generation_tokens,
                         "peak_memory_gb": round(result.peak_memory, 2),
                         **score(reference, result.text)})
            print(f"Iteration {iteration}: {seconds:.3f}s; finish={result.finish_reason}", file=sys.stderr)
    report = {"model": args.model, "prompt": args.prompt, "scale": args.scale,
              "image_size": list(image.size), "load_seconds": round(load_seconds, 2),
              "runs": runs}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"model": args.model, "load_seconds": report["load_seconds"], "runs": [
        {k: v for k, v in r.items() if k != "text"} for r in runs]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
