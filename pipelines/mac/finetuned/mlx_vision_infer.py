#!/usr/bin/env python3
"""Subprocess-helper voor één MLX vision-inferentie."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    model, processor = load(args.model)
    config = load_config(args.model)
    formatted = apply_chat_template(processor, config, prompt, num_images=1)
    output = generate(
        model,
        processor,
        formatted,
        image=args.image,
        max_tokens=args.max_tokens,
        temp=0.1,
        verbose=False,
    )
    print(output.text if hasattr(output, "text") else str(output))


if __name__ == "__main__":
    main()
