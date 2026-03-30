#!/usr/bin/env python3
"""Utility for generating token embeddings from Wernicke's Area."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Resolve the base directory of the ``brain`` package so relative paths work
# regardless of the current working directory.
BASE_DIR = Path(__file__).resolve().parents[2]


def generate(
    model_dir: str | Path,
    output: Path,
    device: str = "cuda",
    batch_size: int = 1024,
) -> None:
    """Create a table of embeddings for every tokenizer token."""
    model_path = Path(model_dir)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    output_path = output
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    if device.startswith("cuda"):
        device_ids = [0, 1, 2, 3]
        if torch.cuda.device_count() < len(device_ids):
            device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            # Move the model to the primary CUDA device before wrapping in
            # ``DataParallel`` so the parameters are replicated correctly.
            primary_device = f"cuda:{device_ids[0]}"
            model.to(primary_device)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            device = primary_device
        else:
            model.to(device)
    else:
        model.to(device)
    model.eval()

    # ``DataParallel`` wraps the model and hides the ``config`` attribute on the
    # outer object. The underlying module retains the original ``config`` so we
    # access it there when present.
    cfg = model.module.config if hasattr(model, "module") else model.config
    emb_dim = getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 768))
    embeddings = np.zeros((len(tokenizer), 768), dtype=np.float32)
    proj = None
    if emb_dim != 768:
        proj = torch.nn.Linear(emb_dim, 768, bias=False, device=device)
    tokens = [tokenizer.decode([i], skip_special_tokens=False) for i in range(len(tokenizer))]

    with torch.no_grad():
        for start in tqdm(range(0, len(tokenizer), batch_size), desc="Embedding tokens"):
            end = min(start + batch_size, len(tokenizer))
            ids = torch.arange(start, end, device=device).unsqueeze(1)
            out = model(input_ids=ids)
            emb = out.last_hidden_state[:, 0, :]
            if proj is not None:
                emb = proj(emb)
            embeddings[start:end] = emb.cpu().numpy().astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {"tokens": tokens, "embeddings": embeddings}, allow_pickle=True)
    print(f"saved {len(tokens)} embeddings to {output_path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate Wernicke token table")
    parser.add_argument("--model_dir", default="models/gpt2")
    # Store generated token embeddings in the repo's persistent directory by
    # default.  ``BASE_DIR`` already points at ``brain/``, so we only append the
    # ``persistent`` folder here to avoid creating ``brain/brain`` paths when
    # running from the repository root.
    parser.add_argument("--output", default="persistent/token_embeddings.npy")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args(argv)

    generate(args.model_dir, Path(args.output), device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

