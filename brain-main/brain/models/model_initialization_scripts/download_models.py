#!/usr/bin/env python3
"""Download necessary base models for brain.

This utility fetches the Hugging Face model checkpoints required for
bootstrapping brain's brain. Models are downloaded sequentially into
the ``models`` directory so they can be loaded offline by the various
brain region services.
"""

from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download

# Mapping from local sub-directory name to Hugging Face repository id
MODELS: Dict[str, str] = {
    "clip-vit-b32": "openai/clip-vit-base-patch32",
    "whisper-small": "openai/whisper-small",
    "gpt2": "gpt2",
    "bert-base-uncased": "bert-base-uncased",
}


def download_model(repo_id: str, target_dir: Path) -> None:
    """Download ``repo_id`` into ``target_dir`` if needed.

    Directories that already contain files are considered complete and will be
    skipped. Empty directories (which sometimes exist due to manual creation or
    aborted downloads) will be populated.
    """

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[skip] {target_dir} already present")
        return

    if target_dir.exists():
        print(f"[download] {repo_id} -> {target_dir} (was empty)")
    else:
        print(f"[download] {repo_id} -> {target_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )


def download_all(root: Path) -> None:
    """Download all models into ``root`` sequentially."""
    root.mkdir(parents=True, exist_ok=True)
    for subdir, repo_id in MODELS.items():
        download_model(repo_id, root / subdir)


def main() -> None:
    # Determine the models directory relative to this script
    models_root = Path(__file__).resolve().parents[1]
    download_all(models_root)


if __name__ == "__main__":
    main()
