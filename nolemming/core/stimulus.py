"""Stimulus ingestion utilities."""

from __future__ import annotations

from pathlib import Path

from nolemming.core.types import Stimulus


def load_stimulus(path: str | Path) -> Stimulus:
    """Load a stimulus from a file path, inferring type from extension."""
    path = Path(path)
    if not path.exists():
        msg = f"Stimulus file not found: {path}"
        raise FileNotFoundError(msg)
    return Stimulus.from_path(path)


def load_text_stimulus(text: str, output_dir: Path | None = None) -> Stimulus:
    """Create a stimulus from raw text content.

    Writes to a temp file and returns a Stimulus pointing to it.
    """
    import tempfile

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "stimulus.txt"
    else:
        fd, path_str = tempfile.mkstemp(suffix=".txt")
        path = Path(path_str)
        import os
        os.close(fd)

    path.write_text(text, encoding="utf-8")
    return Stimulus.from_path(path)
