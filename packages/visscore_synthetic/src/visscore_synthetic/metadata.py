"""JSONL metadata writer for generated images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


class MetadataWriter:
    """Append one JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: TextIO | None = None

    def __enter__(self) -> MetadataWriter:
        self._fp = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fp is None:
            raise RuntimeError("MetadataWriter must be used as context manager")
        self._fp.write(json.dumps(row, sort_keys=True) + "\n")
