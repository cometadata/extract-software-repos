# src/extract_software_repos/checkpoint.py
"""Checkpoint manager for resumable validation."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class CheckpointManager:
    """Manages checkpoint file for resumable validation."""

    def __init__(self, checkpoint_path: Path):
        self.path = Path(checkpoint_path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing checkpoint from disk."""
        if not self.path.exists():
            return

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    # Last entry for a URL wins
                    self._cache[entry["url"]] = entry

    def get_cached_urls(self) -> Dict[str, Dict[str, Any]]:
        """Return dict of URL -> validation result."""
        return self._cache.copy()

    def is_cached(self, url: str) -> bool:
        """Check if URL has already been validated."""
        return url in self._cache

    def save_result(
        self,
        url: str,
        valid: bool,
        method: str,
        error: Optional[str],
    ) -> None:
        """Save a single validation result."""
        entry = {
            "url": url,
            "valid": valid,
            "method": method,
            "error": error,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        self._cache[url] = entry
        self._append_to_file(entry)

    def save_batch(self, results: List[Dict[str, Any]]) -> None:
        """Save multiple validation results."""
        timestamp = datetime.now(timezone.utc).isoformat()
        entries = []

        for r in results:
            entry = {
                "url": r["url"],
                "valid": r["valid"],
                "method": r["method"],
                "error": r.get("error"),
                "checked_at": timestamp,
            }
            self._cache[r["url"]] = entry
            entries.append(entry)

        self._append_batch_to_file(entries)

    def _append_to_file(self, entry: Dict[str, Any]) -> None:
        """Append single entry to checkpoint file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _append_batch_to_file(self, entries: List[Dict[str, Any]]) -> None:
        """Append multiple entries to checkpoint file."""
        with open(self.path, "a", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
