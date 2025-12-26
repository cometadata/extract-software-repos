# tests/test_checkpoint.py
"""Tests for checkpoint/resume functionality."""

import json
import pytest
from pathlib import Path

from extract_software_repos.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test checkpoint save/load functionality."""

    def test_load_empty_checkpoint(self, tmp_path):
        """Loading non-existent checkpoint returns empty dict."""
        cp = CheckpointManager(tmp_path / "cache.jsonl")
        assert cp.get_cached_urls() == {}

    def test_save_and_load_result(self, tmp_path):
        """Can save a result and load it back."""
        cp_path = tmp_path / "cache.jsonl"
        cp = CheckpointManager(cp_path)

        cp.save_result("https://github.com/user/repo", True, "graphql", None)

        # Reload from disk
        cp2 = CheckpointManager(cp_path)
        cached = cp2.get_cached_urls()

        assert "https://github.com/user/repo" in cached
        assert cached["https://github.com/user/repo"]["valid"] is True

    def test_save_batch_results(self, tmp_path):
        """Can save multiple results at once."""
        cp_path = tmp_path / "cache.jsonl"
        cp = CheckpointManager(cp_path)

        results = [
            {"url": "https://github.com/a/b", "valid": True, "method": "graphql", "error": None},
            {"url": "https://github.com/c/d", "valid": False, "method": "graphql", "error": "not_found"},
        ]
        cp.save_batch(results)

        cp2 = CheckpointManager(cp_path)
        cached = cp2.get_cached_urls()

        assert len(cached) == 2
        assert cached["https://github.com/a/b"]["valid"] is True
        assert cached["https://github.com/c/d"]["valid"] is False
