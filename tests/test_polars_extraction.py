"""Tests for Polars-based extraction."""

import pytest
import polars as pl


class TestPolarsPreprocessing:
    """Test Polars preprocessing expressions."""

    def test_removes_soft_hyphens(self):
        from extract_software_repos.polars_extraction import preprocess_content
        df = pl.DataFrame({"content": ["git\u00ADhub.com/user/repo"]})
        result = df.select(preprocess_content("content"))
        assert "github.com" in result["content"][0]

    def test_fixes_hyphenated_linebreaks(self):
        from extract_software_repos.polars_extraction import preprocess_content
        df = pl.DataFrame({"content": ["user-\nname/repo"]})
        result = df.select(preprocess_content("content"))
        assert "username" in result["content"][0]

    def test_collapses_spaces_around_slashes(self):
        from extract_software_repos.polars_extraction import preprocess_content
        df = pl.DataFrame({"content": ["github.com / user / repo"]})
        result = df.select(preprocess_content("content"))
        assert "github.com/user/repo" in result["content"][0]

    def test_fixes_spaced_github_domain(self):
        from extract_software_repos.polars_extraction import preprocess_content
        df = pl.DataFrame({"content": ["g i t h u b . c o m/user/repo"]})
        result = df.select(preprocess_content("content"))
        assert "github.com" in result["content"][0]
