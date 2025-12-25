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


class TestPolarsExtraction:
    """Test Polars URL extraction."""

    def test_extracts_github_url(self):
        from extract_software_repos.polars_extraction import extract_urls_polars_df
        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo for code"]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        assert len(result) == 1
        assert result["id"][0] == "doc1"
        urls = result["urls"][0]
        assert any("github.com/user/repo" in u["url"] for u in urls)

    def test_extracts_multiple_urls(self):
        from extract_software_repos.polars_extraction import extract_urls_polars_df
        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["""
                GitHub: https://github.com/org/repo
                PyPI: https://pypi.org/project/pkg
            """]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        assert len(urls) >= 2

    def test_returns_empty_for_no_urls(self):
        from extract_software_repos.polars_extraction import extract_urls_polars_df
        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["No URLs in this text at all."]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        assert len(result) == 1
        assert len(result["urls"][0]) == 0

    def test_deduplicates_same_repo(self):
        from extract_software_repos.polars_extraction import extract_urls_polars_df
        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["""
                https://github.com/user/repo
                https://github.com/user/repo/releases
                https://github.com/user/repo/blob/main/README.md
            """]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        # Should deduplicate to single repo
        github_urls = [u for u in urls if "github" in u["type"]]
        assert len(github_urls) == 1
