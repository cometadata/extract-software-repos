"""Tests for Polars extraction functionality."""

import polars as pl
import pytest


class TestPolarsExtraction:
    """Verify Polars extraction works correctly."""

    def test_finds_github_urls(self):
        """Extraction finds GitHub URLs."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo for code"]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        assert any("github.com/user/repo" in u["url"] for u in urls)

    def test_finds_pypi_urls(self):
        """Extraction finds PyPI URLs."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Install from https://pypi.org/project/mypackage"]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        assert any("pypi.org/project/mypackage" in u["url"] for u in urls)

    def test_finds_multiple_urls(self):
        """Extraction finds multiple URL types."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = """
        GitHub: https://github.com/org/repo
        PyPI: https://pypi.org/project/pkg
        GitLab: https://gitlab.com/team/project
        """

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert len(urls) == 3

    def test_deduplicates_same_repo(self):
        """Extraction deduplicates same repo with different paths."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = """
        https://github.com/user/repo
        https://github.com/user/repo/releases
        https://github.com/user/repo/blob/main/README.md
        """

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert len(urls) == 1

    def test_handles_pdf_artifacts(self):
        """Extraction handles PDF extraction artifacts."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = "Code at git-\nhub.com / user / repo here"

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert any("github" in u["url"] for u in urls)

    def test_excludes_wiki_paths(self):
        """Extraction filters out wiki paths."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = "See https://github.com/user/repo/wiki for docs"

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert len(urls) == 0

    def test_excludes_issues_paths(self):
        """Extraction filters out issues paths."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = "Report at https://github.com/user/repo/issues/123"

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert len(urls) == 0

    def test_normalizes_to_base_repo(self):
        """Extraction normalizes to base repo URL."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = "https://github.com/user/repo/tree/main/src"

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        result = extract_urls_polars_df(df, id_col="id", content_col="content")
        urls = result["urls"][0]

        assert any(u["url"] == "https://github.com/user/repo" for u in urls)

    def test_preserves_docs_without_urls(self):
        """Extraction preserves documents without URLs."""
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        df = pl.DataFrame({
            "id": ["doc1", "doc2"],
            "content": ["No URLs here", "Check https://github.com/user/repo"]
        })
        result = extract_urls_polars_df(df, id_col="id", content_col="content")

        assert len(result) == 2
        assert len(result["urls"][0]) == 0
        assert len(result["urls"][1]) == 1
