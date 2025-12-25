"""Parity tests to verify Polars extraction matches original implementation."""

import tempfile
from pathlib import Path
import polars as pl
import pytest


class TestExtractionParity:
    """Verify Polars extraction produces same results as original."""

    def test_github_url_parity(self):
        """Same GitHub URLs extracted by both implementations."""
        from extract_software_repos.extraction import extract_urls_with_types
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = "Check https://github.com/user/repo for code"

        # Original
        original = extract_urls_with_types(text)
        original_urls = {u["url"] for u in original}

        # Polars
        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        polars_result = extract_urls_polars_df(df, id_col="id", content_col="content")
        polars_urls = {u["url"] for u in polars_result["urls"][0]}

        assert original_urls == polars_urls

    def test_multiple_url_parity(self):
        """Same multiple URLs extracted by both implementations."""
        from extract_software_repos.extraction import extract_urls_with_types
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = """
        GitHub: https://github.com/org/repo
        PyPI: https://pypi.org/project/pkg
        GitLab: https://gitlab.com/team/project
        """

        original = extract_urls_with_types(text)
        original_urls = {u["url"] for u in original}

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        polars_result = extract_urls_polars_df(df, id_col="id", content_col="content")
        polars_urls = {u["url"] for u in polars_result["urls"][0]}

        assert original_urls == polars_urls

    def test_deduplication_parity(self):
        """Same deduplication behavior in both implementations."""
        from extract_software_repos.extraction import extract_urls_with_types
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        text = """
        https://github.com/user/repo
        https://github.com/user/repo/releases
        https://github.com/user/repo/blob/main/README.md
        """

        original = extract_urls_with_types(text)

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        polars_result = extract_urls_polars_df(df, id_col="id", content_col="content")
        polars_urls = polars_result["urls"][0]

        # Both should deduplicate to single repo URL
        assert len(original) == len(polars_urls)

    def test_preprocessing_parity(self):
        """Same preprocessing fixes in both implementations."""
        from extract_software_repos.extraction import extract_urls_with_types
        from extract_software_repos.polars_extraction import extract_urls_polars_df

        # Text with PDF artifacts
        text = "Code at git-\nhub.com / user / repo here"

        original = extract_urls_with_types(text)

        df = pl.DataFrame({"id": ["doc1"], "content": [text]})
        polars_result = extract_urls_polars_df(df, id_col="id", content_col="content")
        polars_urls = polars_result["urls"][0]

        # Both should find the GitHub URL after preprocessing
        original_has_github = any("github" in u["url"] for u in original)
        polars_has_github = any("github" in u["url"] for u in polars_urls)

        assert original_has_github == polars_has_github
