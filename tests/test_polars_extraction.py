"""Tests for Polars-based extraction."""

import tempfile
from pathlib import Path

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


class TestPolarsParquetProcessing:
    """Test Polars parquet processing."""

    def test_processes_parquet_file(self):
        from extract_software_repos.polars_extraction import process_parquet_polars

        # Create test parquet
        df = pl.DataFrame({
            "relative_path": ["2308.11197v1.md", "2309.12345v2.md"],
            "content": [
                "Code at https://github.com/user/repo",
                "Package: https://pypi.org/project/mylib"
            ]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "output.jsonl"
            df.write_parquet(input_path)

            stats = process_parquet_polars(
                input_path,
                output_path,
                id_field="relative_path",
                content_field="content",
            )

            assert stats["total_papers"] == 2
            assert stats["papers_with_urls"] == 2
            assert output_path.exists()

    def test_processes_with_progress(self):
        from extract_software_repos.polars_extraction import process_parquet_polars

        df = pl.DataFrame({
            "relative_path": ["2308.11197v1.md"],
            "content": ["https://github.com/user/repo"]
        })

        progress_calls = []
        def callback(processed, with_urls, total):
            progress_calls.append((processed, with_urls, total))

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "output.jsonl"
            df.write_parquet(input_path)

            process_parquet_polars(
                input_path,
                output_path,
                progress_callback=callback,
            )

            assert len(progress_calls) >= 1


class TestPolarsPatterns:
    """Test Polars URL patterns."""

    def test_patterns_dict_exists(self):
        from extract_software_repos.polars_extraction import POLARS_URL_PATTERNS
        assert isinstance(POLARS_URL_PATTERNS, dict)
        assert "github" in POLARS_URL_PATTERNS
        assert "pypi" in POLARS_URL_PATTERNS

    def test_pattern_matches_github(self):
        import re
        from extract_software_repos.polars_extraction import POLARS_URL_PATTERNS
        pattern = POLARS_URL_PATTERNS["github"]
        match = re.search(pattern, "see https://github.com/user/repo for code")
        assert match is not None
        assert "github.com/user/repo" in match.group(0)


class TestFilterPatterns:
    """Test filter pattern constants."""

    def test_excluded_paths_pattern_exists(self):
        from extract_software_repos.polars_extraction import EXCLUDED_PATHS_PATTERN
        assert isinstance(EXCLUDED_PATHS_PATTERN, str)

    def test_excluded_paths_matches_wiki(self):
        import re
        from extract_software_repos.polars_extraction import EXCLUDED_PATHS_PATTERN
        assert re.search(EXCLUDED_PATHS_PATTERN, "github.com/user/repo/wiki/Page")

    def test_excluded_paths_matches_issues(self):
        import re
        from extract_software_repos.polars_extraction import EXCLUDED_PATHS_PATTERN
        assert re.search(EXCLUDED_PATHS_PATTERN, "github.com/user/repo/issues/123")

    def test_data_extensions_pattern_exists(self):
        from extract_software_repos.polars_extraction import DATA_EXTENSIONS_PATTERN
        assert isinstance(DATA_EXTENSIONS_PATTERN, str)

    def test_data_extensions_matches_csv(self):
        import re
        from extract_software_repos.polars_extraction import DATA_EXTENSIONS_PATTERN
        assert re.search(DATA_EXTENSIONS_PATTERN, "file.csv")


class TestNativeExtraction:
    """Test Polars-native URL extraction."""

    def test_extract_urls_native_exists(self):
        from extract_software_repos.polars_extraction import extract_urls_native
        assert callable(extract_urls_native)

    def test_extract_urls_native_returns_dataframe(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo"]
        })
        result = extract_urls_native(df, id_col="id", content_col="content")
        assert isinstance(result, pl.DataFrame)

    def test_extract_urls_native_has_correct_columns(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo"]
        })
        result = extract_urls_native(df, id_col="id", content_col="content")
        assert "doc_id" in result.columns
        assert "url" in result.columns
        assert "type" in result.columns

    def test_extract_urls_native_extracts_github(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo for code"]
        })
        result = extract_urls_native(df, id_col="id", content_col="content")
        assert len(result) >= 1
        assert any("github.com/user/repo" in url for url in result["url"].to_list())
