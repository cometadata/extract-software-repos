"""Tests for Polars-based extraction."""

import tempfile
from pathlib import Path

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


class TestNormalization:
    """Test URL normalization."""

    def test_normalize_urls_exists(self):
        from extract_software_repos.polars_extraction import normalize_urls
        assert callable(normalize_urls)

    def test_normalize_adds_https(self):
        import polars as pl
        from extract_software_repos.polars_extraction import normalize_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["github.com/user/repo"],
            "type": ["github"],
        })
        result = normalize_urls(df)
        assert result["url"][0].startswith("https://")

    def test_normalize_strips_git_suffix(self):
        import polars as pl
        from extract_software_repos.polars_extraction import normalize_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["https://github.com/user/repo.git"],
            "type": ["github"],
        })
        result = normalize_urls(df)
        assert not result["url"][0].endswith(".git")

    def test_normalize_extracts_user_repo(self):
        import polars as pl
        from extract_software_repos.polars_extraction import normalize_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["https://github.com/user/repo/blob/main/file.py"],
            "type": ["github"],
        })
        result = normalize_urls(df)
        assert result["url"][0] == "https://github.com/user/repo"

    def test_normalize_strips_trailing_punctuation(self):
        import polars as pl
        from extract_software_repos.polars_extraction import normalize_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["https://github.com/user/repo)."],
            "type": ["github"],
        })
        result = normalize_urls(df)
        assert not result["url"][0].endswith(").")


class TestFiltering:
    """Test URL filtering."""

    def test_filter_urls_exists(self):
        from extract_software_repos.polars_extraction import filter_urls
        assert callable(filter_urls)

    def test_filter_removes_wiki_urls(self):
        import polars as pl
        from extract_software_repos.polars_extraction import filter_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://github.com/user/repo/wiki/Page"],
            "type": ["github", "github"],
        })
        result = filter_urls(df)
        assert len(result) == 1
        assert "wiki" not in result["url"][0]

    def test_filter_removes_issues_urls(self):
        import polars as pl
        from extract_software_repos.polars_extraction import filter_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://github.com/user/repo/issues/123"],
            "type": ["github", "github"],
        })
        result = filter_urls(df)
        assert len(result) == 1
        assert "issues" not in result["url"][0]

    def test_filter_removes_github_pages(self):
        import polars as pl
        from extract_software_repos.polars_extraction import filter_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://user.github.io/project"],
            "type": ["github", "github"],
        })
        result = filter_urls(df)
        assert len(result) == 1
        assert "github.io" not in result["url"][0]

    def test_filter_removes_data_files_in_blob(self):
        import polars as pl
        from extract_software_repos.polars_extraction import filter_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://github.com/user/repo/blob/main/data.csv"],
            "type": ["github", "github"],
        })
        result = filter_urls(df)
        assert len(result) == 1
        assert "data.csv" not in result["url"][0]

    def test_filter_keeps_valid_urls(self):
        import polars as pl
        from extract_software_repos.polars_extraction import filter_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["https://github.com/user/repo"],
            "type": ["github"],
        })
        result = filter_urls(df)
        assert len(result) == 1


class TestDeduplication:
    """Test URL deduplication."""

    def test_deduplicate_urls_exists(self):
        from extract_software_repos.polars_extraction import deduplicate_urls
        assert callable(deduplicate_urls)

    def test_deduplicate_removes_exact_duplicates(self):
        import polars as pl
        from extract_software_repos.polars_extraction import deduplicate_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://github.com/user/repo"],
            "type": ["github", "github"],
        })
        result = deduplicate_urls(df)
        assert len(result) == 1

    def test_deduplicate_keeps_different_repos(self):
        import polars as pl
        from extract_software_repos.polars_extraction import deduplicate_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo1", "https://github.com/user/repo2"],
            "type": ["github", "github"],
        })
        result = deduplicate_urls(df)
        assert len(result) == 2

    def test_deduplicate_per_document(self):
        import polars as pl
        from extract_software_repos.polars_extraction import deduplicate_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "url": ["https://github.com/user/repo", "https://github.com/user/repo"],
            "type": ["github", "github"],
        })
        result = deduplicate_urls(df)
        # Same URL in different docs should both be kept
        assert len(result) == 2


class TestRegrouping:
    """Test URL regrouping."""

    def test_regroup_urls_exists(self):
        from extract_software_repos.polars_extraction import regroup_urls
        assert callable(regroup_urls)

    def test_regroup_creates_list_column(self):
        import polars as pl
        from extract_software_repos.polars_extraction import regroup_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1"],
            "url": ["https://github.com/user/repo", "https://pypi.org/project/pkg"],
            "type": ["github", "pypi"],
        })
        result = regroup_urls(df)
        assert "urls" in result.columns
        assert result["urls"].dtype == pl.List

    def test_regroup_one_row_per_doc(self):
        import polars as pl
        from extract_software_repos.polars_extraction import regroup_urls

        df = pl.DataFrame({
            "doc_id": ["doc1", "doc1", "doc2"],
            "url": ["https://github.com/user/repo", "https://pypi.org/project/pkg", "https://gitlab.com/team/proj"],
            "type": ["github", "pypi", "gitlab"],
        })
        result = regroup_urls(df)
        assert len(result) == 2

    def test_regroup_preserves_url_and_type(self):
        import polars as pl
        from extract_software_repos.polars_extraction import regroup_urls

        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "url": ["https://github.com/user/repo"],
            "type": ["github"],
        })
        result = regroup_urls(df)
        urls_list = result["urls"][0]
        assert len(urls_list) == 1
        assert urls_list[0]["url"] == "https://github.com/user/repo"
        assert urls_list[0]["type"] == "github"


class TestNativePipeline:
    """Test complete native extraction pipeline."""

    def test_extract_urls_polars_native_exists(self):
        from extract_software_repos.polars_extraction import extract_urls_polars_native
        assert callable(extract_urls_polars_native)

    def test_pipeline_extracts_github(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_polars_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Check https://github.com/user/repo for code"]
        })
        result = extract_urls_polars_native(df, id_col="id", content_col="content")
        assert len(result) == 1
        urls = result["urls"][0]
        assert any("github.com/user/repo" in u["url"] for u in urls)

    def test_pipeline_filters_exclusions(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_polars_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["https://github.com/user/repo and https://github.com/user/repo/wiki/Page"]
        })
        result = extract_urls_polars_native(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        # Wiki should be filtered out
        assert not any("wiki" in u["url"] for u in urls)

    def test_pipeline_deduplicates(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_polars_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["https://github.com/user/repo and https://github.com/user/repo/releases"]
        })
        result = extract_urls_polars_native(df, id_col="id", content_col="content")
        urls = result["urls"][0]
        github_urls = [u for u in urls if u["type"] == "github"]
        assert len(github_urls) == 1

    def test_pipeline_handles_empty(self):
        import polars as pl
        from extract_software_repos.polars_extraction import extract_urls_polars_native

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["No URLs here at all"]
        })
        result = extract_urls_polars_native(df, id_col="id", content_col="content")
        assert len(result) == 1
        assert len(result["urls"][0]) == 0
