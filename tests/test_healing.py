"""Tests for markdown healing module."""

import pytest


class TestUrlProtection:
    """Tests for URL protection during word recovery."""

    def test_protect_urls_finds_https_urls(self):
        from extract_software_repos.healing import protect_urls

        text = "Check https://github.com/user/repo for code"
        protected, placeholders = protect_urls(text)

        assert "https://github.com/user/repo" not in protected
        assert len(placeholders) == 1
        assert "https://github.com/user/repo" in placeholders.values()

    def test_protect_urls_finds_domain_without_protocol(self):
        from extract_software_repos.healing import protect_urls

        text = "See github.com/user/repo for details"
        protected, placeholders = protect_urls(text)

        assert "github.com/user/repo" not in protected
        assert len(placeholders) == 1

    def test_restore_urls_restores_placeholders(self):
        from extract_software_repos.healing import protect_urls, restore_urls

        text = "Check https://github.com/user/repo here"
        protected, placeholders = protect_urls(text)
        restored = restore_urls(protected, placeholders)

        assert restored == text

    def test_protect_urls_handles_multiple_urls(self):
        from extract_software_repos.healing import protect_urls

        text = "Use https://github.com/a/b and https://pypi.org/project/pkg"
        protected, placeholders = protect_urls(text)

        assert len(placeholders) == 2
        assert "https://github.com/a/b" not in protected
        assert "https://pypi.org/project/pkg" not in protected


class TestInputValidation:
    """Tests for input quality validation."""

    def test_validate_empty_text_fails(self):
        from extract_software_repos.healing import validate_input_quality

        is_valid, reason = validate_input_quality("")
        assert not is_valid
        assert "empty" in reason.lower()

    def test_validate_normal_text_passes(self):
        from extract_software_repos.healing import validate_input_quality

        text = "This is a normal paragraph with spaces and punctuation."
        is_valid, reason = validate_input_quality(text)
        assert is_valid

    def test_validate_low_whitespace_fails(self):
        from extract_software_repos.healing import validate_input_quality

        # Text with almost no whitespace
        text = "thisisalongstringwithnospacesatallwhichismalformed" * 10
        is_valid, reason = validate_input_quality(text)
        assert not is_valid
        assert "whitespace" in reason.lower()

    def test_validate_very_long_lines_fails(self):
        from extract_software_repos.healing import validate_input_quality

        # Many lines over 1000 chars
        text = "\n".join(["x" * 1500 for _ in range(10)])
        is_valid, reason = validate_input_quality(text)
        assert not is_valid
        assert "long" in reason.lower()


class TestCoreHealing:
    """Tests for core healing functions."""

    def test_fix_hyphenation_rejoins_words(self):
        from extract_software_repos.healing import fix_hyphenation

        text = "This is a hyph-\nenated word"
        result = fix_hyphenation(text)
        assert result == "This is a hyphenated word"

    def test_normalize_bullets_converts_fancy_bullets(self):
        from extract_software_repos.healing import normalize_bullets

        text = "• First item\n● Second item\n◦ Third item"
        result = normalize_bullets(text)
        assert result == "- First item\n- Second item\n- Third item"

    def test_collapse_blank_lines_limits_to_two(self):
        from extract_software_repos.healing import collapse_blank_lines

        text = "Para 1\n\n\n\n\nPara 2"
        result = collapse_blank_lines(text)
        assert result == "Para 1\n\n\nPara 2"

    def test_strip_repeated_lines_removes_headers(self):
        from extract_software_repos.healing import strip_repeated_lines

        # Simulate repeated header appearing on every "page"
        lines = []
        for i in range(5):
            lines.append("Page Header Text")
            lines.extend([f"Content line {i*10 + j}" for j in range(10)])

        text = "\n".join(lines)
        result, removed = strip_repeated_lines(
            text, page_length_estimate=10, threshold=0.8, max_words=12
        )

        assert "Page Header Text" not in result
        assert "Page Header Text" in removed


class TestLineMerging:
    """Tests for broken line merging."""

    def test_merge_broken_lines_joins_fragments(self):
        from extract_software_repos.healing import merge_broken_lines

        text = "This is a sentence that was\nbroken across multiple\nlines incorrectly."
        result = merge_broken_lines(text)
        assert "that was broken across" in result

    def test_merge_broken_lines_preserves_paragraphs(self):
        from extract_software_repos.healing import merge_broken_lines

        text = "First paragraph here.\n\nSecond paragraph here."
        result = merge_broken_lines(text)
        # Should preserve paragraph break
        assert "\n\n" in result or "\n" in result
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_merge_broken_lines_preserves_list_items(self):
        from extract_software_repos.healing import merge_broken_lines

        text = "- First item\n- Second item\n- Third item"
        result = merge_broken_lines(text)
        assert "- First item" in result
        assert "- Second item" in result

    def test_merge_broken_lines_preserves_code_blocks(self):
        from extract_software_repos.healing import merge_broken_lines

        text = "Text before\n```python\ndef foo():\n    pass\n```\nText after"
        result = merge_broken_lines(text)
        assert "```python" in result
        assert "def foo():" in result


class TestHealText:
    """Tests for main heal_text function."""

    def test_heal_text_returns_tuple(self):
        from extract_software_repos.healing import heal_text

        text = "Normal paragraph text here."
        result, warnings = heal_text(text)

        assert isinstance(result, str)
        assert isinstance(warnings, list)

    def test_heal_text_fixes_hyphenation(self):
        from extract_software_repos.healing import heal_text

        text = "This word was hyphen-\nated incorrectly."
        result, warnings = heal_text(text)

        assert "hyphenated" in result

    def test_heal_text_preserves_urls(self):
        from extract_software_repos.healing import heal_text

        text = "Check https://github.com/user/repo for code"
        result, warnings = heal_text(text)

        assert "https://github.com/user/repo" in result

    def test_heal_text_returns_warnings_for_issues(self):
        from extract_software_repos.healing import heal_text

        # Text with repeated headers
        lines = []
        for i in range(5):
            lines.append("Repeated Header")
            lines.extend([f"Content {j}" for j in range(10)])

        text = "\n".join(lines)
        result, warnings = heal_text(text)

        assert len(warnings) > 0

    def test_heal_text_handles_empty_gracefully(self):
        from extract_software_repos.healing import heal_text

        result, warnings = heal_text("")
        assert result == ""
        assert len(warnings) > 0


import tempfile
from pathlib import Path
import polars as pl


class TestParallelHealing:
    """Test parallel healing with multiprocessing."""

    def test_heals_parquet_parallel(self):
        from extract_software_repos.healing import heal_parquet_parallel

        df = pl.DataFrame({
            "id": ["doc1", "doc2"],
            "content": [
                "This is some text with git-\nhub.com/user/repo link",
                "Another doc with py-\npi.org/project/pkg"
            ]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"
            df.write_parquet(input_path)

            stats = heal_parquet_parallel(
                input_path,
                output_path,
                content_field="content",
                workers=2,
            )

            assert stats["total"] == 2
            assert output_path.exists()

            result = pl.read_parquet(output_path)
            # Check hyphenation was fixed
            assert "github.com" in result["content"][0]
            assert "pypi.org" in result["content"][1]

    def test_preserves_all_columns(self):
        from extract_software_repos.healing import heal_parquet_parallel

        df = pl.DataFrame({
            "id": ["doc1"],
            "content": ["Some text"],
            "extra_col": ["preserved"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"
            df.write_parquet(input_path)

            heal_parquet_parallel(input_path, output_path, content_field="content")

            result = pl.read_parquet(output_path)
            assert "extra_col" in result.columns
            assert result["extra_col"][0] == "preserved"
