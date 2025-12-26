# tests/test_validation_utils.py
"""Tests for validation utility functions."""


from extract_software_repos.validation import (
    categorize_urls,
    deduplicate_urls,
)


class TestCategorizeUrls:
    """Test URL categorization by type."""

    def test_github_url(self):
        category = categorize_urls(["https://github.com/user/repo"])
        assert category["github"] == ["https://github.com/user/repo"]

    def test_mixed_urls(self):
        urls = [
            "https://github.com/user/repo",
            "https://pypi.org/project/requests",
            "https://gitlab.com/user/repo",
        ]
        category = categorize_urls(urls)
        assert len(category["github"]) == 1
        assert len(category["pypi"]) == 1
        assert len(category["gitlab"]) == 1


class TestDeduplicateUrls:
    """Test URL deduplication."""

    def test_removes_duplicates(self):
        records = [
            {"enrichedValue": {"relatedIdentifier": "https://github.com/a/b"}},
            {"enrichedValue": {"relatedIdentifier": "https://github.com/a/b"}},
            {"enrichedValue": {"relatedIdentifier": "https://github.com/c/d"}},
        ]
        unique_urls, url_to_records = deduplicate_urls(records)

        assert len(unique_urls) == 2
        assert len(url_to_records["https://github.com/a/b"]) == 2
        assert len(url_to_records["https://github.com/c/d"]) == 1

    def test_handles_null_urls(self):
        records = [
            {"enrichedValue": {"relatedIdentifier": "https://github.com/a/b"}},
            {"enrichedValue": {"relatedIdentifier": None}},
            {"enrichedValue": {}},
        ]
        unique_urls, url_to_records = deduplicate_urls(records)

        assert len(unique_urls) == 1
