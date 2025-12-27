"""Tests for GitHub GraphQL promotion data fetching."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from extract_software_repos.github_graphql import (
    build_promotion_query,
    GitHubPromotionData,
    GitHubPromotionFetcher,
    parse_github_url,
)


class TestBuildPromotionQuery:
    """Tests for building promotion GraphQL queries."""

    def test_single_repo(self):
        query = build_promotion_query([("owner", "repo")])
        assert "repository(owner:" in query
        assert "description" in query
        assert "README.md" in query
        assert "mentionableUsers" in query

    def test_multiple_repos(self):
        query = build_promotion_query([("a", "b"), ("c", "d")])
        assert "repo0:" in query
        assert "repo1:" in query

    def test_readme_variants(self):
        query = build_promotion_query([("o", "r")])
        # Should try multiple README filenames
        assert "README.md" in query


class TestGitHubPromotionData:
    """Tests for promotion data structure."""

    def test_promotion_data_has_exists_property(self):
        """GitHubPromotionData should indicate existence based on fetch_error."""
        # Successful fetch = exists
        data = GitHubPromotionData(url="https://github.com/user/repo", description="test")
        assert data.exists is True

        # fetch_error = not_found means doesn't exist
        data_not_found = GitHubPromotionData(url="https://github.com/user/repo", fetch_error="not_found")
        assert data_not_found.exists is False

        # Other fetch errors still mean doesn't exist (for validation purposes)
        data_error = GitHubPromotionData(url="https://github.com/user/repo", fetch_error="request_failed")
        assert data_error.exists is False

    def test_creation(self):
        data = GitHubPromotionData(
            url="https://github.com/owner/repo",
            description="A cool project",
            readme_content="# README\nSee paper 2308.11197",
            contributors=[
                {"login": "user1", "name": "User One", "email": "u1@example.com"}
            ],
        )
        assert data.url == "https://github.com/owner/repo"
        assert data.description == "A cool project"
        assert "2308.11197" in data.readme_content
        assert len(data.contributors) == 1

    def test_optional_fields(self):
        data = GitHubPromotionData(url="https://github.com/o/r")
        assert data.description is None
        assert data.readme_content is None
        assert data.contributors == []


class TestGitHubPromotionFetcher:
    """Tests for the promotion data fetcher."""

    def test_init_requires_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token required"):
                GitHubPromotionFetcher(token=None)

    def test_init_with_token(self):
        fetcher = GitHubPromotionFetcher(token="test_token")
        assert fetcher.token == "test_token"

    def test_parse_response_extracts_data(self):
        fetcher = GitHubPromotionFetcher(token="test")

        response_data = {
            "data": {
                "repo0": {
                    "description": "Test description",
                    "readme_md": {"text": "# README"},
                    "readme_rst": None,
                    "mentionableUsers": {
                        "nodes": [
                            {"login": "user1", "name": "User", "email": "u@e.com"}
                        ]
                    }
                }
            }
        }

        urls = ["https://github.com/owner/repo"]
        results = fetcher._parse_promotion_response(urls, response_data)

        assert len(results) == 1
        assert results[0].description == "Test description"
        assert results[0].readme_content == "# README"
        assert len(results[0].contributors) == 1

    def test_parse_response_handles_missing_readme(self):
        fetcher = GitHubPromotionFetcher(token="test")

        response_data = {
            "data": {
                "repo0": {
                    "description": "Test",
                    "readme_md": None,
                    "readme_rst": None,
                    "readme_txt": None,
                    "readme_plain": None,
                    "mentionableUsers": {"nodes": []}
                }
            }
        }

        urls = ["https://github.com/owner/repo"]
        results = fetcher._parse_promotion_response(urls, response_data)

        assert len(results) == 1
        assert results[0].readme_content is None
