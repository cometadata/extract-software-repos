# tests/test_github_graphql.py
"""Tests for GitHub GraphQL batch validation."""


from extract_software_repos.github_graphql import (
    parse_github_url,
    build_graphql_query,
)


class TestParseGitHubUrl:
    """Test GitHub URL parsing."""

    def test_simple_url(self):
        owner, repo = parse_github_url("https://github.com/pytorch/pytorch")
        assert owner == "pytorch"
        assert repo == "pytorch"

    def test_url_with_trailing_slash(self):
        owner, repo = parse_github_url("https://github.com/user/repo/")
        assert owner == "user"
        assert repo == "repo"

    def test_url_with_git_suffix(self):
        owner, repo = parse_github_url("https://github.com/user/repo.git")
        assert owner == "user"
        assert repo == "repo"

    def test_url_with_subpath(self):
        owner, repo = parse_github_url("https://github.com/user/repo/tree/main/src")
        assert owner == "user"
        assert repo == "repo"

    def test_invalid_url_returns_none(self):
        result = parse_github_url("https://gitlab.com/user/repo")
        assert result is None

    def test_malformed_url_returns_none(self):
        result = parse_github_url("https://github.com/onlyowner")
        assert result is None


class TestBuildGraphQLQuery:
    """Test GraphQL query building."""

    def test_single_repo_query(self):
        repos = [("pytorch", "pytorch")]
        query = build_graphql_query(repos)
        assert "repo0: repository(owner: \"pytorch\", name: \"pytorch\")" in query
        assert "{ id }" in query

    def test_multiple_repos_query(self):
        repos = [("user1", "repo1"), ("user2", "repo2")]
        query = build_graphql_query(repos)
        assert "repo0: repository(owner: \"user1\", name: \"repo1\")" in query
        assert "repo1: repository(owner: \"user2\", name: \"repo2\")" in query
