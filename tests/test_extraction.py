"""Tests for DataCite URL extraction."""


from extract_software_repos.extraction import (
    extract_software_urls,
    preprocess_text,
    extract_base_repo,
    is_duplicate,
)


class TestExtractSoftwareUrls:
    """Test extract_software_urls function."""

    def test_extracts_github_url(self):
        text = "Check out https://github.com/user/repo for details."
        urls = extract_software_urls(text)
        assert "https://github.com/user/repo" in urls

    def test_extracts_pypi_url(self):
        text = "Install via https://pypi.org/project/mypackage"
        urls = extract_software_urls(text)
        assert "https://pypi.org/project/mypackage" in urls

    def test_extracts_multiple_urls(self):
        text = """
        Code at https://github.com/org/project
        Package: https://pypi.org/project/mylib
        Also on https://gitlab.com/org/other
        """
        urls = extract_software_urls(text)
        assert len(urls) == 3

    def test_excludes_wiki_paths(self):
        text = "See https://github.com/user/repo/wiki for docs"
        urls = extract_software_urls(text)
        assert len(urls) == 0

    def test_excludes_issues_paths(self):
        text = "Report at https://github.com/user/repo/issues/123"
        urls = extract_software_urls(text)
        assert len(urls) == 0

    def test_normalizes_to_base_repo(self):
        text = "https://github.com/user/repo/tree/main/src"
        urls = extract_software_urls(text)
        assert "https://github.com/user/repo" in urls

    def test_deduplicates_same_repo(self):
        text = """
        https://github.com/user/repo
        https://github.com/user/repo/releases
        https://github.com/user/repo/blob/main/README.md
        """
        urls = extract_software_urls(text)
        assert len(urls) == 1


class TestPreprocessText:
    """Test text preprocessing for PDF artifacts."""

    def test_fixes_spaced_domain(self):
        text = "g i t h u b . c o m/user/repo"
        result = preprocess_text(text)
        assert "github.com" in result

    def test_fixes_hyphenated_linebreak(self):
        text = "user-\nname/repo"
        result = preprocess_text(text)
        assert "username" in result

    def test_removes_soft_hyphens(self):
        text = "git\u00ADhub.com"
        result = preprocess_text(text)
        assert "github.com" in result

    def test_collapses_spaces_around_slashes(self):
        text = "github.com / user / repo"
        result = preprocess_text(text)
        assert "github.com/user/repo" in result


class TestExtractBaseRepo:
    """Test base repo extraction for deduplication."""

    def test_github_base(self):
        url = "https://github.com/user/repo/tree/main/src"
        base = extract_base_repo(url)
        assert base == "github.com/user/repo"

    def test_pypi_base(self):
        url = "https://pypi.org/project/mypackage/1.0/"
        base = extract_base_repo(url)
        assert "pypi.org/project/mypackage" in base


class TestIsDuplicate:
    """Test duplicate detection."""

    def test_detects_duplicate_url(self):
        existing = [
            {"relatedIdentifier": "https://github.com/user/repo", "relatedIdentifierType": "URL"}
        ]
        assert is_duplicate("https://github.com/user/repo", existing) is True

    def test_detects_duplicate_with_different_path(self):
        existing = [
            {"relatedIdentifier": "https://github.com/user/repo/releases", "relatedIdentifierType": "URL"}
        ]
        assert is_duplicate("https://github.com/user/repo", existing) is True

    def test_not_duplicate_different_repo(self):
        existing = [
            {"relatedIdentifier": "https://github.com/user/repo", "relatedIdentifierType": "URL"}
        ]
        assert is_duplicate("https://github.com/other/project", existing) is False
