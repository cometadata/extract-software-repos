# tests/test_validation.py
"""Tests for URL validation."""

import pytest
from unittest.mock import patch, MagicMock

from extract_software_repos.validation import (
    ValidationResult,
    ValidationStats,
    validate_url,
    validate_git_repo,
    validate_pypi,
    can_validate,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_valid_result(self):
        result = ValidationResult(
            url="https://github.com/user/repo",
            url_type="github",
            is_valid=True,
            method="git_ls_remote",
        )
        assert result.is_valid is True
        assert result.error is None

    def test_create_invalid_result(self):
        result = ValidationResult(
            url="https://github.com/user/nonexistent",
            url_type="github",
            is_valid=False,
            method="git_ls_remote",
            error="not_found",
        )
        assert result.is_valid is False
        assert result.error == "not_found"


class TestValidationStats:
    """Test ValidationStats tracking."""

    def test_add_valid_result(self):
        stats = ValidationStats()
        result = ValidationResult(
            url="https://github.com/user/repo",
            url_type="github",
            is_valid=True,
            method="git_ls_remote",
        )
        stats.add_result(result)
        assert stats.total_urls == 1
        assert stats.valid_urls == 1
        assert stats.invalid_urls == 0

    def test_add_invalid_result(self):
        stats = ValidationStats()
        result = ValidationResult(
            url="https://github.com/user/repo",
            url_type="github",
            is_valid=False,
            method="git_ls_remote",
            error="not_found",
        )
        stats.add_result(result)
        assert stats.total_urls == 1
        assert stats.valid_urls == 0
        assert stats.invalid_urls == 1


class TestCanValidate:
    """Test validator availability check."""

    def test_github_can_validate(self):
        assert can_validate("github") is True

    def test_pypi_can_validate(self):
        assert can_validate("pypi") is True

    def test_unknown_cannot_validate(self):
        assert can_validate("unknown_registry") is False


class TestValidateGitRepo:
    """Test git repository validation."""

    @patch("subprocess.run")
    def test_valid_repo(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        is_valid, method, error = validate_git_repo("https://github.com/user/repo")
        assert is_valid is True
        assert method == "git_ls_remote"
        assert error is None

    @patch("subprocess.run")
    def test_not_found(self, mock_run):
        mock_run.return_value = MagicMock(returncode=2, stderr="")
        is_valid, method, error = validate_git_repo("https://github.com/user/nonexistent")
        assert is_valid is False
        assert error == "not_found"


class TestValidatePypi:
    """Test PyPI validation."""

    @patch("requests.head")
    def test_valid_package(self, mock_head):
        mock_head.return_value = MagicMock(status_code=200)
        is_valid, method, error = validate_pypi("https://pypi.org/project/requests")
        assert is_valid is True
        assert method == "api_check"

    @patch("requests.head")
    def test_not_found(self, mock_head):
        mock_head.return_value = MagicMock(status_code=404)
        is_valid, method, error = validate_pypi("https://pypi.org/project/nonexistent12345")
        assert is_valid is False
        assert error == "not_found"


class TestValidateUrl:
    """Test main validation dispatcher."""

    @patch("subprocess.run")
    def test_validates_github(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        result = validate_url("https://github.com/user/repo", "github")
        assert result.is_valid is True
        assert result.url_type == "github"

    def test_no_validator_available(self):
        result = validate_url("https://example.com/something", "unknown")
        assert result.is_valid is False
        assert result.error == "no_validator"
