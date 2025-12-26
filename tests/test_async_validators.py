# tests/test_async_validators.py
"""Tests for async HTTP validators."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from extract_software_repos.async_validators import (
    AsyncHTTPValidator,
    validate_pypi_async,
)


class TestValidatePyPIAsync:
    """Test async PyPI validation."""

    @pytest.mark.asyncio
    async def test_valid_package(self):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.head.return_value = mock_response

        valid, error = await validate_pypi_async(
            "https://pypi.org/project/requests",
            mock_session,
        )
        assert valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_not_found(self):
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.head.return_value = mock_response

        valid, error = await validate_pypi_async(
            "https://pypi.org/project/nonexistent12345",
            mock_session,
        )
        assert valid is False
        assert error == "not_found"


class TestAsyncHTTPValidator:
    """Test the async validator orchestrator."""

    @pytest.mark.asyncio
    async def test_validates_multiple_urls(self):
        validator = AsyncHTTPValidator(max_concurrency=10)

        urls = [
            {"url": "https://pypi.org/project/requests", "type": "pypi"},
            {"url": "https://pypi.org/project/nonexistent99999", "type": "pypi"},
        ]

        # We'll mock the actual HTTP calls in integration tests
        # This just tests the structure
        with patch.object(validator, "_validate_single") as mock_validate:
            mock_validate.side_effect = [
                {"url": urls[0]["url"], "valid": True, "method": "api_check", "error": None},
                {"url": urls[1]["url"], "valid": False, "method": "api_check", "error": "not_found"},
            ]

            results = await validator.validate_urls(urls)

            assert len(results) == 2
            assert results[0]["valid"] is True
            assert results[1]["valid"] is False
