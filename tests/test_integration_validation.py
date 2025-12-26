# tests/test_integration_validation.py
"""Integration tests for fast validation pipeline."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from click.testing import CliRunner

from extract_software_repos.cli import cli


@pytest.fixture
def sample_enrichments(tmp_path):
    """Create sample enrichment file."""
    enrichments = [
        {
            "doi": "10.1234/test1",
            "enrichedValue": {"relatedIdentifier": "https://github.com/pytorch/pytorch"},
        },
        {
            "doi": "10.1234/test2",
            "enrichedValue": {"relatedIdentifier": "https://github.com/pytorch/pytorch"},  # duplicate
        },
        {
            "doi": "10.1234/test3",
            "enrichedValue": {"relatedIdentifier": "https://pypi.org/project/requests"},
        },
    ]

    input_file = tmp_path / "enrichments.jsonl"
    with open(input_file, "w") as f:
        for e in enrichments:
            f.write(json.dumps(e) + "\n")

    return input_file


class TestFastValidationIntegration:
    """Integration tests for the full validation pipeline."""

    @patch.dict("os.environ", {"GITHUB_TOKEN": "test_token"})
    @patch("extract_software_repos.github_graphql.aiohttp.ClientSession")
    @patch("extract_software_repos.async_validators.aiohttp.ClientSession")
    def test_validates_mixed_urls(
        self,
        mock_async_session,
        mock_github_session,
        sample_enrichments,
        tmp_path,
    ):
        """Test validation of mixed URL types."""
        # Mock GitHub GraphQL response
        mock_github_response = AsyncMock()
        mock_github_response.status = 200
        mock_github_response.json = AsyncMock(return_value={
            "data": {"repo0": {"id": "123"}}
        })
        mock_github_response.headers = {"X-RateLimit-Remaining": "4999"}

        # Mock PyPI response
        mock_pypi_response = AsyncMock()
        mock_pypi_response.status = 200

        runner = CliRunner()
        output_file = tmp_path / "validated.jsonl"
        checkpoint_file = tmp_path / "cache.jsonl"

        with patch("extract_software_repos.fast_validation.GitHubGraphQLValidator") as mock_gh:
            mock_gh.return_value.validate_urls = AsyncMock(return_value=[
                MagicMock(url="https://github.com/pytorch/pytorch", valid=True, error=None)
            ])

            with patch("extract_software_repos.fast_validation.AsyncHTTPValidator") as mock_http:
                mock_http.return_value.validate_urls = AsyncMock(return_value=[
                    {"url": "https://pypi.org/project/requests", "valid": True, "method": "api_check", "error": None}
                ])

                result = runner.invoke(cli, [
                    "validate",
                    str(sample_enrichments),
                    "-o", str(output_file),
                    "--checkpoint", str(checkpoint_file),
                ])

        # Should complete without error
        assert result.exit_code == 0, result.output
        assert "Validation complete" in result.output

        # Check output file
        assert output_file.exists()
        with open(output_file) as f:
            output_records = [json.loads(line) for line in f]

        # Should have records with validation info
        assert len(output_records) >= 1
        assert "_validation" in output_records[0]

    def test_checkpoint_resume(self, sample_enrichments, tmp_path):
        """Test that checkpoint enables resume."""
        checkpoint_file = tmp_path / "cache.jsonl"

        # Pre-populate checkpoint
        with open(checkpoint_file, "w") as f:
            f.write(json.dumps({
                "url": "https://github.com/pytorch/pytorch",
                "valid": True,
                "method": "graphql",
                "error": None,
                "checked_at": "2025-01-01T00:00:00Z",
            }) + "\n")
            f.write(json.dumps({
                "url": "https://pypi.org/project/requests",
                "valid": True,
                "method": "api_check",
                "error": None,
                "checked_at": "2025-01-01T00:00:00Z",
            }) + "\n")

        runner = CliRunner()
        output_file = tmp_path / "validated.jsonl"

        # Should skip validation (all cached)
        with patch.dict("os.environ", {"GITHUB_TOKEN": "test_token"}):
            result = runner.invoke(cli, [
                "validate",
                str(sample_enrichments),
                "-o", str(output_file),
                "--checkpoint", str(checkpoint_file),
            ])

        assert result.exit_code == 0
        assert output_file.exists()
