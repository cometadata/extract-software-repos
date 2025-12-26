# tests/test_cli.py
"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
import polars as pl
from click.testing import CliRunner

from extract_software_repos.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestExtractCommand:
    """Test extract command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract software repository URLs" in result.output

    def test_extracts_from_datacite(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "records.jsonl"
            output_file = Path(tmpdir) / "enrichments.jsonl"

            records = [
                {
                    "id": "10.1234/test1",
                    "attributes": {
                        "descriptions": [
                            {"descriptionType": "Abstract", "description": "See https://github.com/user/repo"}
                        ]
                    }
                }
            ]
            with open(input_file, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            result = runner.invoke(cli, [
                "extract",
                str(input_file),
                "--from-datacite-abstract",
                "-o", str(output_file)
            ])

            assert result.exit_code == 0
            assert output_file.exists()

            with open(output_file) as f:
                enrichments = [json.loads(line) for line in f]

            assert len(enrichments) == 1
            assert enrichments[0]["doi"] == "10.1234/test1"

    def test_extracts_from_parquet(self, runner):
        df = pl.DataFrame({
            "relative_path": ["2308.11197v1.md"],
            "content": ["Check https://github.com/user/repo"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "output.jsonl"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "extract",
                str(input_path),
                "-o", str(output_path),
            ])

            assert result.exit_code == 0
            assert output_path.exists()


class TestValidateCommand:
    """Test validate command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate extracted URLs" in result.output

    def test_checkpoint_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--checkpoint" in result.output

    def test_wait_for_ratelimit_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--wait-for-ratelimit" in result.output

    def test_ignore_checkpoint_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--ignore-checkpoint" in result.output

    def test_http_concurrency_option_exists(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert "--http-concurrency" in result.output


class TestHealFulltextCommand:
    """Test heal-fulltext command."""

    def test_heal_fulltext_parallel(self, runner):
        df = pl.DataFrame({
            "content": ["Some text with git-\nhub.com link"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "healed.parquet"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "heal-fulltext",
                str(input_path),
                "-o", str(output_path),
                "--workers", "2",
            ])

            assert result.exit_code == 0
            assert output_path.exists()
