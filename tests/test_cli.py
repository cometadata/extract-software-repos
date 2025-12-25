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


class TestExtractSoftwareCommand:
    """Test extract-software command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["extract-software", "--help"])
        assert result.exit_code == 0
        assert "Extract software URLs from record abstracts" in result.output

    def test_extracts_from_jsonl(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "records.jsonl"
            output_file = Path(tmpdir) / "enrichments.jsonl"

            # Create test input
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
                "extract-software",
                str(input_file),
                "-o", str(output_file)
            ])

            assert result.exit_code == 0
            assert output_file.exists()

            with open(output_file) as f:
                enrichments = [json.loads(line) for line in f]

            assert len(enrichments) == 1
            assert enrichments[0]["doi"] == "10.1234/test1"


class TestValidateCommand:
    """Test validate command."""

    def test_help(self, runner):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "Validate extracted URLs" in result.output


class TestUpdatedCLI:
    """Test updated CLI with Polars backend."""

    def test_extract_urls_uses_polars(self, runner):
        df = pl.DataFrame({
            "relative_path": ["2308.11197v1.md"],
            "content": ["Check https://github.com/user/repo"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "output.jsonl"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "extract-urls",
                str(input_path),
                "-o", str(output_path),
            ])

            assert result.exit_code == 0
            assert output_path.exists()

    def test_heal_text_parallel(self, runner):
        df = pl.DataFrame({
            "content": ["Some text with git-\nhub.com link"]
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.parquet"
            output_path = Path(tmpdir) / "healed.parquet"
            df.write_parquet(input_path)

            result = runner.invoke(cli, [
                "heal-text",
                str(input_path),
                "-o", str(output_path),
                "--workers", "2",
            ])

            assert result.exit_code == 0
            assert output_path.exists()
