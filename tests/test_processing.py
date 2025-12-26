# tests/test_processing.py
"""Tests for record and fulltext processing."""

import pytest
from datetime import datetime

from extract_software_repos.processing import (
    normalize_doi,
    create_enrichment,
    process_record,
    parse_arxiv_id,
    derive_doi,
)


class TestNormalizeDoi:
    """Test DOI normalization."""

    def test_lowercase(self):
        assert normalize_doi("10.1234/ABC") == "10.1234/abc"

    def test_strips_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_doi_prefix(self):
        assert normalize_doi("doi:10.1234/abc") == "10.1234/abc"

    def test_strips_whitespace(self):
        assert normalize_doi("  10.1234/abc  ") == "10.1234/abc"


class TestCreateEnrichment:
    """Test enrichment record creation."""

    def test_creates_valid_structure(self):
        enrichment = create_enrichment("10.1234/test", "https://github.com/user/repo")

        assert enrichment["doi"] == "10.1234/test"
        assert enrichment["action"] == "insert_child"
        assert enrichment["field"] == "relatedIdentifiers"
        assert enrichment["enrichedValue"]["relatedIdentifier"] == "https://github.com/user/repo"
        assert enrichment["enrichedValue"]["relatedIdentifierType"] == "URL"
        assert enrichment["enrichedValue"]["relationType"] == "IsSupplementedBy"

    def test_custom_relation_type(self):
        """Test that custom relation_type is used (e.g., References for fulltext)."""
        enrichment = create_enrichment("10.1234/test", "https://github.com/user/repo", relation_type="References")
        assert enrichment["enrichedValue"]["relationType"] == "References"

    def test_default_relation_type_is_supplemented_by(self):
        """Test that default relation_type is IsSupplementedBy (for records extraction)."""
        enrichment = create_enrichment("10.1234/test", "https://github.com/user/repo")
        assert enrichment["enrichedValue"]["relationType"] == "IsSupplementedBy"

    def test_includes_source(self):
        enrichment = create_enrichment("10.1234/test", "https://github.com/user/repo")
        assert len(enrichment["sources"]) == 1
        assert enrichment["sources"][0]["name"] == "COMET Project"

    def test_includes_timestamps(self):
        enrichment = create_enrichment("10.1234/test", "https://github.com/user/repo")
        assert "created" in enrichment
        assert "updated" in enrichment
        assert "produced" in enrichment


class TestProcessRecord:
    """Test DataCite record processing."""

    def test_extracts_from_abstract(self):
        record = {
            "id": "10.1234/test",
            "attributes": {
                "descriptions": [
                    {
                        "descriptionType": "Abstract",
                        "description": "Code at https://github.com/user/repo"
                    }
                ]
            }
        }
        enrichments = process_record(record)
        assert len(enrichments) == 1
        assert enrichments[0]["enrichedValue"]["relatedIdentifier"] == "https://github.com/user/repo"

    def test_skips_existing_urls(self):
        record = {
            "id": "10.1234/test",
            "attributes": {
                "descriptions": [
                    {
                        "descriptionType": "Abstract",
                        "description": "Code at https://github.com/user/repo"
                    }
                ],
                "relatedIdentifiers": [
                    {
                        "relatedIdentifier": "https://github.com/user/repo",
                        "relatedIdentifierType": "URL"
                    }
                ]
            }
        }
        enrichments = process_record(record)
        assert len(enrichments) == 0

    def test_returns_empty_for_no_urls(self):
        record = {
            "id": "10.1234/test",
            "attributes": {
                "descriptions": [
                    {
                        "descriptionType": "Abstract",
                        "description": "No URLs in this abstract."
                    }
                ]
            }
        }
        enrichments = process_record(record)
        assert len(enrichments) == 0


class TestArxivHelpers:
    """Test arxiv ID parsing."""

    def test_parse_valid_id(self):
        assert parse_arxiv_id("2308.11197v3.md") == "2308.11197"

    def test_parse_without_version(self):
        assert parse_arxiv_id("2308.11197.md") == "2308.11197"

    def test_parse_invalid_returns_none(self):
        assert parse_arxiv_id("invalid.md") is None

    def test_derive_doi(self):
        assert derive_doi("2308.11197") == "10.48550/arxiv.2308.11197"
