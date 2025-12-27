"""Tests for paper record extraction."""

import pytest
from extract_software_repos.paper_records import (
    extract_paper_info_datacite,
    extract_paper_info,
    normalize_doi,
)


class TestNormalizeDoi:
    """Tests for DOI normalization."""

    def test_lowercase(self):
        assert normalize_doi("10.1234/ABC") == "10.1234/abc"

    def test_strips_url_prefix(self):
        assert normalize_doi("https://doi.org/10.1234/abc") == "10.1234/abc"
        assert normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"

    def test_strips_dx_prefix(self):
        assert normalize_doi("https://dx.doi.org/10.1234/abc") == "10.1234/abc"

    def test_already_normalized(self):
        assert normalize_doi("10.1234/abc") == "10.1234/abc"


class TestExtractPaperInfoDatacite:
    """Tests for DataCite record extraction."""

    def test_full_record(self):
        record = {
            "id": "10.48550/arxiv.2308.11197",
            "attributes": {
                "doi": "10.48550/arxiv.2308.11197",
                "titles": [{"title": "Extended Isolation Forest"}],
                "creators": [
                    {
                        "nameType": "Personal",
                        "givenName": "John",
                        "familyName": "Smith",
                        "name": "Smith, John",
                    }
                ],
                "alternateIdentifiers": [
                    {"alternateIdentifier": "2308.11197", "alternateIdentifierType": "arXiv"}
                ],
            },
        }
        info = extract_paper_info_datacite(record)
        assert info.doi == "10.48550/arxiv.2308.11197"
        assert info.title == "Extended Isolation Forest"
        assert info.authors == ["John Smith"]
        assert info.arxiv_id == "2308.11197"

    def test_name_format_last_first(self):
        record = {
            "attributes": {
                "doi": "10.1234/test",
                "titles": [{"title": "Test"}],
                "creators": [
                    {
                        "nameType": "Personal",
                        "name": "Smith, John",
                    }
                ],
            },
        }
        info = extract_paper_info_datacite(record)
        assert info.authors == ["John Smith"]

    def test_organizational_creator_skipped(self):
        record = {
            "attributes": {
                "doi": "10.1234/test",
                "titles": [{"title": "Test"}],
                "creators": [
                    {"nameType": "Organizational", "name": "CERN"},
                    {"nameType": "Personal", "name": "Smith, John"},
                ],
            },
        }
        info = extract_paper_info_datacite(record)
        assert info.authors == ["John Smith"]
        assert "CERN" not in info.authors

    def test_arxiv_id_from_identifiers(self):
        record = {
            "attributes": {
                "doi": "10.1234/test",
                "titles": [{"title": "Test"}],
                "creators": [],
                "identifiers": [
                    {"identifier": "1234.56789", "identifierType": "arXiv"}
                ],
            },
        }
        info = extract_paper_info_datacite(record)
        assert info.arxiv_id == "1234.56789"

    def test_missing_fields(self):
        record = {"attributes": {}}
        info = extract_paper_info_datacite(record)
        assert info.doi is None
        assert info.title is None
        assert info.authors == []
        assert info.arxiv_id is None

    def test_doi_fallback_to_id(self):
        record = {
            "id": "10.1234/from-id",
            "attributes": {
                "titles": [{"title": "Test"}],
            },
        }
        info = extract_paper_info_datacite(record)
        assert info.doi == "10.1234/from-id"


class TestExtractPaperInfo:
    """Tests for the main extraction function with record types."""

    def test_datacite_type(self):
        record = {
            "attributes": {
                "doi": "10.1234/test",
                "titles": [{"title": "Test Paper"}],
                "creators": [],
            },
        }
        info = extract_paper_info(record, record_type="datacite")
        assert info.doi == "10.1234/test"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown record type"):
            extract_paper_info({}, record_type="unknown")
