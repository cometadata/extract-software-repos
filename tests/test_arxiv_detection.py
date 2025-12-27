"""Tests for arXiv ID detection heuristic."""

from extract_software_repos.heuristics.arxiv_detection import (
    detect_arxiv_id,
    normalize_arxiv_id,
    extract_arxiv_ids_from_text,
)


class TestNormalizeArxivId:
    """Tests for arXiv ID normalization."""

    def test_bare_id(self):
        assert normalize_arxiv_id("2308.11197") == "2308.11197"

    def test_version_suffix_stripped(self):
        assert normalize_arxiv_id("2308.11197v2") == "2308.11197"
        assert normalize_arxiv_id("2308.11197v15") == "2308.11197"

    def test_old_format_lowercase(self):
        assert normalize_arxiv_id("hep-th/9901001") == "hep-th/9901001"
        assert normalize_arxiv_id("HEP-TH/9901001") == "hep-th/9901001"

    def test_doi_format_extracted(self):
        assert normalize_arxiv_id("10.48550/arXiv.2308.11197") == "2308.11197"

    def test_url_format_extracted(self):
        assert normalize_arxiv_id("https://arxiv.org/abs/2308.11197") == "2308.11197"
        assert normalize_arxiv_id("arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"


class TestExtractArxivIdsFromText:
    """Tests for extracting arXiv IDs from text."""

    def test_bare_id_in_text(self):
        text = "See our paper 2308.11197 for details"
        ids = extract_arxiv_ids_from_text(text)
        assert "2308.11197" in ids

    def test_url_in_text(self):
        text = "Paper: https://arxiv.org/abs/2308.11197"
        ids = extract_arxiv_ids_from_text(text)
        assert "2308.11197" in ids

    def test_doi_in_text(self):
        text = "DOI: 10.48550/arXiv.2308.11197"
        ids = extract_arxiv_ids_from_text(text)
        assert "2308.11197" in ids

    def test_old_format_in_text(self):
        text = "See hep-th/9901001"
        ids = extract_arxiv_ids_from_text(text)
        assert "hep-th/9901001" in ids

    def test_multiple_ids(self):
        text = "Papers 2308.11197 and 1234.56789v2"
        ids = extract_arxiv_ids_from_text(text)
        assert "2308.11197" in ids
        assert "1234.56789" in ids

    def test_no_ids(self):
        text = "This is just regular text without any IDs"
        ids = extract_arxiv_ids_from_text(text)
        assert len(ids) == 0

    def test_version_normalized(self):
        text = "Paper 2308.11197v3"
        ids = extract_arxiv_ids_from_text(text)
        assert "2308.11197" in ids
        assert "2308.11197v3" not in ids


class TestDetectArxivId:
    """Tests for the main detection function."""

    def test_match_in_readme(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content="Implementation of paper 2308.11197",
            description=None,
        )
        assert result.matched is True
        assert result.location == "readme"
        assert result.found_id == "2308.11197"

    def test_match_in_description(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content=None,
            description="Code for arxiv.org/abs/2308.11197",
        )
        assert result.matched is True
        assert result.location == "description"

    def test_no_match(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content="Some unrelated content",
            description="A cool project",
        )
        assert result.matched is False
        assert result.location is None

    def test_different_id_no_match(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content="Implementation of 1234.56789",
            description=None,
        )
        assert result.matched is False

    def test_version_mismatch_still_matches(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content="Paper 2308.11197v2",
            description=None,
        )
        assert result.matched is True

    def test_no_paper_arxiv_id(self):
        result = detect_arxiv_id(
            paper_arxiv_id=None,
            readme_content="Paper 2308.11197",
            description=None,
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_paper_arxiv_id"

    def test_no_content(self):
        result = detect_arxiv_id(
            paper_arxiv_id="2308.11197",
            readme_content=None,
            description=None,
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_content"
