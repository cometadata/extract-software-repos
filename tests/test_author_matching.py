"""Tests for author matching heuristic."""

from unittest.mock import MagicMock, patch
from extract_software_repos.heuristics.author_matching import (
    ContributorInfo,
    format_matching_input,
    match_authors_to_contributors,
)


class TestFormatMatchingInput:
    """Tests for input formatting."""

    def test_full_info(self):
        text = format_matching_input(
            username="jsmith",
            name="John Smith",
            email="john@example.com",
            author_name="John Smith",
        )
        assert "<username>jsmith</username>" in text
        assert "<name>John Smith</name>" in text
        assert "<email>john@example.com</email>" in text
        assert "<author-details>" in text
        assert "John Smith" in text

    def test_minimal_info(self):
        text = format_matching_input(
            username="jsmith",
            name=None,
            email=None,
            author_name="John Smith",
        )
        assert "<username>jsmith</username>" in text
        assert "<name>None</name>" in text
        assert "<email>None</email>" in text

    def test_template_format_preserved(self):
        # The template format must match exactly what sci-soft-models expects
        text = format_matching_input(
            username="test",
            name="Test User",
            email="test@test.com",
            author_name="Author Name",
        )
        assert "<developer-details>" in text
        assert "</developer-details>" in text
        assert "---" in text
        assert "<author-details>" in text
        assert "</author-details>" in text


class TestContributorInfo:
    """Tests for ContributorInfo dataclass."""

    def test_creation(self):
        info = ContributorInfo(login="jsmith", name="John Smith", email="j@example.com")
        assert info.login == "jsmith"
        assert info.name == "John Smith"
        assert info.email == "j@example.com"

    def test_optional_fields(self):
        info = ContributorInfo(login="jsmith")
        assert info.login == "jsmith"
        assert info.name is None
        assert info.email is None


class TestMatchAuthorsToContributors:
    """Tests for the main matching function."""

    def test_no_contributors(self):
        result = match_authors_to_contributors(
            contributors=[],
            paper_authors=["John Smith"],
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_contributors"

    def test_no_authors(self):
        result = match_authors_to_contributors(
            contributors=[ContributorInfo(login="jsmith")],
            paper_authors=[],
        )
        assert result.matched is False
        assert result.skipped is True
        assert result.skip_reason == "no_authors"

    @patch("extract_software_repos.heuristics.author_matching.load_author_matching_model")
    def test_match_found(self, mock_load):
        # Mock the model to return a match
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "match", "score": 0.95}]
        mock_load.return_value = mock_pipeline

        result = match_authors_to_contributors(
            contributors=[ContributorInfo(login="jsmith", name="John Smith")],
            paper_authors=["John Smith"],
        )
        assert result.matched is True
        assert len(result.matches) == 1
        assert result.matches[0].contributor_login == "jsmith"
        assert result.matches[0].author_name == "John Smith"
        assert result.matches[0].confidence == 0.95

    @patch("extract_software_repos.heuristics.author_matching.load_author_matching_model")
    def test_no_match(self, mock_load):
        # Mock the model to return no match
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "no_match", "score": 0.85}]
        mock_load.return_value = mock_pipeline

        result = match_authors_to_contributors(
            contributors=[ContributorInfo(login="jsmith", name="John Smith")],
            paper_authors=["Jane Doe"],
        )
        assert result.matched is False
        assert len(result.matches) == 0

    @patch("extract_software_repos.heuristics.author_matching.load_author_matching_model")
    def test_multiple_pairs(self, mock_load):
        # Mock to return match for first pair only
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "match", "score": 0.95},
            {"label": "no_match", "score": 0.80},
        ]
        mock_load.return_value = mock_pipeline

        result = match_authors_to_contributors(
            contributors=[ContributorInfo(login="jsmith", name="John Smith")],
            paper_authors=["John Smith", "Jane Doe"],
        )
        assert result.matched is True
        assert len(result.matches) == 1

    @patch("extract_software_repos.heuristics.author_matching.load_author_matching_model")
    def test_cached_model(self, mock_load):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "match", "score": 0.9}]

        # Pass pre-loaded model
        result = match_authors_to_contributors(
            contributors=[ContributorInfo(login="jsmith")],
            paper_authors=["John Smith"],
            loaded_model=mock_pipeline,
        )

        # Model loader should not be called when model is provided
        mock_load.assert_not_called()
        assert result.matched is True
