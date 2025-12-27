"""Integration tests for batch promotion engine."""

from unittest.mock import patch

from extract_software_repos.promotion import BatchPromotionEngine
from extract_software_repos.paper_records import PaperInfo
from extract_software_repos.github_graphql import GitHubPromotionData


class TestBatchPromotionIntegration:
    """Integration tests for end-to-end batch promotion."""

    def test_multiple_chunks_track_promoted_urls(self):
        """Promoted URLs should persist across chunks."""
        engine = BatchPromotionEngine(
            promotion_threshold=2,
            chunk_size=2,  # Small chunks for testing
        )

        # Same URL in different chunks
        records = [
            {"doi": "10.1234/a", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo1", "relationType": "References"}, "_validation": {"is_valid": True}},
            {"doi": "10.1234/b", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo2", "relationType": "References"}, "_validation": {"is_valid": True}},
            # Chunk 2
            {"doi": "10.1234/c", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo1", "relationType": "References"}, "_validation": {"is_valid": True}},  # Same as first
            {"doi": "10.1234/d", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo3", "relationType": "References"}, "_validation": {"is_valid": True}},
        ]

        paper_index = {
            "10.1234/a": PaperInfo(doi="10.1234/a", title="repo1", authors=["A"], arxiv_id="1111.1111"),
            "10.1234/b": PaperInfo(doi="10.1234/b", title="Different", authors=["B"], arxiv_id=None),
            "10.1234/c": PaperInfo(doi="10.1234/c", title="Also Different", authors=["C"], arxiv_id=None),
            "10.1234/d": PaperInfo(doi="10.1234/d", title="repo3", authors=["D"], arxiv_id="3333.3333"),
        }

        github_data = {
            "https://github.com/owner/repo1": GitHubPromotionData(
                url="https://github.com/owner/repo1",
                description="1111.1111",
                readme_content="repo1 - arxiv:1111.1111",
                contributors=[],
            ),
            "https://github.com/owner/repo2": GitHubPromotionData(
                url="https://github.com/owner/repo2",
                description="",
                readme_content="# Repo 2",
                contributors=[],
            ),
            "https://github.com/owner/repo3": GitHubPromotionData(
                url="https://github.com/owner/repo3",
                description="3333.3333",
                readme_content="repo3 - 3333.3333",
                contributors=[],
            ),
        }

        promoted_urls = set()

        # Process chunk 1
        chunk1 = records[0:2]
        promoted1 = engine.process_chunk(chunk1, paper_index, github_data, promoted_urls)

        # repo1 should be promoted (arxiv + name match)
        assert records[0]["_promotion"]["promoted"] is True
        assert "https://github.com/owner/repo1" in promoted_urls
        assert promoted1 == 1

        # Process chunk 2
        chunk2 = records[2:4]
        promoted2 = engine.process_chunk(chunk2, paper_index, github_data, promoted_urls)

        # repo1 again - should be skipped (already promoted in chunk 1)
        assert records[2]["_promotion"]["promoted"] is False
        assert records[2]["_promotion"]["skip_reason"] == "url_already_promoted"

        # repo3 should be promoted
        assert records[3]["_promotion"]["promoted"] is True
        assert promoted2 == 1

    @patch("extract_software_repos.heuristics.author_matching.batch_match_authors")
    def test_author_matching_only_called_when_needed(self, mock_batch_match):
        """Author matching should only run for records that need it."""
        mock_batch_match.return_value = {}

        engine = BatchPromotionEngine(promotion_threshold=2)

        records = [
            # This one has 2 fast signals - no author matching needed
            {"doi": "10.1234/a", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/matching-title", "relationType": "References"}, "_validation": {"is_valid": True}},
            # This one has 0 signals - can't reach threshold, no author matching
            {"doi": "10.1234/b", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/other", "relationType": "References"}, "_validation": {"is_valid": True}},
            # This one has 1 signal - needs author matching
            {"doi": "10.1234/c", "enrichedValue": {"relatedIdentifier": "https://github.com/owner/somerepo", "relationType": "References"}, "_validation": {"is_valid": True}},
        ]

        paper_index = {
            "10.1234/a": PaperInfo(doi="10.1234/a", title="Matching Title", authors=["A"], arxiv_id="1111.1111"),
            "10.1234/b": PaperInfo(doi="10.1234/b", title="Unrelated", authors=["B"], arxiv_id=None),
            "10.1234/c": PaperInfo(doi="10.1234/c", title="Completely Different Paper Name", authors=["C"], arxiv_id="3333.3333"),
        }

        github_data = {
            "https://github.com/owner/matching-title": GitHubPromotionData(
                url="https://github.com/owner/matching-title",
                description="1111.1111",  # arxiv match
                readme_content="# Matching Title",  # name match
                contributors=[{"login": "a"}],
            ),
            "https://github.com/owner/other": GitHubPromotionData(
                url="https://github.com/owner/other",
                description="",
                readme_content="# Other",
                contributors=[{"login": "b"}],
            ),
            "https://github.com/owner/somerepo": GitHubPromotionData(
                url="https://github.com/owner/somerepo",
                description="3333.3333",  # Only arxiv match
                readme_content="# Something Else",
                contributors=[{"login": "c"}],
            ),
        }

        promoted_urls = set()
        engine.process_chunk(records, paper_index, github_data, promoted_urls)

        # batch_match_authors should be called with only the record that needs it
        assert mock_batch_match.called
        call_args = mock_batch_match.call_args[0][0]  # First positional arg
        assert len(call_args) == 1
        assert call_args[0]["key"] == ("https://github.com/owner/somerepo", "10.1234/c")
