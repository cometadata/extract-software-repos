"""Integration tests for the promotion pipeline."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from extract_software_repos.promotion import PromotionEngine
from extract_software_repos.paper_records import PaperInfo
from extract_software_repos.github_graphql import GitHubPromotionData


class TestPromotionIntegration:
    """End-to-end tests for promotion pipeline."""

    @pytest.fixture
    def sample_records(self):
        return [
            {
                "doi": "10.48550/arxiv.2308.11197",
                "enrichedValue": {
                    "relatedIdentifier": "https://github.com/owner/repo",
                    "relatedIdentifierType": "URL",
                    "relationType": "References",
                },
                "_validation": {"is_valid": True, "method": "graphql"},
            },
            {
                "doi": "10.48550/arxiv.1234.56789",
                "enrichedValue": {
                    "relatedIdentifier": "https://github.com/other/project",
                    "relatedIdentifierType": "URL",
                    "relationType": "References",
                },
                "_validation": {"is_valid": True, "method": "graphql"},
            },
        ]

    @pytest.fixture
    def sample_papers(self):
        return [
            PaperInfo(
                doi="10.48550/arxiv.2308.11197",
                title="Extended Isolation Forest",
                authors=["John Smith", "Jane Doe"],
                arxiv_id="2308.11197",
            ),
            PaperInfo(
                doi="10.48550/arxiv.1234.56789",
                title="Another Paper",
                authors=["Alice Bob"],
                arxiv_id="1234.56789",
            ),
        ]

    @pytest.mark.asyncio
    async def test_promotion_with_arxiv_and_author_match(self, sample_records, sample_papers):
        """Test promotion when arXiv ID and author both match."""
        engine = PromotionEngine(
            github_token="test_token",
            promotion_threshold=2,
        )

        # Mock GitHub data fetch
        mock_promotion_data = [
            GitHubPromotionData(
                url="https://github.com/owner/repo",
                description="Implementation of Extended Isolation Forest",
                readme_content="# Extended Isolation Forest\n\nPaper: arxiv.org/abs/2308.11197",
                contributors=[
                    {"login": "jsmith", "name": "John Smith", "email": "j@example.com"}
                ],
            ),
            GitHubPromotionData(
                url="https://github.com/other/project",
                description="Some other project",
                readme_content="# Project\n\nNo paper reference here",
                contributors=[
                    {"login": "random", "name": "Random User", "email": None}
                ],
            ),
        ]

        # Mock author matching model
        mock_model = MagicMock()
        mock_model.return_value = [{"label": "match", "score": 0.95}]

        with patch.object(engine, '_get_author_model', return_value=mock_model):
            with patch('extract_software_repos.promotion.GitHubPromotionFetcher') as MockFetcher:
                mock_fetcher = MockFetcher.return_value
                mock_fetcher.fetch_promotion_data = AsyncMock(return_value=mock_promotion_data)

                results = await engine.promote_records(sample_records, sample_papers)

        # First record should be promoted (arXiv in readme + author match)
        assert results[0]["_promotion"]["promoted"] is True
        assert results[0]["enrichedValue"]["relationType"] == "isSupplementedBy"
        assert "arxiv_id_in_readme" in results[0]["_promotion"]["signals"]

        # Second record should not be promoted
        assert results[1]["_promotion"]["promoted"] is False
        assert results[1]["enrichedValue"]["relationType"] == "References"

    @pytest.mark.asyncio
    async def test_promotion_threshold_respected(self, sample_records, sample_papers):
        """Test that promotion threshold is enforced."""
        engine = PromotionEngine(
            github_token="test_token",
            promotion_threshold=3,  # Require all 3 signals
        )

        # Mock data with only 2 matching signals
        mock_promotion_data = [
            GitHubPromotionData(
                url="https://github.com/owner/repo",
                readme_content="# Paper: 2308.11197",  # arXiv match
                description=None,
                contributors=[{"login": "jsmith", "name": "John Smith", "email": None}],  # Author match
            ),
            GitHubPromotionData(
                url="https://github.com/other/project",
                readme_content="",
                description="",
                contributors=[],
            ),
        ]

        mock_model = MagicMock()
        mock_model.return_value = [{"label": "match", "score": 0.95}]

        with patch.object(engine, '_get_author_model', return_value=mock_model):
            with patch('extract_software_repos.promotion.GitHubPromotionFetcher') as MockFetcher:
                mock_fetcher = MockFetcher.return_value
                mock_fetcher.fetch_promotion_data = AsyncMock(return_value=mock_promotion_data)

                results = await engine.promote_records(sample_records, sample_papers)

        # Should not be promoted because only 2 signals matched
        assert results[0]["_promotion"]["promoted"] is False
        # But evidence should show what matched
        assert len(results[0]["_promotion"]["signals"]) == 2

    @pytest.mark.asyncio
    async def test_promotion_preserves_original_relation(self, sample_records, sample_papers):
        """Test that original relation type is preserved in metadata."""
        engine = PromotionEngine(
            github_token="test_token",
            promotion_threshold=2,
        )

        mock_promotion_data = [
            GitHubPromotionData(
                url="https://github.com/owner/repo",
                readme_content="Paper: 2308.11197",
                description="Extended Isolation Forest implementation",
                contributors=[{"login": "jsmith", "name": "John Smith", "email": None}],
            ),
            GitHubPromotionData(
                url="https://github.com/other/project",
                readme_content="",
                description="",
                contributors=[],
            ),
        ]

        mock_model = MagicMock()
        mock_model.return_value = [{"label": "match", "score": 0.95}]

        with patch.object(engine, '_get_author_model', return_value=mock_model):
            with patch('extract_software_repos.promotion.GitHubPromotionFetcher') as MockFetcher:
                mock_fetcher = MockFetcher.return_value
                mock_fetcher.fetch_promotion_data = AsyncMock(return_value=mock_promotion_data)

                results = await engine.promote_records(sample_records, sample_papers)

        # Original relation should be preserved in metadata
        assert results[0]["_promotion"]["original_relation"] == "References"

    @pytest.mark.asyncio
    async def test_promotion_handles_invalid_records(self, sample_papers):
        """Test that invalid records are skipped."""
        records = [
            {
                "doi": "10.48550/arxiv.2308.11197",
                "enrichedValue": {
                    "relatedIdentifier": "https://github.com/owner/repo",
                    "relationType": "References",
                },
                "_validation": {"is_valid": False},  # Invalid
            },
        ]

        engine = PromotionEngine(
            github_token="test_token",
            promotion_threshold=2,
        )

        with patch('extract_software_repos.promotion.GitHubPromotionFetcher') as MockFetcher:
            mock_fetcher = MockFetcher.return_value
            mock_fetcher.fetch_promotion_data = AsyncMock(return_value=[])

            results = await engine.promote_records(records, sample_papers)

        # Record should pass through unchanged (no _promotion field)
        assert "_promotion" not in results[0]
        assert results[0]["enrichedValue"]["relationType"] == "References"
