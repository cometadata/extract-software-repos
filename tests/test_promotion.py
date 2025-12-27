"""Tests for promotion engine."""

from extract_software_repos.promotion import (
    evaluate_promotion,
    PromotionEngine,
)
from extract_software_repos.heuristics.arxiv_detection import ArxivDetectionResult
from extract_software_repos.heuristics.name_similarity import NameSimilarityResult
from extract_software_repos.heuristics.author_matching import AuthorMatchResult, AuthorMatchDetail


class TestEvaluatePromotion:
    """Tests for promotion decision logic."""

    def test_two_signals_promotes(self):
        arxiv = ArxivDetectionResult(matched=True, location="readme", found_id="2308.11197")
        name = NameSimilarityResult(matched=True, score=0.6)
        author = AuthorMatchResult(matched=False)

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        assert result.promoted is True
        assert "arxiv_id_in_readme" in result.signals
        assert "name_similarity" in result.signals

    def test_three_signals_promotes(self):
        arxiv = ArxivDetectionResult(matched=True, location="description", found_id="2308.11197")
        name = NameSimilarityResult(matched=True, score=0.7)
        author = AuthorMatchResult(
            matched=True,
            matches=[AuthorMatchDetail("jsmith", "John Smith", 0.95)],
        )

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        assert result.promoted is True
        assert len(result.signals) == 3

    def test_one_signal_no_promotion(self):
        arxiv = ArxivDetectionResult(matched=True, location="readme", found_id="2308.11197")
        name = NameSimilarityResult(matched=False, score=0.2)
        author = AuthorMatchResult(matched=False)

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        assert result.promoted is False

    def test_zero_signals_no_promotion(self):
        arxiv = ArxivDetectionResult(matched=False)
        name = NameSimilarityResult(matched=False, score=0.1)
        author = AuthorMatchResult(matched=False)

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        assert result.promoted is False
        assert len(result.signals) == 0

    def test_custom_threshold(self):
        arxiv = ArxivDetectionResult(matched=True, location="readme", found_id="2308.11197")
        name = NameSimilarityResult(matched=False, score=0.2)
        author = AuthorMatchResult(matched=False)

        # With threshold=1, one signal is enough
        result = evaluate_promotion(arxiv, name, author, threshold=1)
        assert result.promoted is True

    def test_skipped_signals_dont_count(self):
        arxiv = ArxivDetectionResult(matched=False, skipped=True, skip_reason="no_paper_arxiv_id")
        name = NameSimilarityResult(matched=True, score=0.6)
        author = AuthorMatchResult(matched=True, matches=[AuthorMatchDetail("j", "J", 0.9)])

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        # Only 2 signals ran, both matched
        assert result.promoted is True

    def test_evidence_includes_scores(self):
        arxiv = ArxivDetectionResult(matched=True, location="readme", found_id="2308.11197")
        name = NameSimilarityResult(matched=True, score=0.65)
        author = AuthorMatchResult(matched=False)

        result = evaluate_promotion(arxiv, name, author, threshold=2)
        assert result.evidence["arxiv_id_found"] == "2308.11197"
        assert result.evidence["arxiv_id_location"] == "readme"
        assert result.evidence["name_similarity_score"] == 0.65


class TestPromotionEngine:
    """Tests for the promotion engine."""

    def test_init(self):
        engine = PromotionEngine(github_token="test_token")
        assert engine.github_token == "test_token"

    def test_build_paper_index(self):
        from extract_software_repos.paper_records import PaperInfo

        engine = PromotionEngine(github_token="test")
        papers = [
            PaperInfo(doi="10.1234/a", title="Paper A", authors=["A"], arxiv_id="1111.1111"),
            PaperInfo(doi="10.1234/b", title="Paper B", authors=["B"], arxiv_id="2222.2222"),
        ]

        index = engine._build_paper_index(papers)
        assert "10.1234/a" in index
        assert index["10.1234/a"].title == "Paper A"

    def test_filter_promotable_records(self):
        engine = PromotionEngine(github_token="test")

        records = [
            # Valid GitHub URL with matching paper
            {
                "doi": "10.1234/a",
                "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo"},
                "_validation": {"is_valid": True},
            },
            # Invalid URL
            {
                "doi": "10.1234/b",
                "enrichedValue": {"relatedIdentifier": "https://github.com/owner/repo2"},
                "_validation": {"is_valid": False},
            },
            # Non-GitHub URL
            {
                "doi": "10.1234/c",
                "enrichedValue": {"relatedIdentifier": "https://gitlab.com/owner/repo"},
                "_validation": {"is_valid": True},
            },
        ]

        from extract_software_repos.paper_records import PaperInfo
        paper_index = {"10.1234/a": PaperInfo(doi="10.1234/a")}

        promotable = engine._filter_promotable_records(records, paper_index)
        assert len(promotable) == 1
        assert promotable[0]["doi"] == "10.1234/a"


class TestBatchPromotionEngine:
    """Tests for batched promotion engine."""

    def test_init(self):
        from extract_software_repos.promotion import BatchPromotionEngine

        engine = BatchPromotionEngine(
            promotion_threshold=2,
            name_similarity_threshold=0.45,
            chunk_size=1000,
            model_batch_size=256,
        )
        assert engine.promotion_threshold == 2
        assert engine.chunk_size == 1000

    def test_build_evidence(self):
        from extract_software_repos.promotion import BatchPromotionEngine

        engine = BatchPromotionEngine()

        arxiv = ArxivDetectionResult(matched=True, location="readme", found_id="2308.11197")
        name = NameSimilarityResult(matched=True, score=0.65, containment_score=0.7, token_overlap_score=0.5, fuzzy_score=0.6)
        author = AuthorMatchResult(matched=True, matches=[AuthorMatchDetail("jsmith", "John Smith", 0.95)])

        evidence = engine._build_evidence(arxiv, name, author)

        assert evidence["arxiv_id_found"] == "2308.11197"
        assert evidence["name_similarity_score"] == 0.65
        assert len(evidence["author_matches"]) == 1
