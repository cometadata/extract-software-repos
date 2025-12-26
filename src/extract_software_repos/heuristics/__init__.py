"""Heuristics for repo-paper matching."""

from .arxiv_detection import detect_arxiv_id, normalize_arxiv_id, ArxivDetectionResult
from .name_similarity import compute_name_similarity, NameSimilarityResult
from .author_matching import match_authors_to_contributors, AuthorMatchResult

__all__ = [
    "detect_arxiv_id",
    "normalize_arxiv_id",
    "ArxivDetectionResult",
    "compute_name_similarity",
    "NameSimilarityResult",
    "match_authors_to_contributors",
    "AuthorMatchResult",
]
