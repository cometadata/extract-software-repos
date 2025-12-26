"""Heuristics for repo-paper matching."""

# Import modules as they are implemented
# Delay imports for name_similarity and author_matching until they exist
from .arxiv_detection import detect_arxiv_id, normalize_arxiv_id, ArxivDetectionResult

__all__ = [
    "detect_arxiv_id",
    "normalize_arxiv_id",
    "ArxivDetectionResult",
]
