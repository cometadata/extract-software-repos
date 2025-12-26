"""Name similarity heuristic for matching repo names to paper titles."""

import string
from dataclasses import dataclass
from typing import Optional, Set

from Levenshtein import ratio as levenshtein_ratio


@dataclass
class NameSimilarityResult:
    """Result of name similarity comparison."""
    matched: bool
    score: float = 0.0
    containment_score: float = 0.0
    token_overlap_score: float = 0.0
    fuzzy_score: float = 0.0
    skipped: bool = False
    skip_reason: Optional[str] = None


# Common stopwords to filter out
STOPWORDS = frozenset([
    "a", "an", "the", "for", "of", "on", "in", "with", "to", "and", "or",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "it", "its",
    "by", "from", "at", "as",
])


def normalize_text(text: str) -> Set[str]:
    """Normalize text for comparison.

    - Lowercase
    - Replace hyphens/underscores with spaces
    - Remove punctuation
    - Remove stopwords
    - Split into token set

    Args:
        text: Raw text

    Returns:
        Set of normalized tokens
    """
    # Lowercase
    text = text.lower()

    # Replace hyphens and underscores with spaces
    text = text.replace("-", " ").replace("_", " ")

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split and filter
    tokens = set(text.split())
    tokens = tokens - STOPWORDS

    return tokens


def compute_containment_score(repo_name: str, paper_title: str) -> float:
    """Compute containment score.

    Checks if normalized repo name tokens are contained in title or vice versa.

    Args:
        repo_name: Repository name
        paper_title: Paper title

    Returns:
        Score from 0 to 1
    """
    repo_tokens = normalize_text(repo_name)
    title_tokens = normalize_text(paper_title)

    if not repo_tokens or not title_tokens:
        return 0.0

    # Check both directions
    repo_in_title = len(repo_tokens & title_tokens) / len(repo_tokens)
    title_in_repo = len(repo_tokens & title_tokens) / len(title_tokens)

    # Return the better containment direction
    return max(repo_in_title, title_in_repo)


def compute_token_overlap_score(repo_name: str, paper_title: str) -> float:
    """Compute Jaccard similarity of token sets.

    Args:
        repo_name: Repository name
        paper_title: Paper title

    Returns:
        Score from 0 to 1
    """
    repo_tokens = normalize_text(repo_name)
    title_tokens = normalize_text(paper_title)

    if not repo_tokens or not title_tokens:
        return 0.0

    intersection = len(repo_tokens & title_tokens)
    union = len(repo_tokens | title_tokens)

    return intersection / union if union > 0 else 0.0


def compute_fuzzy_score(repo_name: str, paper_title: str) -> float:
    """Compute Levenshtein ratio between normalized strings.

    Args:
        repo_name: Repository name
        paper_title: Paper title

    Returns:
        Score from 0 to 1
    """
    # Join tokens back to strings for fuzzy comparison
    repo_normalized = " ".join(sorted(normalize_text(repo_name)))
    title_normalized = " ".join(sorted(normalize_text(paper_title)))

    if not repo_normalized or not title_normalized:
        return 0.0

    return levenshtein_ratio(repo_normalized, title_normalized)


def compute_name_similarity(
    repo_name: Optional[str],
    paper_title: Optional[str],
    threshold: float = 0.45,
    containment_weight: float = 0.4,
    overlap_weight: float = 0.4,
    fuzzy_weight: float = 0.2,
) -> NameSimilarityResult:
    """Compute weighted name similarity between repo name and paper title.

    Args:
        repo_name: Repository name
        paper_title: Paper title
        threshold: Minimum score to consider a match (default: 0.45)
        containment_weight: Weight for containment score (default: 0.4)
        overlap_weight: Weight for token overlap score (default: 0.4)
        fuzzy_weight: Weight for fuzzy score (default: 0.2)

    Returns:
        NameSimilarityResult with match status and scores
    """
    # Check if we can run this heuristic
    if not repo_name:
        return NameSimilarityResult(
            matched=False,
            skipped=True,
            skip_reason="no_repo_name",
        )

    if not paper_title:
        return NameSimilarityResult(
            matched=False,
            skipped=True,
            skip_reason="no_paper_title",
        )

    # Compute individual scores
    containment = compute_containment_score(repo_name, paper_title)
    overlap = compute_token_overlap_score(repo_name, paper_title)
    fuzzy = compute_fuzzy_score(repo_name, paper_title)

    # Compute weighted final score
    final_score = (
        containment_weight * containment +
        overlap_weight * overlap +
        fuzzy_weight * fuzzy
    )

    return NameSimilarityResult(
        matched=final_score >= threshold,
        score=final_score,
        containment_score=containment,
        token_overlap_score=overlap,
        fuzzy_score=fuzzy,
    )
