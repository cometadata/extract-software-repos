"""arXiv ID detection heuristic."""

import re
from dataclasses import dataclass
from typing import Optional, Set


@dataclass
class ArxivDetectionResult:
    """Result of arXiv ID detection."""
    matched: bool
    location: Optional[str] = None  # "readme" or "description"
    found_id: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


# Patterns to extract arXiv IDs from text
ARXIV_PATTERNS = [
    # DOI format: 10.48550/arXiv.2308.11197
    re.compile(r'10\.48550/arXiv\.(\d{4}\.\d{4,5})', re.IGNORECASE),
    # Full URL: arxiv.org/abs/2308.11197 or arxiv.org/abs/hep-th/9901001
    re.compile(r'arxiv\.org/abs/(\d{4}\.\d{4,5})', re.IGNORECASE),
    re.compile(r'arxiv\.org/abs/([a-z-]+/\d{7})', re.IGNORECASE),
    # Bare new format: 2308.11197 or 2308.11197v2
    re.compile(r'\b(\d{4}\.\d{4,5})(?:v\d+)?\b'),
    # Old format: hep-th/9901001
    re.compile(r'\b([a-z-]+/\d{7})\b', re.IGNORECASE),
]

# Pattern to strip version suffix
VERSION_PATTERN = re.compile(r'v\d+$')


def normalize_arxiv_id(arxiv_id: str) -> str:
    """Normalize an arXiv ID to a canonical form.

    - Strips version suffix (v1, v2, etc.)
    - Lowercases old-format category prefixes
    - Extracts ID from DOI or URL format

    Args:
        arxiv_id: Raw arXiv ID in any format

    Returns:
        Normalized arXiv ID
    """
    # Handle DOI format
    if arxiv_id.startswith("10.48550"):
        match = re.search(r'arXiv\.(\d{4}\.\d{4,5})', arxiv_id, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)

    # Handle URL format
    if "arxiv.org" in arxiv_id.lower():
        match = re.search(r'abs/(\d{4}\.\d{4,5}|[a-z-]+/\d{7})', arxiv_id, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)

    # Strip version suffix
    arxiv_id = VERSION_PATTERN.sub('', arxiv_id)

    # Lowercase old-format categories
    if '/' in arxiv_id:
        arxiv_id = arxiv_id.lower()

    return arxiv_id


def extract_arxiv_ids_from_text(text: str) -> Set[str]:
    """Extract all arXiv IDs from text.

    Args:
        text: Text to search

    Returns:
        Set of normalized arXiv IDs found
    """
    ids = set()

    for pattern in ARXIV_PATTERNS:
        for match in pattern.finditer(text):
            raw_id = match.group(1)
            normalized = normalize_arxiv_id(raw_id)
            ids.add(normalized)

    return ids


def detect_arxiv_id(
    paper_arxiv_id: Optional[str],
    readme_content: Optional[str],
    description: Optional[str],
) -> ArxivDetectionResult:
    """Detect if the paper's arXiv ID appears in repo README or description.

    Args:
        paper_arxiv_id: The paper's arXiv ID (from DOI or metadata)
        readme_content: README file content (may be None)
        description: Repository description (may be None)

    Returns:
        ArxivDetectionResult with match status and location
    """
    # Check if we can run this heuristic
    if not paper_arxiv_id:
        return ArxivDetectionResult(
            matched=False,
            skipped=True,
            skip_reason="no_paper_arxiv_id",
        )

    if not readme_content and not description:
        return ArxivDetectionResult(
            matched=False,
            skipped=True,
            skip_reason="no_content",
        )

    # Normalize the paper's arXiv ID
    normalized_paper_id = normalize_arxiv_id(paper_arxiv_id)

    # Check README first (more likely to have detailed references)
    if readme_content:
        found_ids = extract_arxiv_ids_from_text(readme_content)
        if normalized_paper_id in found_ids:
            return ArxivDetectionResult(
                matched=True,
                location="readme",
                found_id=normalized_paper_id,
            )

    # Check description
    if description:
        found_ids = extract_arxiv_ids_from_text(description)
        if normalized_paper_id in found_ids:
            return ArxivDetectionResult(
                matched=True,
                location="description",
                found_id=normalized_paper_id,
            )

    return ArxivDetectionResult(matched=False)
