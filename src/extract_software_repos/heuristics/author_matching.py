"""Author matching heuristic using adapted sci-soft-models approach.

This module adapts the developer-author entity matching from sci-soft-models.
The key constraint preserved is the exact input template format, which the
evamxb/dev-author-em-clf model was trained on.

Model: evamxb/dev-author-em-clf (DeBERTa-based text classifier)
Accuracy: 98.56% on original eval set
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

# Model identifier - must match the trained model
AUTHOR_MATCHING_MODEL = "evamxb/dev-author-em-clf"

# CRITICAL: This template format must be preserved exactly.
# The model was trained on this specific XML structure.
MODEL_INPUT_TEMPLATE = """
<developer-details>
    <username>{dev_username}</username>
    <name>{dev_name}</name>
    <email>{dev_email}</email>
</developer-details>

---

<author-details>
    <name>{author_name}</name>
</author-details>
""".strip()


@dataclass
class ContributorInfo:
    """GitHub contributor information."""
    login: str
    name: Optional[str] = None
    email: Optional[str] = None


@dataclass
class AuthorMatchDetail:
    """Details of a single author-contributor match."""
    contributor_login: str
    author_name: str
    confidence: float


@dataclass
class AuthorMatchResult:
    """Result of author matching."""
    matched: bool
    matches: List[AuthorMatchDetail] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None


def find_device() -> str:
    """Find the best available device for inference.

    Returns:
        Device string: "cuda:0", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_author_matching_model(device: Optional[str] = None) -> Pipeline:
    """Load the author matching model from HuggingFace.

    Args:
        device: Device to use (auto-detected if None)

    Returns:
        HuggingFace text-classification pipeline
    """
    if device is None:
        device = find_device()

    logger.info(f"Loading author matching model on {device}")

    return pipeline(
        "text-classification",
        model=AUTHOR_MATCHING_MODEL,
        device=device,
    )


def format_matching_input(
    username: str,
    name: Optional[str],
    email: Optional[str],
    author_name: str,
) -> str:
    """Format a developer-author pair for model input.

    CRITICAL: This format must match exactly what the model was trained on.

    Args:
        username: GitHub username
        name: Developer's display name (may be None)
        email: Developer's email (may be None)
        author_name: Paper author name

    Returns:
        Formatted input string
    """
    return MODEL_INPUT_TEMPLATE.format(
        dev_username=username,
        dev_name=name,
        dev_email=email,
        author_name=author_name,
    )


def match_authors_to_contributors(
    contributors: List[ContributorInfo],
    paper_authors: List[str],
    loaded_model: Optional[Pipeline] = None,
    device: Optional[str] = None,
) -> AuthorMatchResult:
    """Match paper authors to repository contributors.

    Uses the evamxb/dev-author-em-clf model to predict if any author
    is the same person as any contributor.

    Args:
        contributors: List of GitHub contributor info
        paper_authors: List of paper author names
        loaded_model: Pre-loaded model pipeline (for efficiency)
        device: Device to use if loading model

    Returns:
        AuthorMatchResult with match status and details
    """
    # Check if we can run this heuristic
    if not contributors:
        return AuthorMatchResult(
            matched=False,
            skipped=True,
            skip_reason="no_contributors",
        )

    if not paper_authors:
        return AuthorMatchResult(
            matched=False,
            skipped=True,
            skip_reason="no_authors",
        )

    # Load model if not provided
    if loaded_model is None:
        model = load_author_matching_model(device)
    else:
        model = loaded_model

    # Create all contributor-author pairs
    inputs = []
    pairs = []  # Track which pair each input corresponds to

    for contributor in contributors:
        for author in paper_authors:
            text = format_matching_input(
                username=contributor.login,
                name=contributor.name,
                email=contributor.email,
                author_name=author,
            )
            inputs.append(text)
            pairs.append((contributor, author))

    # Run inference
    logger.debug(f"Running author matching on {len(inputs)} pairs")
    outputs = model(inputs)

    # Extract matches
    matches = []
    for (contributor, author), output in zip(pairs, outputs):
        if output["label"] == "match":
            matches.append(AuthorMatchDetail(
                contributor_login=contributor.login,
                author_name=author,
                confidence=output["score"],
            ))

    return AuthorMatchResult(
        matched=len(matches) > 0,
        matches=matches,
    )
