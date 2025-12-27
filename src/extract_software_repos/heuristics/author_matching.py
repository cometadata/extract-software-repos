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


def load_author_matching_model(device: Optional[str] = None, batch_size: int = 32) -> Pipeline:
    """Load the author matching model from HuggingFace.

    Args:
        device: Device to use (auto-detected if None)
        batch_size: Batch size for inference (default: 32)

    Returns:
        HuggingFace text-classification pipeline
    """
    if device is None:
        device = find_device()

    logger.info(f"Loading author matching model on {device} with batch_size={batch_size}")

    return pipeline(
        "text-classification",
        model=AUTHOR_MATCHING_MODEL,
        device=device,
        batch_size=batch_size,
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


# Default limit for contributors to check (most relevant are usually first)
DEFAULT_MAX_CONTRIBUTORS = 20


def match_authors_to_contributors(
    contributors: List[ContributorInfo],
    paper_authors: List[str],
    loaded_model: Optional[Pipeline] = None,
    device: Optional[str] = None,
    max_contributors: int = DEFAULT_MAX_CONTRIBUTORS,
    early_exit: bool = True,
) -> AuthorMatchResult:
    """Match paper authors to repository contributors.

    Uses the evamxb/dev-author-em-clf model to predict if any author
    is the same person as any contributor.

    Args:
        contributors: List of GitHub contributor info
        paper_authors: List of paper author names
        loaded_model: Pre-loaded model pipeline (for efficiency)
        device: Device to use if loading model
        max_contributors: Maximum contributors to check (default: 20)
        early_exit: Stop after finding first match (default: True)

    Returns:
        AuthorMatchResult with match status and details
    """
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

    if loaded_model is None:
        model = load_author_matching_model(device)
    else:
        model = loaded_model

    # Limit contributors to reduce inference time
    # Core maintainers (most likely to be authors) are typically listed first
    limited_contributors = contributors[:max_contributors]

    if len(contributors) > max_contributors:
        logger.debug(f"Limiting contributors from {len(contributors)} to {max_contributors}")

    inputs = []
    pairs = []

    for contributor in limited_contributors:
        for author in paper_authors:
            text = format_matching_input(
                username=contributor.login,
                name=contributor.name,
                email=contributor.email,
                author_name=author,
            )
            inputs.append(text)
            pairs.append((contributor, author))

    logger.debug(f"Running author matching on {len(inputs)} pairs")

    # For early exit, process in smaller batches
    if early_exit and len(inputs) > 0:
        matches = []
        batch_size = 32

        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            batch_pairs = pairs[batch_start:batch_end]

            outputs = model(batch_inputs)

            for (contributor, author), output in zip(batch_pairs, outputs):
                if output["label"] == "match":
                    matches.append(AuthorMatchDetail(
                        contributor_login=contributor.login,
                        author_name=author,
                        confidence=output["score"],
                    ))
                    # Early exit: we only need one match for the signal
                    if early_exit:
                        logger.debug(f"Early exit: found match {contributor.login} <-> {author}")
                        return AuthorMatchResult(
                            matched=True,
                            matches=matches,
                        )

        return AuthorMatchResult(
            matched=len(matches) > 0,
            matches=matches,
        )

    # Non-early-exit path: process all at once
    outputs = model(inputs)

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
