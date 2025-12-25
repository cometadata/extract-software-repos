"""Polars-based URL extraction for high-performance processing."""

from typing import List, Dict

import polars as pl

from .extraction import extract_urls_with_types

# Domains that need spaced-character fixing
SPACED_DOMAINS = [
    ("github.com", r"g\s*i\s*t\s*h\s*u\s*b\s*\.\s*c\s*o\s*m"),
    ("gitlab.com", r"g\s*i\s*t\s*l\s*a\s*b\s*\.\s*c\s*o\s*m"),
    ("bitbucket.org", r"b\s*i\s*t\s*b\s*u\s*c\s*k\s*e\s*t\s*\.\s*o\s*r\s*g"),
    ("pypi.org", r"p\s*y\s*p\s*i\s*\.\s*o\s*r\s*g"),
    ("sourceforge.net", r"s\s*o\s*u\s*r\s*c\s*e\s*f\s*o\s*r\s*g\s*e\s*\.\s*n\s*e\s*t"),
]


def preprocess_content(col: str) -> pl.Expr:
    """Create Polars expression for text preprocessing.

    Handles PDF extraction artifacts:
    - Soft hyphens and zero-width characters
    - Hyphenated line breaks
    - Spaces around slashes
    - Spaced-out domain names

    Args:
        col: Column name to preprocess.

    Returns:
        Polars expression that preprocesses text.
    """
    expr = pl.col(col)

    # Remove soft hyphens and zero-width characters
    expr = expr.str.replace_all(r"[\u00AD\u200B\u200C\u200D\uFEFF]", "")

    # Fix hyphenated line breaks: word-\nrest -> wordrest
    expr = expr.str.replace_all(r"(\w)-\s*\n\s*(\w)", "$1$2")

    # Collapse spaces around slashes
    expr = expr.str.replace_all(r"\s*/\s*", "/")

    # Fix spaced domains
    for domain, pattern in SPACED_DOMAINS:
        expr = expr.str.replace_all(pattern, domain)

    return expr.alias(col)


def _extract_and_normalize_urls(text: str) -> List[Dict[str, str]]:
    """Extract and normalize URLs from text.

    Args:
        text: Text to search for URLs.

    Returns:
        List of dicts with 'url' and 'type' keys.
    """
    if not text:
        return []

    # Reuse existing extraction logic which handles all normalization and deduplication
    return extract_urls_with_types(text)


def extract_urls_polars_df(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs from a Polars DataFrame.

    Args:
        df: Input DataFrame with id and content columns.
        id_col: Column containing document IDs.
        content_col: Column containing text content.

    Returns:
        DataFrame with id and urls columns.
    """
    # Apply preprocessing
    df = df.with_columns(preprocess_content(content_col))

    # Extract URLs using map_elements (applies Python function row-wise)
    result = df.select([
        pl.col(id_col).alias("id"),
        pl.col(content_col)
          .map_elements(_extract_and_normalize_urls, return_dtype=pl.List(pl.Struct([
              pl.Field("url", pl.Utf8),
              pl.Field("type", pl.Utf8),
          ])))
          .alias("urls")
    ])

    return result
