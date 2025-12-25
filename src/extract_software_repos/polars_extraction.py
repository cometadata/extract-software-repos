"""Polars-based URL extraction for high-performance processing."""

import polars as pl

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
