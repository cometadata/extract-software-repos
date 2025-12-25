"""Polars-based URL extraction for high-performance processing."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import polars as pl

from .extraction import extract_urls_with_types
from .processing import parse_arxiv_id, derive_doi

# Domains that need spaced-character fixing
SPACED_DOMAINS = [
    ("github.com", r"g\s*i\s*t\s*h\s*u\s*b\s*\.\s*c\s*o\s*m"),
    ("gitlab.com", r"g\s*i\s*t\s*l\s*a\s*b\s*\.\s*c\s*o\s*m"),
    ("bitbucket.org", r"b\s*i\s*t\s*b\s*u\s*c\s*k\s*e\s*t\s*\.\s*o\s*r\s*g"),
    ("pypi.org", r"p\s*y\s*p\s*i\s*\.\s*o\s*r\s*g"),
    ("sourceforge.net", r"s\s*o\s*u\s*r\s*c\s*e\s*f\s*o\s*r\s*g\s*e\s*\.\s*n\s*e\s*t"),
]

# URL patterns for Polars str.extract_all() - each returns full URL match
POLARS_URL_PATTERNS = {
    # Code repositories
    "github": r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?",
    "gitlab": r"(?:https?://)?(?:www\.)?gitlab\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?",
    "bitbucket": r"(?:https?://)?(?:www\.)?bitbucket\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?",
    "sourceforge": r"(?:https?://)?(?:www\.)?sourceforge\.net/projects/[a-zA-Z0-9_-]+",
    "codeberg": r"(?:https?://)?(?:www\.)?codeberg\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?",
    # Package registries
    "pypi": r"(?:https?://)?pypi\.org/project/[a-zA-Z0-9_-]+",
    "cran": r"(?:https?://)?cran\.r-project\.org/package=[a-zA-Z0-9_.]+",
    "npm": r"(?:https?://)?(?:www\.)?npmjs\.com/package/[a-zA-Z0-9@/._-]+",
    "conda": r"(?:https?://)?anaconda\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+",
    "rubygems": r"(?:https?://)?rubygems\.org/gems/[a-zA-Z0-9_-]+",
    "cargo": r"(?:https?://)?crates\.io/crates/[a-zA-Z0-9_-]+",
    "packagist": r"(?:https?://)?packagist\.org/packages/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+",
    "bioconductor": r"(?:https?://)?bioconductor\.org/packages/[a-zA-Z0-9_.]+",
    # Archives
    "software_heritage": r"(?:https?://)?archive\.softwareheritage\.org/[^\s)\"'<>]+",
    "codeocean": r"(?:https?://)?codeocean\.com/capsule/[a-zA-Z0-9_-]+",
}

# Filter patterns for exclusions
EXCLUDED_PATHS_PATTERN = r"/(?:wiki|issues|pull|pulls|discussions|projects|actions|security|pulse|graphs|settings|stargazers|watchers|network|forks)(?:/|$)"

DATA_EXTENSIONS_PATTERN = r"\.(?:csv|json|xml|xlsx|xls|tsv|dat|txt|md|rst|pdf|doc|docx|png|jpg|jpeg|gif|svg|bmp|zip|tar|gz|rar|7z)$"

GITHUB_PAGES_PATTERN = r"[a-zA-Z0-9_-]+\.github\.io/"


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


def extract_urls_native(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs using Polars-native operations.

    Returns exploded DataFrame with one row per URL found.

    Args:
        df: Input DataFrame with id and content columns.
        id_col: Column containing document IDs.
        content_col: Column containing text content.

    Returns:
        DataFrame with columns: doc_id, url, type (one row per URL).
    """
    # Apply preprocessing
    df = df.with_columns(preprocess_content(content_col))

    # Extract URLs for each type and collect results
    all_extractions = []

    for url_type, pattern in POLARS_URL_PATTERNS.items():
        # Extract all matches for this pattern
        extracted = df.select([
            pl.col(id_col).alias("doc_id"),
            pl.col(content_col).str.extract_all(pattern).alias("raw_urls"),
            pl.lit(url_type).alias("type"),
        ])

        # Explode to one row per URL
        exploded = extracted.explode("raw_urls").rename({"raw_urls": "url"})

        # Filter out nulls (no matches)
        exploded = exploded.filter(pl.col("url").is_not_null())

        if not exploded.is_empty():
            all_extractions.append(exploded)

    # Combine all extractions
    if all_extractions:
        result = pl.concat(all_extractions)
    else:
        # Return empty DataFrame with correct schema
        result = pl.DataFrame({
            "doc_id": pl.Series([], dtype=pl.Utf8),
            "url": pl.Series([], dtype=pl.Utf8),
            "type": pl.Series([], dtype=pl.Utf8),
        })

    return result


def normalize_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize extracted URLs.

    - Add https:// prefix if missing
    - Strip www.
    - Extract user/repo for code hosting platforms
    - Strip .git suffix
    - Strip trailing punctuation

    Args:
        df: DataFrame with doc_id, url, type columns.

    Returns:
        DataFrame with normalized URLs.
    """
    # Code hosting platforms that need user/repo extraction
    code_hosts = ["github", "gitlab", "bitbucket", "codeberg"]

    result = df.with_columns([
        # First, clean up the URL
        pl.col("url")
            # Remove protocol and www for processing
            .str.replace(r"^https?://", "")
            .str.replace(r"^www\.", "")
            # Strip trailing punctuation
            .str.replace(r"[).,;:!?\"'>\]]+$", "")
            .alias("clean_url")
    ])

    # For code hosts, extract just user/repo
    result = result.with_columns([
        pl.when(pl.col("type").is_in(code_hosts))
        .then(
            # Extract domain/user/repo only
            pl.col("clean_url").str.extract(r"^([^/]+/[^/]+/[^/]+)")
        )
        .otherwise(pl.col("clean_url"))
        .alias("clean_url")
    ])

    # Strip .git suffix
    result = result.with_columns([
        pl.col("clean_url")
            .str.replace(r"\.git$", "")
            # Strip any remaining trailing punctuation
            .str.replace(r"[).,;:!?\"'>\]]+$", "")
            .alias("clean_url")
    ])

    # Add https:// prefix back
    result = result.with_columns([
        pl.concat_str([pl.lit("https://"), pl.col("clean_url")])
            .alias("url")
    ])

    # Drop temporary column
    result = result.drop("clean_url")

    return result


def filter_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out excluded URLs.

    Removes:
    - GitHub Pages URLs (.github.io)
    - Excluded paths (/wiki, /issues, /pull, etc.)
    - Data files in /blob/ or /raw/ paths

    Args:
        df: DataFrame with doc_id, url, type columns.

    Returns:
        Filtered DataFrame.
    """
    return df.filter(
        # Not GitHub Pages
        ~pl.col("url").str.contains(GITHUB_PAGES_PATTERN)
        # Not excluded paths
        & ~pl.col("url").str.contains(EXCLUDED_PATHS_PATTERN)
        # Not data files in blob/raw paths
        & ~(
            pl.col("url").str.contains(r"/(?:blob|raw)/")
            & pl.col("url").str.contains(DATA_EXTENSIONS_PATTERN)
        )
    )


def deduplicate_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate URLs per document.

    Uses base URL (without protocol/www) for deduplication.
    Keeps first occurrence of each unique base URL per document.

    Args:
        df: DataFrame with doc_id, url, type columns.

    Returns:
        Deduplicated DataFrame.
    """
    # Create base URL for deduplication
    result = df.with_columns([
        pl.col("url")
            .str.replace(r"^https?://", "")
            .str.replace(r"^www\.", "")
            .str.to_lowercase()
            .alias("base_url")
    ])

    # Keep first occurrence per doc per base_url
    result = result.unique(subset=["doc_id", "base_url"], keep="first")

    # Drop temporary column
    result = result.drop("base_url")

    return result


def regroup_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Regroup exploded URLs back into list per document.

    Args:
        df: DataFrame with doc_id, url, type columns (one row per URL).

    Returns:
        DataFrame with doc_id, urls columns (one row per document).
    """
    return df.group_by("doc_id").agg([
        pl.struct(["url", "type"]).alias("urls")
    ])


def extract_urls_polars_native(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs using fully native Polars pipeline.

    Complete pipeline: extract -> normalize -> filter -> deduplicate -> regroup.
    All operations run in Rust with automatic parallelization.

    Args:
        df: Input DataFrame with id and content columns.
        id_col: Column containing document IDs.
        content_col: Column containing text content.

    Returns:
        DataFrame with id and urls columns (list of {url, type} structs).
    """
    # Step 1: Extract all URLs (returns exploded DataFrame)
    extracted = extract_urls_native(df, id_col=id_col, content_col=content_col)

    if len(extracted) == 0:
        # No URLs found - return DataFrame with empty lists
        return df.select([
            pl.col(id_col).alias("id"),
        ]).with_columns([
            pl.lit([]).cast(pl.List(pl.Struct([
                pl.Field("url", pl.Utf8),
                pl.Field("type", pl.Utf8),
            ]))).alias("urls")
        ])

    # Step 2: Normalize URLs
    normalized = normalize_urls(extracted)

    # Step 3: Filter exclusions
    filtered = filter_urls(normalized)

    if len(filtered) == 0:
        # All URLs filtered out
        return df.select([
            pl.col(id_col).alias("id"),
        ]).with_columns([
            pl.lit([]).cast(pl.List(pl.Struct([
                pl.Field("url", pl.Utf8),
                pl.Field("type", pl.Utf8),
            ]))).alias("urls")
        ])

    # Step 4: Deduplicate per document
    deduped = deduplicate_urls(filtered)

    # Step 5: Regroup into lists per document
    regrouped = regroup_urls(deduped)

    # Join back to original to preserve docs with no URLs
    result = df.select([pl.col(id_col).alias("id")]).join(
        regrouped.rename({"doc_id": "id"}),
        on="id",
        how="left"
    )

    # Fill nulls with empty list
    result = result.with_columns([
        pl.col("urls").fill_null([])
    ])

    return result


def extract_urls_polars_df(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs from a Polars DataFrame.

    Uses fully native Polars pipeline for maximum performance.

    Args:
        df: Input DataFrame with id and content columns.
        id_col: Column containing document IDs.
        content_col: Column containing text content.

    Returns:
        DataFrame with id and urls columns.
    """
    return extract_urls_polars_native(df, id_col=id_col, content_col=content_col)


def process_parquet_polars(
    parquet_path: Union[str, Path],
    output_path: Union[str, Path],
    id_field: str = "relative_path",
    content_field: str = "content",
    chunk_size: int = 50000,
    heal_markdown: bool = False,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> dict:
    """Process parquet file with Polars for high-performance extraction.

    Args:
        parquet_path: Path to input parquet file.
        output_path: Path to output JSONL file.
        id_field: Column containing document IDs.
        content_field: Column containing text content.
        chunk_size: Rows per processing chunk.
        heal_markdown: Whether to heal markdown before extraction.
        progress_callback: Optional callback(papers_processed, papers_with_urls, total_urls).

    Returns:
        Stats dict with total_papers, papers_with_urls, total_urls, urls_by_type.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    stats = {
        "total_papers": 0,
        "papers_with_urls": 0,
        "total_urls": 0,
        "urls_by_type": {},
        "healing_warnings": 0,
    }

    # Read full dataframe
    df = pl.read_parquet(parquet_path)

    # Apply healing if requested
    if heal_markdown:
        from .healing import heal_text
        healed_content = []
        for content in df[content_field].to_list():
            if content:
                healed, warnings = heal_text(content)
                healed_content.append(healed)
                if warnings:
                    stats["healing_warnings"] += 1
            else:
                healed_content.append("")
        df = df.with_columns(pl.Series(content_field, healed_content))

    # Extract URLs
    result = extract_urls_polars_df(df, id_col=id_field, content_col=content_field)

    # Write results
    with open(output_path, "w", encoding="utf-8") as f:
        for row in result.iter_rows(named=True):
            doc_id = row["id"]
            urls = row["urls"]

            stats["total_papers"] += 1

            if urls:
                # Parse arxiv ID and derive DOI
                arxiv_id = parse_arxiv_id(doc_id) if doc_id else None

                if arxiv_id:
                    stats["papers_with_urls"] += 1
                    stats["total_urls"] += len(urls)

                    for url_info in urls:
                        url_type = url_info["type"]
                        stats["urls_by_type"][url_type] = stats["urls_by_type"].get(url_type, 0) + 1

                    record = {
                        "arxiv_id": arxiv_id,
                        "doi": derive_doi(arxiv_id),
                        "urls": urls,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if progress_callback:
                progress_callback(stats["total_papers"], stats["papers_with_urls"], stats["total_urls"])

    return stats
