"""Polars-based URL extraction for high-performance processing."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import polars as pl

from .processing import parse_arxiv_id, derive_doi

SPACED_DOMAINS = [
    ("github.com", r"g\s*i\s*t\s*h\s*u\s*b\s*\.\s*c\s*o\s*m"),
    ("gitlab.com", r"g\s*i\s*t\s*l\s*a\s*b\s*\.\s*c\s*o\s*m"),
    ("bitbucket.org", r"b\s*i\s*t\s*b\s*u\s*c\s*k\s*e\s*t\s*\.\s*o\s*r\s*g"),
    ("pypi.org", r"p\s*y\s*p\s*i\s*\.\s*o\s*r\s*g"),
    ("sourceforge.net", r"s\s*o\s*u\s*r\s*c\s*e\s*f\s*o\s*r\s*g\s*e\s*\.\s*n\s*e\s*t"),
]

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
    """
    expr = pl.col(col)
    expr = expr.str.replace_all(r"[\u00AD\u200B\u200C\u200D\uFEFF]", "")
    expr = expr.str.replace_all(r"(\w)-\s*\n\s*(\w)", "$1$2")
    expr = expr.str.replace_all(r"\s*/\s*", "/")
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
    """
    df = df.with_columns(preprocess_content(content_col))
    all_extractions = []

    for url_type, pattern in POLARS_URL_PATTERNS.items():
        extracted = df.select([
            pl.col(id_col).alias("doc_id"),
            pl.col(content_col).str.extract_all(pattern).alias("raw_urls"),
            pl.lit(url_type).alias("type"),
        ])
        exploded = extracted.explode("raw_urls").rename({"raw_urls": "url"})
        exploded = exploded.filter(pl.col("url").is_not_null())
        if not exploded.is_empty():
            all_extractions.append(exploded)

    if all_extractions:
        return pl.concat(all_extractions)
    return pl.DataFrame({
        "doc_id": pl.Series([], dtype=pl.Utf8),
        "url": pl.Series([], dtype=pl.Utf8),
        "type": pl.Series([], dtype=pl.Utf8),
    })


def normalize_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize extracted URLs.

    - Add https:// prefix if missing
    - Strip www.
    - Extract user/repo for code hosting platforms
    - Strip .git suffix
    - Strip trailing punctuation
    """
    code_hosts = ["github", "gitlab", "bitbucket", "codeberg"]

    result = df.with_columns([
        pl.col("url")
            .str.replace(r"^https?://", "")
            .str.replace(r"^www\.", "")
            .str.replace(r"[).,;:!?\"'>\]]+$", "")
            .alias("clean_url")
    ])

    result = result.with_columns([
        pl.when(pl.col("type").is_in(code_hosts))
        .then(pl.col("clean_url").str.extract(r"^([^/]+/[^/]+/[^/]+)"))
        .otherwise(pl.col("clean_url"))
        .alias("clean_url")
    ])

    result = result.with_columns([
        pl.col("clean_url")
            .str.replace(r"\.git$", "")
            .str.replace(r"[).,;:!?\"'>\]]+$", "")
            .alias("clean_url")
    ])

    result = result.with_columns([
        pl.concat_str([pl.lit("https://"), pl.col("clean_url")]).alias("url")
    ])

    return result.drop("clean_url")


def filter_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out excluded URLs (GitHub Pages, /wiki, /issues, data files in /blob/)."""
    return df.filter(
        ~pl.col("url").str.contains(GITHUB_PAGES_PATTERN)
        & ~pl.col("url").str.contains(EXCLUDED_PATHS_PATTERN)
        & ~(
            pl.col("url").str.contains(r"/(?:blob|raw)/")
            & pl.col("url").str.contains(DATA_EXTENSIONS_PATTERN)
        )
    )


def deduplicate_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate URLs per document using normalized base URL."""
    result = df.with_columns([
        pl.col("url")
            .str.replace(r"^https?://", "")
            .str.replace(r"^www\.", "")
            .str.to_lowercase()
            .alias("base_url")
    ])
    result = result.unique(subset=["doc_id", "base_url"], keep="first")
    return result.drop("base_url")


def regroup_urls(df: pl.DataFrame) -> pl.DataFrame:
    """Regroup exploded URLs back into list per document."""
    return df.group_by("doc_id").agg([
        pl.struct(["url", "type"]).alias("urls")
    ])


def extract_urls_polars_native(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs using fully native Polars pipeline.

    Pipeline: extract -> filter -> normalize -> deduplicate -> regroup.
    All operations run in Rust with automatic parallelization.
    """
    extracted = extract_urls_native(df, id_col=id_col, content_col=content_col)

    empty_result = df.select([
        pl.col(id_col).alias("id"),
    ]).with_columns([
        pl.lit([]).cast(pl.List(pl.Struct([
            pl.Field("url", pl.Utf8),
            pl.Field("type", pl.Utf8),
        ]))).alias("urls")
    ])

    if len(extracted) == 0:
        return empty_result

    filtered = filter_urls(extracted)

    if len(filtered) == 0:
        return empty_result

    normalized = normalize_urls(filtered)
    deduped = deduplicate_urls(normalized)
    regrouped = regroup_urls(deduped)

    # Join back to preserve docs with no URLs
    result = df.select([pl.col(id_col).alias("id")]).join(
        regrouped.rename({"doc_id": "id"}),
        on="id",
        how="left"
    )
    return result.with_columns([pl.col("urls").fill_null([])])


def extract_urls_polars_df(
    df: pl.DataFrame,
    id_col: str = "relative_path",
    content_col: str = "content",
) -> pl.DataFrame:
    """Extract URLs from a Polars DataFrame using native pipeline."""
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
    """Process parquet file with Polars for high-performance extraction."""
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)

    stats = {
        "total_papers": 0,
        "papers_with_urls": 0,
        "total_urls": 0,
        "urls_by_type": {},
        "healing_warnings": 0,
    }

    df = pl.read_parquet(parquet_path)

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

    result = extract_urls_polars_df(df, id_col=id_field, content_col=content_field)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in result.iter_rows(named=True):
            doc_id = row["id"]
            urls = row["urls"]

            stats["total_papers"] += 1

            if urls:
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
