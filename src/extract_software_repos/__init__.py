"""Extract and validate software repository URLs."""

__version__ = "0.1.0"

from .extraction import (
    extract_software_urls,
    extract_urls_with_types,
    preprocess_text,
    extract_base_repo,
    is_duplicate,
    COMBINED_URL_PATTERN,
)

from .polars_extraction import (
    preprocess_content,
    extract_urls_polars_df,
    process_parquet_polars,
)

from .healing import (
    heal_text,
    heal_parquet_parallel,
)

from .processing import (
    process_record,
    parse_arxiv_id,
    derive_doi,
)

from .validation import (
    validate_url,
    ValidationResult,
    ValidationStats,
)

__all__ = [
    # Original extraction
    "extract_software_urls",
    "extract_urls_with_types",
    "preprocess_text",
    "extract_base_repo",
    "is_duplicate",
    "COMBINED_URL_PATTERN",
    # Polars extraction
    "preprocess_content",
    "extract_urls_polars_df",
    "process_parquet_polars",
    # Healing
    "heal_text",
    "heal_parquet_parallel",
    # Processing
    "process_record",
    "parse_arxiv_id",
    "derive_doi",
    # Validation
    "validate_url",
    "ValidationResult",
    "ValidationStats",
]
