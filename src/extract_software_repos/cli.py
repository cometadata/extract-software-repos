# src/extract_software_repos/cli.py
"""CLI commands for software URL extraction and validation."""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import click

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def _stream_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSONL or JSONL.gz file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@click.group()
@click.version_option()
def cli():
    """Extract and validate software repository URLs."""
    pass


@cli.command("extract-software")
@click.argument("records_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default="software_enrichments.jsonl",
    help="Output file path (default: software_enrichments.jsonl)"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def extract_software(records_file: Path, output: Path, log_level: str):
    """Extract software URLs from record abstracts.

    RECORDS_FILE: Path to records.jsonl or records.jsonl.gz file

    Scans abstracts for software repository URLs (GitHub, PyPI, etc.)
    and creates enrichment records.

    Example:
        extract-software-repos extract-software records.jsonl.gz -o enrichments.jsonl
    """
    from .processing import process_record

    _setup_logging(log_level)

    click.echo(f"Extracting software URLs from {records_file}")

    total_records = 0
    total_enrichments = 0

    with open(output, "w", encoding="utf-8") as out_f:
        for record in _stream_jsonl(records_file):
            total_records += 1

            enrichments = process_record(record)
            for enrichment in enrichments:
                out_f.write(json.dumps(enrichment, ensure_ascii=False) + "\n")
                total_enrichments += 1

            if total_records % 10000 == 0:
                click.echo(f"  Processed {total_records:,} records, {total_enrichments:,} enrichments...")

    click.echo(f"\nExtraction complete!")
    click.echo(f"  Records processed: {total_records:,}")
    click.echo(f"  Enrichments created: {total_enrichments:,}")
    click.echo(f"  Output: {output}")


@cli.command("extract-urls")
@click.argument("parquet_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output JSONL file (default: <input>_urls.jsonl)"
)
@click.option(
    "--chunk-size", "-c",
    type=int,
    default=50000,
    help="Rows per processing chunk (default: 50000)"
)
@click.option(
    "--id-field",
    type=str,
    default="relative_path",
    help="Column containing arxiv ID (default: relative_path)"
)
@click.option(
    "--content-field",
    type=str,
    default="content",
    help="Column containing text (default: content)"
)
@click.option(
    "--heal-markdown",
    is_flag=True,
    help="Preprocess text through markdown healing before extraction"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def extract_urls(
    parquet_file: Path,
    output: Optional[Path],
    chunk_size: int,
    id_field: str,
    content_field: str,
    heal_markdown: bool,
    log_level: str,
):
    """Extract software URLs from full-text parquet file.

    PARQUET_FILE: Path to parquet file with arxiv content.

    Uses Polars for high-performance parallel processing.

    Example:
        extract-software-repos extract-urls arxiv.parquet -o urls.jsonl
        extract-software-repos extract-urls arxiv.parquet --heal-markdown
    """
    import polars as pl
    from tqdm import tqdm
    from .polars_extraction import process_parquet_polars

    _setup_logging(log_level)

    if output is None:
        output = parquet_file.parent / f"{parquet_file.stem}_urls.jsonl"

    heal_status = " (with markdown healing)" if heal_markdown else ""
    click.echo(f"Extracting URLs from {parquet_file}{heal_status}")
    click.echo(f"Output: {output}")

    # Get total rows for progress bar
    total_rows = pl.scan_parquet(parquet_file).select(pl.len()).collect().item()

    pbar = tqdm(total=total_rows, desc="Processing papers", unit="papers")
    last_processed = 0

    def progress_callback(papers_processed, papers_with_urls, total_urls):
        nonlocal last_processed
        pbar.update(papers_processed - last_processed)
        last_processed = papers_processed
        pbar.set_postfix({"with_urls": papers_with_urls, "urls": total_urls})

    stats = process_parquet_polars(
        parquet_file,
        output,
        id_field=id_field,
        content_field=content_field,
        chunk_size=chunk_size,
        heal_markdown=heal_markdown,
        progress_callback=progress_callback,
    )

    pbar.close()

    click.echo(f"\nExtraction complete!")
    click.echo(f"  Total papers: {stats['total_papers']:,}")
    click.echo(f"  Papers with URLs: {stats['papers_with_urls']:,}")
    click.echo(f"  Total URLs: {stats['total_urls']:,}")

    if heal_markdown and stats.get("healing_warnings", 0) > 0:
        click.echo(f"  Healing warnings: {stats['healing_warnings']:,} documents")

    if stats["urls_by_type"]:
        click.echo(f"  URLs by type:")
        for url_type, count in sorted(stats["urls_by_type"].items(), key=lambda x: -x[1]):
            click.echo(f"    {url_type}: {count:,}")

    click.echo(f"  Output: {output}")


@cli.command("validate")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output file (default: <input>_validated.jsonl)"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=10,
    help="Number of parallel workers (default: 10)"
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=10,
    help="Timeout per URL in seconds (default: 10)"
)
@click.option(
    "--keep-invalid",
    is_flag=True,
    help="Keep invalid URLs in output (marked as invalid)"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def validate(
    input_file: Path,
    output: Optional[Path],
    workers: int,
    timeout: int,
    keep_invalid: bool,
    log_level: str,
):
    """Validate extracted URLs against their sources.

    INPUT_FILE: Enrichment JSONL file to validate

    Checks if URLs actually exist using git ls-remote for repositories
    and API/HTTP checks for package registries.

    Example:
        extract-software-repos validate enrichments.jsonl -o validated.jsonl
    """
    from tqdm import tqdm
    from .validation import validate_url, ValidationStats

    _setup_logging(log_level)

    if output is None:
        output = input_file.parent / f"{input_file.stem}_validated.jsonl"

    click.echo(f"Validating URLs from {input_file}")
    click.echo(f"Workers: {workers}, Timeout: {timeout}s")

    # Load enrichments
    enrichments = []
    for line in open(input_file, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            enrichments.append(json.loads(line))

    click.echo(f"Loaded {len(enrichments):,} enrichments")

    stats = ValidationStats()
    valid_enrichments = []

    # Determine URL type from enrichment
    def get_url_type(enrichment: dict) -> str:
        url = enrichment.get("enrichedValue", {}).get("relatedIdentifier", "")
        url_lower = url.lower()
        if "github.com" in url_lower:
            return "github"
        elif "gitlab.com" in url_lower:
            return "gitlab"
        elif "bitbucket.org" in url_lower:
            return "bitbucket"
        elif "pypi.org" in url_lower:
            return "pypi"
        elif "npmjs.com" in url_lower:
            return "npm"
        elif "cran.r-project.org" in url_lower:
            return "cran"
        elif "bioconductor.org" in url_lower:
            return "bioconductor"
        else:
            return "unknown"

    for enrichment in tqdm(enrichments, desc="Validating"):
        url = enrichment.get("enrichedValue", {}).get("relatedIdentifier", "")
        url_type = get_url_type(enrichment)

        result = validate_url(url, url_type, timeout)
        stats.add_result(result)

        if result.is_valid or keep_invalid:
            enrichment["_validation"] = {
                "is_valid": result.is_valid,
                "method": result.method,
                "error": result.error,
            }
            valid_enrichments.append(enrichment)

    # Write output
    with open(output, "w", encoding="utf-8") as f:
        for enrichment in valid_enrichments:
            f.write(json.dumps(enrichment, ensure_ascii=False) + "\n")

    click.echo(f"\nValidation complete!")
    click.echo(stats.summary())
    click.echo(f"Output: {output} ({len(valid_enrichments):,} enrichments)")


@cli.command("heal-text")
@click.argument("parquet_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output parquet file (default: <input>_healed.parquet)"
)
@click.option(
    "--content-field",
    type=str,
    default=None,
    help="Column containing text (auto-detected if not specified)"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count)"
)
@click.option(
    "--chunk-size", "-c",
    type=int,
    default=1000,
    help="Rows per chunk (default: 1000)"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def heal_text_cmd(
    parquet_file: Path,
    output: Optional[Path],
    content_field: Optional[str],
    workers: Optional[int],
    chunk_size: int,
    log_level: str,
):
    """Heal malformed markdown in parquet file.

    PARQUET_FILE: Path to parquet file with text content.

    Uses multiprocessing for parallel healing across CPU cores.

    Example:
        extract-software-repos heal-text arxiv.parquet -o cleaned.parquet
        extract-software-repos heal-text arxiv.parquet --workers 8
    """
    import polars as pl
    from .healing import heal_parquet_parallel

    _setup_logging(log_level)

    if output is None:
        output = parquet_file.parent / f"{parquet_file.stem}_healed.parquet"

    click.echo(f"Healing text in {parquet_file}")
    click.echo(f"Output: {output}")

    # Auto-detect content field if not specified
    if content_field is None:
        df_schema = pl.read_parquet_schema(parquet_file)
        content_candidates = ['content', 'text', 'markdown', 'md', 'body']
        for candidate in content_candidates:
            if candidate in df_schema:
                content_field = candidate
                break

    if content_field is None:
        df_schema = pl.read_parquet_schema(parquet_file)
        click.echo(f"Error: Could not detect content column. Available: {list(df_schema.keys())}")
        raise SystemExit(1)

    click.echo(f"Using content column: {content_field}")

    # Get row count
    total_rows = pl.scan_parquet(parquet_file).select(pl.len()).collect().item()
    click.echo(f"Processing {total_rows:,} documents...")

    from multiprocessing import cpu_count
    actual_workers = workers or cpu_count()
    click.echo(f"Using {actual_workers} parallel workers")

    stats = heal_parquet_parallel(
        parquet_file,
        output,
        content_field=content_field,
        workers=workers,
        chunk_size=chunk_size,
    )

    click.echo(f"\nHealing complete!")
    click.echo(f"  Total documents:     {stats['total']:,}")
    click.echo(f"  Successfully healed: {stats['healed']:,}")
    click.echo(f"  With warnings:       {stats['warnings']:,}")
    click.echo(f"  Output: {output}")


if __name__ == "__main__":
    cli()
