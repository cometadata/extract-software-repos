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


@cli.command("extract")
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--from-datacite-abstract",
    "source_type",
    flag_value="datacite",
    help="Extract from DataCite record abstracts (JSONL input)"
)
@click.option(
    "--from-parquet",
    "source_type",
    flag_value="parquet",
    default=True,
    help="Extract from parquet fulltext (default)"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output JSONL file (default: <input>_enrichments.jsonl)"
)
@click.option(
    "--chunk-size", "-c",
    type=int,
    default=50000,
    help="Rows per chunk for parquet (default: 50000)"
)
@click.option(
    "--id-field",
    type=str,
    default="relative_path",
    help="Column containing ID (default: relative_path)"
)
@click.option(
    "--content-field",
    type=str,
    default="content",
    help="Column containing text (default: content)"
)
@click.option(
    "--heal-fulltext",
    is_flag=True,
    help="Preprocess text through markdown healing before extraction"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level"
)
def extract(
    input_file: Path,
    source_type: str,
    output: Optional[Path],
    chunk_size: int,
    id_field: str,
    content_field: str,
    heal_fulltext: bool,
    log_level: str,
):
    """Extract software repository URLs from text.

    INPUT_FILE: Path to input file (parquet or JSONL)

    Examples:
        extract-software-repos extract papers.parquet -o enrichments.jsonl
        extract-software-repos extract records.jsonl.gz --from-datacite-abstract
    """
    _setup_logging(log_level)

    if output is None:
        output = input_file.parent / f"{input_file.stem}_enrichments.jsonl"

    if source_type == "datacite":
        _extract_datacite(input_file, output)
    else:
        _extract_parquet(input_file, output, chunk_size, id_field, content_field, heal_fulltext)


def _extract_datacite(input_file: Path, output: Path):
    """Extract from DataCite record abstracts."""
    from .processing import process_record

    click.echo(f"Extracting from DataCite abstracts: {input_file}")
    click.echo(f"Output: {output}")

    total_records = 0
    total_enrichments = 0

    with open(output, "w", encoding="utf-8") as out_f:
        for record in _stream_jsonl(input_file):
            total_records += 1

            enrichments = process_record(record)
            for enrichment in enrichments:
                out_f.write(json.dumps(enrichment, ensure_ascii=False) + "\n")
                total_enrichments += 1

            if total_records % 10000 == 0:
                click.echo(f"  Processed {total_records:,} records, {total_enrichments:,} enrichments...")

    click.echo("\nExtraction complete!")
    click.echo(f"  Records processed: {total_records:,}")
    click.echo(f"  Enrichments created: {total_enrichments:,}")
    click.echo(f"  Output: {output}")


def _extract_parquet(
    input_file: Path,
    output: Path,
    chunk_size: int,
    id_field: str,
    content_field: str,
    heal_fulltext: bool,
):
    """Extract from parquet fulltext."""
    import polars as pl
    from tqdm import tqdm
    from .polars_extraction import process_parquet_polars

    heal_status = " (with fulltext healing)" if heal_fulltext else ""
    click.echo(f"Extracting from parquet: {input_file}{heal_status}")
    click.echo(f"Output: {output}")

    total_rows = pl.scan_parquet(input_file).select(pl.len()).collect().item()

    pbar = tqdm(total=total_rows, desc="Processing", unit="docs")
    last_processed = 0

    def progress_callback(papers_processed, papers_with_urls, total_urls):
        nonlocal last_processed
        pbar.update(papers_processed - last_processed)
        last_processed = papers_processed
        pbar.set_postfix({"with_urls": papers_with_urls, "urls": total_urls})

    stats = process_parquet_polars(
        input_file,
        output,
        id_field=id_field,
        content_field=content_field,
        chunk_size=chunk_size,
        heal_fulltext=heal_fulltext,
        progress_callback=progress_callback,
    )

    pbar.close()

    click.echo("\nExtraction complete!")
    click.echo(f"  Total documents: {stats['total_papers']:,}")
    click.echo(f"  With URLs: {stats['papers_with_urls']:,}")
    click.echo(f"  Total enrichments: {stats['total_urls']:,}")

    if heal_fulltext and stats.get("healing_warnings", 0) > 0:
        click.echo(f"  Healing warnings: {stats['healing_warnings']:,} documents")

    if stats["urls_by_type"]:
        click.echo("  By type:")
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
    default=50,
    help="Number of parallel workers for git ls-remote (default: 50)"
)
@click.option(
    "--http-concurrency",
    type=int,
    default=100,
    help="Max concurrent HTTP requests (default: 100)"
)
@click.option(
    "--timeout", "-t",
    type=int,
    default=5,
    help="Timeout per request in seconds (default: 5)"
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    default=Path("validation_cache.jsonl"),
    help="Checkpoint file for resume (default: validation_cache.jsonl)"
)
@click.option(
    "--ignore-checkpoint",
    is_flag=True,
    help="Start fresh, ignore existing checkpoint"
)
@click.option(
    "--wait-for-ratelimit",
    is_flag=True,
    help="Wait when rate limited instead of exiting"
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
    http_concurrency: int,
    timeout: int,
    checkpoint: Path,
    ignore_checkpoint: bool,
    wait_for_ratelimit: bool,
    keep_invalid: bool,
    log_level: str,
):
    """Validate extracted URLs against their sources.

    INPUT_FILE: Enrichment JSONL file to validate

    Uses GitHub GraphQL API (batched) for GitHub URLs, async HTTP for
    package registries, and git ls-remote for other git hosts.

    Requires GITHUB_TOKEN environment variable for GitHub validation.

    Example:
        export GITHUB_TOKEN=ghp_xxxx
        extract-software-repos validate enrichments.jsonl -o validated.jsonl
    """
    import os
    from tqdm import tqdm
    from .fast_validation import FastValidator, deduplicate_urls
    from .github_graphql import RateLimitExceeded

    _setup_logging(log_level)

    if output is None:
        output = input_file.parent / f"{input_file.stem}_validated.jsonl"

    # Check for GitHub token
    if not os.environ.get("GITHUB_TOKEN"):
        click.echo("Warning: GITHUB_TOKEN not set. GitHub validation will fail.", err=True)

    # Handle checkpoint
    if ignore_checkpoint and checkpoint.exists():
        click.echo(f"Ignoring existing checkpoint: {checkpoint}")
        checkpoint.unlink()

    click.echo(f"Validating URLs from {input_file}")
    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Workers: {workers}, HTTP concurrency: {http_concurrency}, Timeout: {timeout}s")

    # Load records
    records = []
    for line in open(input_file, "r", encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))

    click.echo(f"Loaded {len(records):,} enrichment records")

    # Deduplicate for display
    unique_urls, url_to_records = deduplicate_urls(records)
    click.echo(f"Unique URLs: {len(unique_urls):,}")

    # Set up validator
    validator = FastValidator(
        checkpoint_path=checkpoint,
        workers=workers,
        http_concurrency=http_concurrency,
        timeout=timeout,
        wait_for_ratelimit=wait_for_ratelimit,
    )

    # Progress bars for each stage
    stages = {}

    def progress_callback(stage: str, completed: int, total: int):
        if stage not in stages:
            stages[stage] = tqdm(total=total, desc=stage, unit="urls")
        stages[stage].n = completed
        stages[stage].refresh()

    # Validate
    try:
        results = validator.validate_all(records, progress_callback)
    except RateLimitExceeded as e:
        click.echo(f"\nRate limit exceeded. Resets at {e.rate_limit.reset_at}", err=True)
        click.echo(f"Progress saved to {checkpoint}. Re-run to continue.", err=True)
        click.echo("Use --wait-for-ratelimit to auto-wait next time.", err=True)
        raise SystemExit(1)
    finally:
        for pbar in stages.values():
            pbar.close()

    # Apply results to records
    valid_count = 0
    invalid_count = 0
    output_records = []

    for record in records:
        url = record.get("enrichedValue", {}).get("relatedIdentifier")
        if not url:
            continue

        result = results.get(url)
        if not result:
            continue

        if result["valid"]:
            valid_count += 1
        else:
            invalid_count += 1

        if result["valid"] or keep_invalid:
            record["_validation"] = {
                "is_valid": result["valid"],
                "method": result["method"],
                "error": result.get("error"),
            }
            output_records.append(record)

    # Write output
    with open(output, "w", encoding="utf-8") as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    click.echo("\nValidation complete!")
    click.echo(f"  Valid URLs: {valid_count:,}")
    click.echo(f"  Invalid URLs: {invalid_count:,}")
    click.echo(f"  Output: {output} ({len(output_records):,} records)")


@cli.command("heal-fulltext")
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

    click.echo("\nHealing complete!")
    click.echo(f"  Total documents:     {stats['total']:,}")
    click.echo(f"  Successfully healed: {stats['healed']:,}")
    click.echo(f"  With warnings:       {stats['warnings']:,}")
    click.echo(f"  Output: {output}")


if __name__ == "__main__":
    cli()
