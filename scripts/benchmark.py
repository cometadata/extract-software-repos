#!/usr/bin/env python
"""Benchmark extraction performance comparing legacy vs Polars implementations."""

import time
from pathlib import Path
import tempfile


def benchmark_legacy(input_path: Path, sample_size: int = None):
    """Benchmark legacy pyarrow/pandas extraction."""
    import polars as pl
    from extract_software_repos.processing import process_parquet_with_progress

    # Read and sample data
    df = pl.read_parquet(input_path)
    if sample_size:
        df = df.head(sample_size)

    actual_size = len(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "input.parquet"
        df.write_parquet(tmp_input)

        start = time.perf_counter()
        results, stats = process_parquet_with_progress(
            tmp_input,
            progress_callback=lambda *args: None,
            id_field="relative_path",
            content_field="content",
        )
        elapsed = time.perf_counter() - start

        return {
            "method": "legacy (pyarrow)",
            "rows": actual_size,
            "elapsed_seconds": elapsed,
            "rows_per_second": actual_size / elapsed,
            "urls_found": stats["total_urls"],
            "papers_with_urls": stats["papers_with_urls"],
        }


def benchmark_polars(input_path: Path, sample_size: int = None):
    """Benchmark Polars extraction."""
    import polars as pl
    from extract_software_repos.polars_extraction import process_parquet_polars

    # Read data
    df = pl.read_parquet(input_path)
    if sample_size:
        df = df.head(sample_size)

    actual_size = len(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "input.parquet"
        output_path = Path(tmpdir) / "output.jsonl"
        df.write_parquet(tmp_input)

        start = time.perf_counter()
        stats = process_parquet_polars(
            tmp_input,
            output_path,
            id_field="relative_path",
            content_field="content",
        )
        elapsed = time.perf_counter() - start

        return {
            "method": "polars",
            "rows": actual_size,
            "elapsed_seconds": elapsed,
            "rows_per_second": actual_size / elapsed,
            "urls_found": stats["total_urls"],
            "papers_with_urls": stats["papers_with_urls"],
        }


def print_result(result: dict):
    """Print benchmark result."""
    print(f"  Time:              {result['elapsed_seconds']:.2f}s")
    print(f"  Throughput:        {result['rows_per_second']:.1f} rows/sec")
    print(f"  Papers with URLs:  {result['papers_with_urls']:,}")
    print(f"  Total URLs found:  {result['urls_found']:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark extraction performance")
    parser.add_argument("input_file", type=Path, help="Parquet file to benchmark")
    parser.add_argument("--sample", type=int, help="Sample size (default: use all)")
    parser.add_argument("--polars-only", action="store_true", help="Only run Polars benchmark")
    parser.add_argument("--legacy-only", action="store_true", help="Only run legacy benchmark")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return 1

    print("=" * 60)
    print("EXTRACTION BENCHMARK")
    print("=" * 60)
    print(f"Input: {args.input_file}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    print()

    results = []

    # Run legacy benchmark
    if not args.polars_only:
        print("Legacy (pyarrow) implementation...")
        try:
            legacy_result = benchmark_legacy(args.input_file, args.sample)
            results.append(legacy_result)
            print_result(legacy_result)
        except ImportError as e:
            print(f"  Skipped: {e}")
            print("  (Install pyarrow: uv sync --all-extras)")
        print()

    # Run Polars benchmark
    if not args.legacy_only:
        print("Polars implementation...")
        polars_result = benchmark_polars(args.input_file, args.sample)
        results.append(polars_result)
        print_result(polars_result)
        print()

    # Print comparison
    if len(results) == 2:
        legacy = results[0]
        polars = results[1]
        speedup = legacy["elapsed_seconds"] / polars["elapsed_seconds"]

        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"Rows processed:      {polars['rows']:,}")
        print(f"Legacy time:         {legacy['elapsed_seconds']:.2f}s")
        print(f"Polars time:         {polars['elapsed_seconds']:.2f}s")
        print(f"Speedup:             {speedup:.1f}x")
        print()

        # Verify same results
        if legacy["urls_found"] == polars["urls_found"]:
            print("✓ Both implementations found the same number of URLs")
        else:
            print(f"⚠ URL count differs: legacy={legacy['urls_found']}, polars={polars['urls_found']}")


if __name__ == "__main__":
    main()
