#!/usr/bin/env python
"""Benchmark extraction performance."""

import time
from pathlib import Path
import tempfile


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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark extraction performance")
    parser.add_argument("input_file", type=Path, help="Parquet file to benchmark")
    parser.add_argument("--sample", type=int, help="Sample size (default: use all)")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File not found: {args.input_file}")
        return 1

    print("Running extraction benchmark...")
    print(f"Input: {args.input_file}")
    if args.sample:
        print(f"Sample size: {args.sample}")
    print()

    result = benchmark_polars(args.input_file, args.sample)

    print(f"Results:")
    print(f"  Rows processed:    {result['rows']:,}")
    print(f"  Time:              {result['elapsed_seconds']:.2f}s")
    print(f"  Throughput:        {result['rows_per_second']:.0f} rows/sec")
    print(f"  Papers with URLs:  {result['papers_with_urls']:,}")
    print(f"  Total URLs found:  {result['urls_found']:,}")


if __name__ == "__main__":
    main()
