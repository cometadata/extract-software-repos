# extract-software-repos

Extract and validate software repository URLs from text and DataCite records.

## Overview

This tool scans text (abstracts, full-text papers) for references to software repositories and package registries, then validates that those URLs actually exist. It outputs enrichment records compatible with [datacite-enrichment](https://github.com/cometadata/datacite-enrichment) for merging into DataCite metadata.

## Installation

```bash
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

## Usage

### Extract from DataCite Records

Scan record abstracts for software URLs:

```bash
extract-software-repos extract-software records.jsonl.gz -o enrichments.jsonl
```

**Options:**
- `-o, --output PATH` - Output file (default: `software_enrichments.jsonl`)
- `--log-level` - DEBUG, INFO, WARNING, ERROR

### Extract from Full-Text (Parquet)

Extract URLs from arXiv full-text papers in Parquet format:

```bash
extract-software-repos extract-urls arxiv.parquet -o urls.jsonl
```

**Options:**
- `-o, --output PATH` - Output file (default: `<input>_urls.jsonl`)
- `-c, --chunk-size INT` - Rows per chunk (default: 50000)
- `--id-field` - Column with arXiv ID (default: `relative_path`)
- `--content-field` - Column with text (default: `content`)
- `--heal-markdown` - Preprocess text through markdown healing before extraction

### Heal Markdown Text

Clean malformed PDF-extracted text before or during extraction:

```bash
# Heal during extraction
extract-software-repos extract-urls arxiv.parquet -o urls.jsonl --heal-markdown

# Heal first, then extract
extract-software-repos heal-text arxiv.parquet -o arxiv_healed.parquet
extract-software-repos extract-urls arxiv_healed.parquet -o urls.jsonl
```

**heal-text Options:**
- `-o, --output PATH` - Output file (default: `<input>_healed.parquet`)
- `--content-field` - Column with text (auto-detected)
- `-c, --chunk-size INT` - Rows per chunk (default: 1000)

### Validate URLs

Check if extracted URLs actually exist:

```bash
extract-software-repos validate enrichments.jsonl -o validated.jsonl
```

**Options:**
- `-o, --output PATH` - Output file (default: `<input>_validated.jsonl`)
- `-w, --workers INT` - Parallel workers (default: 10)
- `-t, --timeout INT` - Timeout per URL in seconds (default: 10)
- `--keep-invalid` - Include invalid URLs in output (marked as invalid)

**Validation Methods:**
- Git repositories: `git ls-remote`
- Package registries: API/HTTP HEAD checks
- Archives: HTTP HEAD requests

## Output Format

### Enrichment Records (extract-software)

```json
{
  "doi": "10.1234/example",
  "action": "insert_child",
  "field": "relatedIdentifiers",
  "enrichedValue": {
    "relatedIdentifier": "https://github.com/user/repo",
    "relatedIdentifierType": "URL",
    "relationType": "IsSupplementedBy"
  },
  "sources": [{"name": "COMET Project", ...}],
  "created": "2024-01-01T00:00:00Z",
  ...
}
```

### URL Extraction Results (extract-urls)

```json
{
  "arxiv_id": "2308.11197",
  "doi": "10.48550/arxiv.2308.11197",
  "urls": [
    {"url": "https://github.com/user/repo", "type": "github"},
    {"url": "https://pypi.org/project/package", "type": "pypi"}
  ]
}
```

## Pipeline Example

```bash
# 1. Extract URLs from full-text (with healing for better extraction)
extract-software-repos extract-urls arxiv.parquet -o urls.jsonl --heal-markdown

# 2. Validate URLs exist
extract-software-repos validate urls.jsonl -o urls_valid.jsonl

# 3. Use with datacite-enrichment to merge into records
datacite-enrich merge records.jsonl.gz enrichments.jsonl -o merged.jsonl
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check .
```