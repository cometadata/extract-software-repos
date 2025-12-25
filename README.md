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

### Extract from Full-Text (Parquet)

Extract URLs from arXiv full-text papers in Parquet format (default):

```bash
extract-software-repos extract papers.parquet -o enrichments.jsonl
```

**Options:**
- `-o, --output PATH` - Output file (default: `<input>_enrichments.jsonl`)
- `-c, --chunk-size INT` - Rows per chunk (default: 50000)
- `--id-field` - Column with arXiv ID (default: `relative_path`)
- `--content-field` - Column with text (default: `content`)
- `--heal-fulltext` - Preprocess text through healing before extraction

### Extract from DataCite Records

Scan record abstracts for software URLs:

```bash
extract-software-repos extract records.jsonl.gz --from-datacite-abstract -o enrichments.jsonl
```

### Heal Fulltext

Clean malformed PDF-extracted text before or during extraction:

```bash
# Heal during extraction
extract-software-repos extract papers.parquet -o enrichments.jsonl --heal-fulltext

# Heal first, then extract
extract-software-repos heal-fulltext papers.parquet -o papers_healed.parquet
extract-software-repos extract papers_healed.parquet -o enrichments.jsonl
```

**heal-fulltext Options:**
- `-o, --output PATH` - Output file (default: `<input>_healed.parquet`)
- `--content-field` - Column with text (auto-detected)
- `-w, --workers INT` - Parallel workers (default: CPU count)
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

The `extract` command outputs enrichment records compatible with [datacite-enrichment](https://github.com/cometadata/datacite-enrichment):

```json
{
  "doi": "10.48550/arxiv.2308.11197",
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

One record is output per URL found.

## Pipeline Example

```bash
# 1. Extract URLs from full-text (with healing for better extraction)
extract-software-repos extract papers.parquet -o enrichments.jsonl --heal-fulltext

# 2. Validate URLs exist
extract-software-repos validate enrichments.jsonl -o enrichments_valid.jsonl

# 3. Use with datacite-enrichment to merge into records
datacite-enrich merge records.jsonl.gz enrichments_valid.jsonl -o merged.jsonl
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