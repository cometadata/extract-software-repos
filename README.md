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

Check if extracted URLs actually exist. Uses optimized validators:
- **GitHub**: GraphQL API (batched, 100 repos/query)
- **Package registries**: Async HTTP (PyPI, npm, CRAN, etc.)
- **Other git hosts**: Parallel git ls-remote

**Requires** `GITHUB_TOKEN` environment variable (no scopes needed for public repos):

```bash
export GITHUB_TOKEN=ghp_your_token_here
extract-software-repos validate enrichments.jsonl -o validated.jsonl
```

**Options:**
- `-o, --output PATH` - Output file (default: `<input>_validated.jsonl`)
- `-w, --workers INT` - Threads for git ls-remote (default: 50)
- `--http-concurrency INT` - Max concurrent HTTP requests (default: 100)
- `-t, --timeout INT` - Timeout per request in seconds (default: 5)
- `--checkpoint PATH` - Checkpoint file for resume (default: `validation_cache.jsonl`)
- `--ignore-checkpoint` - Start fresh, ignore existing checkpoint
- `--wait-for-ratelimit` - Auto-wait when GitHub rate limited (up to 1 hour)
- `--keep-invalid` - Include invalid URLs in output (marked as invalid)

**Combined Validation + Promotion:**

You can run validation and promotion in a single step, which uses a single GitHub API call:

```bash
extract-software-repos validate enrichments.jsonl \
  --promote \
  --records /path/to/records.jsonl.gz \
  --github-cache cache.jsonl \
  -o validated_promoted.jsonl
```

**Promotion Options (when using --promote):**
- `--records PATH` - Paper records file (required with --promote)
- `--record-type` - Record format: `datacite` (default)
- `--promotion-threshold INT` - Signals required (default: 2)
- `--name-similarity-threshold FLOAT` - Name match threshold (default: 0.45)
- `--batch-size INT` - GitHub API batch size (default: 50)
- `--github-cache PATH` - Cache file for GitHub data (saves/resumes fetching)

**Resume:** If interrupted, re-run the same command. Cached URLs are skipped.

### Promote to isSupplementedBy

Promote validated repo links from `References` to `isSupplementedBy` when heuristics indicate the repo is the paper's official implementation.

**Requires** `GITHUB_TOKEN` with `repo` and `user:email` scopes for the author matching:

```bash
export GITHUB_TOKEN=ghp_your_token_here
extract-software-repos promote validated.jsonl \
  --records /path/to/datacite_records.jsonl.gz \
  -o promoted.jsonl
```

**Promotion heuristics used:**
- arXiv ID detection: Paper's arXiv ID found in README or repo description
- Name similarity: Repo name matches paper title
- Author matching: Repo contributors match paper authors using the [evamxb/dev-author-em-clf](https://huggingface.co/evamxb/dev-author-em-clf) model from [sci-soft-models](https://github.com/evamaxfield/sci-soft-models) ([doi:10.5281/zenodo.17401862](https://doi.org/10.5281/zenodo.17401862))

Promotion requires 2+ matching signals by default.

**Options:**
- `-o, --output PATH` - Output file (default: `<input>_promoted.jsonl`)
- `--records PATH` - DataCite paper records (required)
- `--record-type` - Record format: `datacite` (default)
- `--promotion-threshold INT` - Signals required (default: 2)
- `--name-similarity-threshold FLOAT` - Name match threshold (default: 0.45)
- `--batch-size INT` - GitHub API batch size (default: 50)
- `--github-cache PATH` - Cache file for GitHub data (saves/resumes fetching)

**Resume:** Use `--github-cache` to save GitHub API responses. Re-running with the same cache file skips already-fetched repos.

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

# 2. Validate + promote in one step (recommended)
extract-software-repos validate enrichments.jsonl \
  --promote \
  --records records.jsonl.gz \
  --github-cache github_cache.jsonl \
  -o enrichments_final.jsonl

# Or run separately:
# 2a. Validate URLs exist
# extract-software-repos validate enrichments.jsonl -o enrichments_valid.jsonl
# 2b. Promote implementation repos
# extract-software-repos promote enrichments_valid.jsonl --records records.jsonl.gz -o enrichments_final.jsonl

# 3. Use with datacite-enrichment to merge into records
datacite-enrich merge records.jsonl.gz enrichments_final.jsonl -o merged.jsonl
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