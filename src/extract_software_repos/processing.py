# src/extract_software_repos/processing.py
"""Process DataCite records and fulltext to extract software URLs."""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from .extraction import extract_software_urls, extract_urls_with_types, is_duplicate

logger = logging.getLogger(__name__)


def normalize_doi(doi: str) -> str:
    """Normalize DOI to lowercase without URL prefix."""
    doi = doi.lower().strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def _iso_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_enrichment(doi: str, url: str) -> dict:
    """Create an enrichment record for a software URL.

    Args:
        doi: Target DOI for the enrichment.
        url: Normalized software URL.

    Returns:
        Enrichment record dictionary.
    """
    timestamp = _iso_timestamp()

    return {
        "doi": doi,
        "sources": [
            {
                "name": "COMET Project",
                "contributorType": "Producer",
                "nameType": "Organizational",
            }
        ],
        "processResources": [
            {
                "relatedIdentifier": None,
                "relatedIdentifierType": "DOI",
                "relationType": "IsDescribedBy",
                "resourceTypeGeneral": "Workflow",
            }
        ],
        "action": "insert_child",
        "field": "relatedIdentifiers",
        "originalValue": None,
        "enrichedValue": {
            "relatedIdentifier": url,
            "relatedIdentifierType": "URL",
            "relationType": "IsSupplementedBy",
        },
        "created": timestamp,
        "updated": timestamp,
        "produced": timestamp,
    }


def process_record(record: dict) -> List[dict]:
    """Process a DataCite record and extract software URL enrichments.

    Args:
        record: DataCite record dictionary.

    Returns:
        List of enrichment record dictionaries.
    """
    enrichments = []

    doi = record.get("id") or record.get("attributes", {}).get("doi")
    if not doi:
        return enrichments

    attributes = record.get("attributes", {})
    existing_identifiers = attributes.get("relatedIdentifiers", []) or []

    urls_added: Set[str] = set()

    descriptions = attributes.get("descriptions", []) or []

    for desc in descriptions:
        desc_type = desc.get("descriptionType", "")
        if desc_type not in ("Abstract", "Other"):
            continue

        text = desc.get("description", "")
        if not text:
            continue

        urls = extract_software_urls(text)

        for url in urls:
            if url in urls_added:
                continue

            if is_duplicate(url, existing_identifiers):
                continue

            enrichment = create_enrichment(doi, url)
            enrichments.append(enrichment)
            urls_added.add(url)

    return enrichments


ARXIV_ID_PATTERN = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?\.md$")


def parse_arxiv_id(filename: str) -> Optional[str]:
    """Extract arxiv ID from filename, stripping version suffix.

    Args:
        filename: Filename like "2308.11197v3.md"

    Returns:
        Arxiv ID without version (e.g., "2308.11197") or None if invalid.
    """
    match = ARXIV_ID_PATTERN.match(filename)
    if match:
        return match.group(1)
    return None


def derive_doi(arxiv_id: str) -> str:
    """Derive DOI from arxiv ID.

    Args:
        arxiv_id: Arxiv ID like "2308.11197"

    Returns:
        DOI like "10.48550/arxiv.2308.11197"
    """
    return f"10.48550/arxiv.{arxiv_id}"


def process_paper(filename: str, content: str) -> Optional[Dict]:
    """Process a single paper and extract URLs.

    Args:
        filename: Paper filename (e.g., "2308.11197v3.md")
        content: Full text content of the paper.

    Returns:
        Dict with arxiv_id, doi, and urls list, or None if no URLs found.
    """
    if not content:
        return None

    arxiv_id = parse_arxiv_id(filename)
    if not arxiv_id:
        return None

    urls = extract_urls_with_types(content)
    if not urls:
        return None

    return {
        "arxiv_id": arxiv_id,
        "doi": derive_doi(arxiv_id),
        "urls": urls,
    }


def process_parquet(
    parquet_path: Union[str, Path],
    batch_size: int = 1000,
    id_field: str = "relative_path",
    content_field: str = "content",
) -> Iterator[Dict]:
    """Stream process a parquet file, yielding papers with URLs.

    Args:
        parquet_path: Path to parquet file.
        batch_size: Number of rows per batch.
        id_field: Column containing arxiv ID/filename.
        content_field: Column containing text content.

    Yields:
        Dicts with arxiv_id, doi, and urls for papers with URLs.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[id_field, content_field]):
        batch_dict = batch.to_pydict()
        filenames = batch_dict[id_field]
        contents = batch_dict[content_field]

        for filename, content in zip(filenames, contents):
            result = process_paper(filename, content or "")
            if result:
                yield result


def process_parquet_with_progress(
    parquet_path: Union[str, Path],
    progress_callback,
    batch_size: int = 1000,
    id_field: str = "relative_path",
    content_field: str = "content",
) -> Tuple[List[Dict], Dict]:
    """Process parquet file with progress callback.

    Args:
        parquet_path: Path to parquet file.
        progress_callback: Callable receiving (papers_processed, papers_with_urls, total_urls).
        batch_size: Number of rows per batch.
        id_field: Column containing arxiv ID/filename.
        content_field: Column containing text content.

    Returns:
        Tuple of (results list, stats dict).
    """
    import pyarrow.parquet as pq

    stats = {
        "total_papers": 0,
        "papers_with_urls": 0,
        "total_urls": 0,
        "urls_by_type": {},
    }
    results = []

    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=batch_size, columns=[id_field, content_field]):
        batch_dict = batch.to_pydict()
        filenames = batch_dict[id_field]
        contents = batch_dict[content_field]

        for filename, content in zip(filenames, contents):
            stats["total_papers"] += 1
            result = process_paper(filename, content or "")
            if result:
                stats["papers_with_urls"] += 1
                stats["total_urls"] += len(result["urls"])
                for url_info in result["urls"]:
                    url_type = url_info["type"]
                    stats["urls_by_type"][url_type] = stats["urls_by_type"].get(url_type, 0) + 1
                results.append(result)

            progress_callback(stats["total_papers"], stats["papers_with_urls"], stats["total_urls"])

    return results, stats
