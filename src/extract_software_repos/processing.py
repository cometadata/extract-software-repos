"""Process DataCite records to extract software URLs."""

import re
from datetime import datetime, timezone
from typing import List, Optional, Set

from .extraction import extract_software_urls, is_duplicate


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
    """Create an enrichment record for a software URL."""
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
    """Process a DataCite record and extract software URL enrichments."""
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
    """Extract arxiv ID from filename, stripping version suffix."""
    match = ARXIV_ID_PATTERN.match(filename)
    if match:
        return match.group(1)
    return None


def derive_doi(arxiv_id: str) -> str:
    """Derive DOI from arxiv ID."""
    return f"10.48550/arxiv.{arxiv_id}"
