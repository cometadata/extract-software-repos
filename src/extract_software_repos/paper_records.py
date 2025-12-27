"""Paper record extraction for different metadata formats."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


@dataclass
class PaperInfo:
    """Extracted paper information."""
    doi: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    arxiv_id: Optional[str] = None


def normalize_doi(doi: str) -> str:
    """Normalize a DOI to a canonical form.

    - Lowercase
    - Strip URL prefixes (https://doi.org/, etc.)

    Args:
        doi: Raw DOI string

    Returns:
        Normalized DOI
    """
    doi = doi.lower()

    # Strip common URL prefixes
    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]
    for prefix in prefixes:
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break

    return doi


def extract_paper_info_datacite(record: Dict[str, Any]) -> PaperInfo:
    """Extract paper info from a DataCite record.

    DataCite records have structure:
    {
        "id": "10.xxxx/...",
        "attributes": {
            "doi": "10.xxxx/...",
            "titles": [{"title": "..."}],
            "creators": [{"name": "...", "givenName": "...", "familyName": "...", "nameType": "Personal"}],
            "alternateIdentifiers": [{"alternateIdentifier": "...", "alternateIdentifierType": "arXiv"}],
            "identifiers": [{"identifier": "...", "identifierType": "arXiv"}],
        }
    }

    Args:
        record: DataCite record dict

    Returns:
        Extracted PaperInfo
    """
    attrs = record.get("attributes", {})

    doi = attrs.get("doi") or record.get("id")

    titles = attrs.get("titles", [])
    title = titles[0]["title"] if titles else None

    authors = []
    for creator in attrs.get("creators", []):
        if creator.get("nameType") != "Personal":
            continue

        if creator.get("givenName") and creator.get("familyName"):
            name = f"{creator['givenName']} {creator['familyName']}"
        else:
            name = creator.get("name", "")
            if ", " in name:
                parts = name.split(", ", 1)
                name = f"{parts[1]} {parts[0]}"

        if name:
            authors.append(name)

    arxiv_id = None

    for alt_id in attrs.get("alternateIdentifiers", []):
        if alt_id.get("alternateIdentifierType") == "arXiv":
            arxiv_id = alt_id.get("alternateIdentifier")
            break

    if not arxiv_id:
        for ident in attrs.get("identifiers", []):
            if ident.get("identifierType") == "arXiv":
                arxiv_id = ident.get("identifier")
                break

    return PaperInfo(
        doi=doi,
        title=title,
        authors=authors,
        arxiv_id=arxiv_id,
    )


# Registry of record type extractors
RECORD_EXTRACTORS = {
    "datacite": extract_paper_info_datacite,
}


def extract_paper_info(record: Dict[str, Any], record_type: str = "datacite") -> PaperInfo:
    """Extract paper info from a record using the specified type.

    Args:
        record: Record dict
        record_type: Type of record ("datacite", future: "crossref", "openalex")

    Returns:
        Extracted PaperInfo

    Raises:
        ValueError: If record_type is unknown
    """
    extractor = RECORD_EXTRACTORS.get(record_type)
    if extractor is None:
        raise ValueError(f"Unknown record type: {record_type}. Known types: {list(RECORD_EXTRACTORS.keys())}")

    return extractor(record)


def load_papers_for_dois(
    records_path: Path,
    dois_needed: Set[str],
    record_type: str = "datacite",
) -> List[PaperInfo]:
    """Load paper records matching specific DOIs using DuckDB.

    Uses DuckDB to efficiently query large JSONL/JSONL.gz files,
    loading only records that match the requested DOIs.

    Args:
        records_path: Path to JSONL or JSONL.gz file
        dois_needed: Set of normalized DOIs to load
        record_type: Record format type

    Returns:
        List of PaperInfo for matching DOIs
    """
    import duckdb

    if not dois_needed:
        return []

    logger.info(f"Loading papers for {len(dois_needed):,} DOIs from {records_path}")

    dois_list = list(dois_needed)

    if record_type == "datacite":
        query = """
            SELECT *
            FROM read_json_auto(?, maximum_object_size=104857600, ignore_errors=true)
            WHERE LOWER(COALESCE(attributes.doi, id)) IN (SELECT UNNEST(?::VARCHAR[]))
        """
    else:
        raise ValueError(f"Unknown record type: {record_type}")

    conn = duckdb.connect(":memory:")

    try:
        result = conn.execute(query, [str(records_path), dois_list]).fetchall()
        columns = [desc[0] for desc in conn.description]
    finally:
        conn.close()

    logger.info(f"Found {len(result):,} matching paper records")

    extractor = RECORD_EXTRACTORS.get(record_type)
    if extractor is None:
        raise ValueError(f"Unknown record type: {record_type}")

    papers = []
    for row in result:
        record = dict(zip(columns, row))
        papers.append(extractor(record))

    return papers
