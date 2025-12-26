"""Paper record extraction for different metadata formats."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


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
