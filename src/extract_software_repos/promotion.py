"""Promotion engine for upgrading repo links to isSupplementedBy."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .github_graphql import GitHubPromotionFetcher, GitHubPromotionData, parse_github_url
from .heuristics.arxiv_detection import ArxivDetectionResult, detect_arxiv_id
from .heuristics.name_similarity import NameSimilarityResult, compute_name_similarity
from .heuristics.author_matching import (
    AuthorMatchResult,
    ContributorInfo,
    match_authors_to_contributors,
    load_author_matching_model,
)
from .paper_records import PaperInfo, normalize_doi

logger = logging.getLogger(__name__)


@dataclass
class PromotionResult:
    """Result of promotion evaluation for a single record."""
    promoted: bool
    signals: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    original_relation: str = "References"
    skipped: bool = False
    skip_reason: Optional[str] = None


def evaluate_promotion(
    arxiv_result: ArxivDetectionResult,
    name_result: NameSimilarityResult,
    author_result: AuthorMatchResult,
    threshold: int = 2,
) -> PromotionResult:
    """Evaluate whether a record should be promoted based on heuristic results.

    Args:
        arxiv_result: Result from arXiv ID detection
        name_result: Result from name similarity
        author_result: Result from author matching
        threshold: Minimum number of signals required (default: 2)

    Returns:
        PromotionResult with decision and evidence
    """
    signals = []
    evidence = {}

    if arxiv_result.matched:
        signals.append(f"arxiv_id_in_{arxiv_result.location}")
        evidence["arxiv_id_found"] = arxiv_result.found_id
        evidence["arxiv_id_location"] = arxiv_result.location
    elif arxiv_result.skipped:
        evidence["arxiv_skipped"] = arxiv_result.skip_reason

    if name_result.matched:
        signals.append("name_similarity")
    evidence["name_similarity_score"] = name_result.score
    evidence["name_containment_score"] = name_result.containment_score
    evidence["name_token_overlap_score"] = name_result.token_overlap_score
    evidence["name_fuzzy_score"] = name_result.fuzzy_score
    if name_result.skipped:
        evidence["name_skipped"] = name_result.skip_reason

    if author_result.matched:
        signals.append("author_match")
        evidence["author_matches"] = [
            {
                "contributor": m.contributor_login,
                "author": m.author_name,
                "confidence": m.confidence,
            }
            for m in author_result.matches
        ]
    elif author_result.skipped:
        evidence["author_skipped"] = author_result.skip_reason

    return PromotionResult(
        promoted=len(signals) >= threshold,
        signals=signals,
        evidence=evidence,
    )


class PromotionEngine:
    """Engine for evaluating and applying promotions to enrichment records."""

    def __init__(
        self,
        github_token: Optional[str] = None,
        promotion_threshold: int = 2,
        name_similarity_threshold: float = 0.45,
        batch_size: int = 50,
    ):
        self.github_token = github_token
        self.promotion_threshold = promotion_threshold
        self.name_similarity_threshold = name_similarity_threshold
        self.batch_size = batch_size
        self._author_model = None

    def _build_paper_index(self, papers: List[PaperInfo]) -> Dict[str, PaperInfo]:
        """Build a DOI-indexed lookup for papers.

        Args:
            papers: List of paper info

        Returns:
            Dict mapping normalized DOI to PaperInfo
        """
        index = {}
        for paper in papers:
            if paper.doi:
                normalized = normalize_doi(paper.doi)
                index[normalized] = paper
        return index

    def _filter_promotable_records(
        self,
        records: List[Dict[str, Any]],
        paper_index: Dict[str, PaperInfo],
    ) -> List[Dict[str, Any]]:
        """Filter records to those eligible for promotion.

        Criteria:
        - Valid GitHub URL
        - Has matching paper record

        Args:
            records: Enrichment records
            paper_index: DOI-indexed paper lookup

        Returns:
            Filtered list of promotable records
        """
        promotable = []
        for record in records:
            validation = record.get("_validation", {})
            if not validation.get("is_valid"):
                continue

            url = record.get("enrichedValue", {}).get("relatedIdentifier", "")
            if not parse_github_url(url):
                continue

            doi = record.get("doi", "")
            if doi:
                normalized_doi = normalize_doi(doi)
                if normalized_doi in paper_index:
                    promotable.append(record)

        return promotable

    def _get_author_model(self):
        """Get or load the author matching model."""
        if self._author_model is None:
            self._author_model = load_author_matching_model()
        return self._author_model

    def _evaluate_record(
        self,
        record: Dict[str, Any],
        paper: PaperInfo,
        promotion_data: GitHubPromotionData,
    ) -> PromotionResult:
        """Evaluate a single record for promotion.

        Args:
            record: Enrichment record
            paper: Matched paper info
            promotion_data: GitHub data for the repo

        Returns:
            PromotionResult
        """
        url = record.get("enrichedValue", {}).get("relatedIdentifier", "")
        parsed = parse_github_url(url)
        repo_name = parsed[1] if parsed else None

        arxiv_result = detect_arxiv_id(
            paper_arxiv_id=paper.arxiv_id,
            readme_content=promotion_data.readme_content,
            description=promotion_data.description,
        )

        name_result = compute_name_similarity(
            repo_name=repo_name,
            paper_title=paper.title,
            threshold=self.name_similarity_threshold,
        )

        contributors = [
            ContributorInfo(
                login=c["login"],
                name=c.get("name"),
                email=c.get("email"),
            )
            for c in promotion_data.contributors
            if c.get("login")
        ]
        author_result = match_authors_to_contributors(
            contributors=contributors,
            paper_authors=paper.authors,
            loaded_model=self._get_author_model(),
        )

        return evaluate_promotion(
            arxiv_result=arxiv_result,
            name_result=name_result,
            author_result=author_result,
            threshold=self.promotion_threshold,
        )

    async def promote_records(
        self,
        records: List[Dict[str, Any]],
        papers: List[PaperInfo],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        cached_github_data: Optional[Dict[str, GitHubPromotionData]] = None,
        github_data_callback: Optional[Callable[[List[GitHubPromotionData]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Promote eligible records based on heuristics.

        Args:
            records: Enrichment records (with _validation)
            papers: Paper info list
            progress_callback: Optional callback(stage, completed, total)
            cached_github_data: Pre-fetched GitHub data (url -> data)
            github_data_callback: Called with newly fetched data for caching

        Returns:
            Updated records with _promotion field and updated relationType
        """
        paper_index = self._build_paper_index(papers)
        logger.info(f"Built paper index with {len(paper_index)} entries")

        promotable = self._filter_promotable_records(records, paper_index)
        logger.info(f"Found {len(promotable)} promotable records out of {len(records)}")

        if not promotable:
            return records

        all_urls = list(set(
            r.get("enrichedValue", {}).get("relatedIdentifier", "")
            for r in promotable
        ))

        url_to_data = dict(cached_github_data) if cached_github_data else {}
        urls_to_fetch = [u for u in all_urls if u not in url_to_data]

        if urls_to_fetch:
            logger.info(f"Fetching {len(urls_to_fetch)} URLs ({len(all_urls) - len(urls_to_fetch)} cached)")
            fetcher = GitHubPromotionFetcher(token=self.github_token, batch_size=self.batch_size)

            def github_progress(completed, total):
                if progress_callback:
                    progress_callback("Fetching GitHub data", completed, total)

            def batch_callback(batch_results):
                if github_data_callback:
                    github_data_callback(batch_results)

            new_data = await fetcher.fetch_promotion_data(
                urls_to_fetch,
                progress_callback=github_progress,
                batch_callback=batch_callback,
            )
            for d in new_data:
                url_to_data[d.url] = d
        else:
            logger.info(f"All {len(all_urls)} URLs found in cache")

        results = {}
        total_to_evaluate = len(promotable)

        for i, record in enumerate(promotable):
            url = record.get("enrichedValue", {}).get("relatedIdentifier", "")

            if url in results:
                continue

            doi = normalize_doi(record.get("doi", ""))
            paper = paper_index.get(doi)
            promotion_data = url_to_data.get(url)

            if paper and promotion_data and not promotion_data.fetch_error:
                result = self._evaluate_record(record, paper, promotion_data)
                results[url] = result
            else:
                results[url] = PromotionResult(
                    promoted=False,
                    skipped=True,
                    skip_reason=promotion_data.fetch_error if promotion_data else "no_promotion_data",
                )

            if progress_callback:
                progress_callback("Evaluating heuristics", i + 1, total_to_evaluate)

        output_records = []
        promoted_count = 0

        for record in records:
            url = record.get("enrichedValue", {}).get("relatedIdentifier", "")
            result = results.get(url)

            if result:
                original_relation = record.get("enrichedValue", {}).get("relationType", "References")
                result.original_relation = original_relation

                record["_promotion"] = {
                    "promoted": result.promoted,
                    "original_relation": original_relation,
                    "signals": result.signals,
                    "evidence": result.evidence,
                }

                if result.promoted:
                    record["enrichedValue"]["relationType"] = "isSupplementedBy"
                    promoted_count += 1

            output_records.append(record)

        logger.info(f"Promoted {promoted_count} records to isSupplementedBy")
        return output_records

    def promote_records_sync(
        self,
        records: List[Dict[str, Any]],
        papers: List[PaperInfo],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        cached_github_data: Optional[Dict[str, GitHubPromotionData]] = None,
        github_data_callback: Optional[Callable[[List[GitHubPromotionData]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for promote_records."""
        return asyncio.run(self.promote_records(
            records, papers, progress_callback, cached_github_data, github_data_callback
        ))
