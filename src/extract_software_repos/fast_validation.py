# src/extract_software_repos/fast_validation.py
"""Fast validation orchestrator with deduplication and routing."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .checkpoint import CheckpointManager
from .github_graphql import GitHubGraphQLValidator, RateLimitExceeded
from .async_validators import AsyncHTTPValidator
from .validation import validate_git_repo

logger = logging.getLogger(__name__)


def categorize_urls(urls: List[str]) -> Dict[str, List[str]]:
    """Categorize URLs by their type.

    Returns:
        Dict mapping category to list of URLs
    """
    categories: Dict[str, List[str]] = {
        "github": [],
        "gitlab": [],
        "bitbucket": [],
        "codeberg": [],
        "pypi": [],
        "npm": [],
        "cran": [],
        "bioconductor": [],
        "zenodo": [],
        "figshare": [],
        "codeocean": [],
        "software_heritage": [],
        "other": [],
    }

    for url in urls:
        url_lower = url.lower()
        if "github.com" in url_lower:
            categories["github"].append(url)
        elif "gitlab" in url_lower:
            categories["gitlab"].append(url)
        elif "bitbucket" in url_lower:
            categories["bitbucket"].append(url)
        elif "codeberg.org" in url_lower:
            categories["codeberg"].append(url)
        elif "pypi.org" in url_lower:
            categories["pypi"].append(url)
        elif "npmjs.com" in url_lower:
            categories["npm"].append(url)
        elif "cran.r-project.org" in url_lower:
            categories["cran"].append(url)
        elif "bioconductor.org" in url_lower:
            categories["bioconductor"].append(url)
        elif "zenodo.org" in url_lower:
            categories["zenodo"].append(url)
        elif "figshare.com" in url_lower:
            categories["figshare"].append(url)
        elif "codeocean.com" in url_lower:
            categories["codeocean"].append(url)
        elif "softwareheritage.org" in url_lower:
            categories["software_heritage"].append(url)
        else:
            categories["other"].append(url)

    return categories


def deduplicate_urls(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, List[int]]]:
    """Extract unique URLs and track which records they appear in.

    Args:
        records: List of enrichment records

    Returns:
        (unique_urls, url_to_record_indices) tuple
    """
    unique_urls: List[str] = []
    seen: Set[str] = set()
    url_to_records: Dict[str, List[int]] = {}

    for i, record in enumerate(records):
        url = record.get("enrichedValue", {}).get("relatedIdentifier")
        if not url:
            continue

        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
            url_to_records[url] = []

        url_to_records[url].append(i)

    return unique_urls, url_to_records


class FastValidator:
    """Orchestrates fast validation across multiple validators."""

    def __init__(
        self,
        checkpoint_path: Path,
        workers: int = 50,
        http_concurrency: int = 100,
        timeout: int = 5,
        wait_for_ratelimit: bool = False,
    ):
        self.checkpoint = CheckpointManager(checkpoint_path)
        self.workers = workers
        self.http_concurrency = http_concurrency
        self.timeout = timeout
        self.wait_for_ratelimit = wait_for_ratelimit

        # Results accumulator
        self.results: Dict[str, Dict[str, Any]] = {}

    def validate_all(
        self,
        records: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Validate all URLs in records.

        Args:
            records: Enrichment records to validate
            progress_callback: Optional callback(stage, completed, total)

        Returns:
            Dict mapping URL to validation result
        """
        # Step 1: Deduplicate
        unique_urls, url_to_records = deduplicate_urls(records)
        logger.info(f"Deduplicated: {len(records)} records -> {len(unique_urls)} unique URLs")

        # Step 2: Filter out already-cached URLs
        cached = self.checkpoint.get_cached_urls()
        urls_to_validate = [u for u in unique_urls if u not in cached]
        logger.info(f"After cache filter: {len(urls_to_validate)} URLs to validate")

        # Copy cached results
        self.results = {url: cached[url] for url in unique_urls if url in cached}

        if not urls_to_validate:
            return self.results

        # Step 3: Categorize URLs
        categories = categorize_urls(urls_to_validate)

        # Step 4: Validate each category
        # GitHub (GraphQL batched)
        if categories["github"]:
            self._validate_github(
                categories["github"],
                lambda c, t: progress_callback("GitHub", c, t) if progress_callback else None,
            )

        # Other git hosts (threaded git ls-remote)
        git_urls = categories["gitlab"] + categories["bitbucket"] + categories["codeberg"]
        if git_urls:
            self._validate_git_repos(
                git_urls,
                lambda c, t: progress_callback("Git repos", c, t) if progress_callback else None,
            )

        # HTTP-based validators (async)
        http_urls = []
        for url_type in ["pypi", "npm", "cran", "bioconductor", "zenodo", "figshare", "codeocean", "software_heritage"]:
            for url in categories[url_type]:
                http_urls.append({"url": url, "type": url_type})

        if http_urls:
            asyncio.run(self._validate_http_async(
                http_urls,
                lambda c, t: progress_callback("HTTP", c, t) if progress_callback else None,
            ))

        return self.results

    def _validate_github(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Validate GitHub URLs using GraphQL API."""
        try:
            validator = GitHubGraphQLValidator()
        except ValueError as e:
            logger.error(f"GitHub validation skipped: {e}")
            # Mark all as failed due to missing token
            for url in urls:
                result = {"url": url, "valid": False, "method": "graphql", "error": "no_token"}
                self.results[url] = result
                self.checkpoint.save_result(url, False, "graphql", "no_token")
            return

        try:
            results = asyncio.run(validator.validate_urls(urls, progress_callback))

            for r in results:
                result = {"url": r.url, "valid": r.valid, "method": "graphql", "error": r.error}
                self.results[r.url] = result
                self.checkpoint.save_result(r.url, r.valid, "graphql", r.error)

        except RateLimitExceeded as e:
            if self.wait_for_ratelimit:
                # Wait and retry
                wait_seconds = (e.rate_limit.reset_at - e.rate_limit.reset_at).total_seconds()
                logger.info(f"Rate limit exceeded. Waiting {wait_seconds:.0f}s...")
                import time
                time.sleep(max(0, wait_seconds) + 5)
                self._validate_github(urls, progress_callback)
            else:
                raise

    def _validate_git_repos(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Validate non-GitHub git repos using git ls-remote."""
        total = len(urls)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(validate_git_repo, url, self.timeout): url
                for url in urls
            }

            for future in as_completed(futures):
                url = futures[future]
                is_valid, method, error = future.result()

                result = {"url": url, "valid": is_valid, "method": method, "error": error}
                self.results[url] = result
                self.checkpoint.save_result(url, is_valid, method, error)

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

    async def _validate_http_async(
        self,
        urls: List[Dict[str, str]],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Validate HTTP-based URLs asynchronously."""
        validator = AsyncHTTPValidator(
            max_concurrency=self.http_concurrency,
            timeout=self.timeout,
        )

        results = await validator.validate_urls(urls, progress_callback)

        batch = []
        for r in results:
            self.results[r["url"]] = r
            batch.append(r)

        self.checkpoint.save_batch(batch)
