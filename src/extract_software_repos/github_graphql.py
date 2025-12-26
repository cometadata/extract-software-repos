# src/extract_software_repos/github_graphql.py
"""GitHub GraphQL batch validation."""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
BATCH_SIZE = 100  # Max repos per query
MIN_BATCH_INTERVAL = 3.0  # Seconds between batches to stay under 2000 points/min


def parse_github_url(url: str) -> Optional[Tuple[str, str]]:
    """Extract owner and repo from GitHub URL.

    Returns (owner, repo) tuple or None if URL is invalid.
    """
    # Match github.com/owner/repo with optional trailing parts
    pattern = r"github\.com/([^/]+)/([^/?.#]+)"
    match = re.search(pattern, url)

    if not match:
        return None

    owner = match.group(1)
    repo = match.group(2)

    if repo.endswith(".git"):
        repo = repo[:-4]

    return (owner, repo)


def build_graphql_query(repos: List[Tuple[str, str]]) -> str:
    """Build a GraphQL query for multiple repositories.

    Args:
        repos: List of (owner, repo) tuples

    Returns:
        GraphQL query string
    """
    parts = []
    for i, (owner, repo) in enumerate(repos):
        # Escape quotes in names (shouldn't happen but be safe)
        owner_escaped = owner.replace('"', '\\"')
        repo_escaped = repo.replace('"', '\\"')
        parts.append(f'repo{i}: repository(owner: "{owner_escaped}", name: "{repo_escaped}") {{ id }}')

    return "query { " + " ".join(parts) + " }"


@dataclass
class RateLimitInfo:
    """GitHub API rate limit information."""
    remaining: int
    reset_at: datetime
    limit: int


@dataclass
class GitHubValidationResult:
    """Result of validating a GitHub URL."""
    url: str
    valid: bool
    error: Optional[str] = None


class GitHubGraphQLValidator:
    """Validates GitHub repositories using GraphQL API."""

    def __init__(
        self,
        token: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
        max_retries: int = 3,
    ):
        self.token = token or os.environ.get("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token required. Set GITHUB_TOKEN environment variable.")

        self.batch_size = batch_size
        self.max_retries = max_retries
        self.rate_limit: Optional[RateLimitInfo] = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _update_rate_limit(self, headers: Dict[str, str]) -> None:
        """Update rate limit info from response headers."""
        try:
            remaining = int(headers.get("X-RateLimit-Remaining", -1))
            reset_ts = int(headers.get("X-RateLimit-Reset", 0))
            limit = int(headers.get("X-RateLimit-Limit", 5000))

            if remaining < 0 or reset_ts == 0:
                return

            self.rate_limit = RateLimitInfo(
                remaining=remaining,
                reset_at=datetime.fromtimestamp(reset_ts, tz=timezone.utc),
                limit=limit,
            )
        except (ValueError, TypeError):
            pass

    async def validate_batch(
        self,
        urls: List[str],
        session: aiohttp.ClientSession,
    ) -> List[GitHubValidationResult]:
        """Validate a batch of GitHub URLs.

        Args:
            urls: List of GitHub URLs to validate
            session: aiohttp session to use

        Returns:
            List of validation results
        """
        url_to_repo: Dict[str, Tuple[str, str]] = {}
        results: List[GitHubValidationResult] = []

        for url in urls:
            parsed = parse_github_url(url)
            if parsed is None:
                results.append(GitHubValidationResult(url=url, valid=False, error="invalid_url"))
            else:
                url_to_repo[url] = parsed

        if not url_to_repo:
            return results

        repos = list(url_to_repo.values())
        url_list = list(url_to_repo.keys())
        query = build_graphql_query(repos)

        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    GITHUB_GRAPHQL_URL,
                    json={"query": query},
                    headers=self._get_headers(),
                ) as response:
                    self._update_rate_limit(dict(response.headers))

                    if response.status == 200:
                        data = await response.json()
                        return self._parse_response(url_list, data, results)
                    elif response.status == 401:
                        raise ValueError("Invalid GitHub token")
                    elif response.status == 403:
                        if self.rate_limit and self.rate_limit.remaining == 0:
                            raise RateLimitExceeded(self.rate_limit)
                        error_text = await response.text()
                        logger.warning(f"GitHub 403 (not rate limit): {error_text[:200]}")
                    elif response.status >= 500:
                        pass
                    else:
                        error_text = await response.text()
                        logger.warning(f"GitHub API error {response.status}: {error_text}")

                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(2 ** attempt)

        for url in url_list:
            results.append(GitHubValidationResult(url=url, valid=False, error="request_failed"))

        return results

    def _parse_response(
        self,
        urls: List[str],
        data: Dict[str, Any],
        existing_results: List[GitHubValidationResult],
    ) -> List[GitHubValidationResult]:
        """Parse GraphQL response into validation results."""
        results = existing_results.copy()
        response_data = data.get("data", {})
        errors = data.get("errors", [])

        # Build error lookup by path
        error_lookup: Dict[str, str] = {}
        for error in errors:
            path = error.get("path", [])
            if path:
                error_lookup[path[0]] = error.get("type", "error")

        for i, url in enumerate(urls):
            repo_key = f"repo{i}"
            repo_data = response_data.get(repo_key)

            if repo_data is not None:
                results.append(GitHubValidationResult(url=url, valid=True))
            else:
                error_type = error_lookup.get(repo_key, "not_found")
                results.append(GitHubValidationResult(url=url, valid=False, error=error_type))

        return results

    async def validate_urls(
        self,
        urls: List[str],
        progress_callback=None,
        batch_callback=None,
    ) -> List[GitHubValidationResult]:
        """Validate multiple GitHub URLs in batches.

        Args:
            urls: List of GitHub URLs
            progress_callback: Optional callback(completed, total)
            batch_callback: Optional callback(batch_results) called after each batch

        Returns:
            List of validation results
        """
        results: List[GitHubValidationResult] = []
        total = len(urls)
        last_batch_time = 0.0

        async with aiohttp.ClientSession() as session:
            for i in range(0, total, self.batch_size):
                elapsed = time.monotonic() - last_batch_time
                if elapsed < MIN_BATCH_INTERVAL and last_batch_time > 0:
                    await asyncio.sleep(MIN_BATCH_INTERVAL - elapsed)

                last_batch_time = time.monotonic()
                batch = urls[i:i + self.batch_size]
                batch_results = await self.validate_batch(batch, session)
                results.extend(batch_results)

                if batch_callback:
                    batch_callback(batch_results)

                if progress_callback:
                    progress_callback(len(results), total)

        return results


class RateLimitExceeded(Exception):
    """Raised when GitHub rate limit is exceeded."""

    def __init__(self, rate_limit: RateLimitInfo):
        self.rate_limit = rate_limit
        super().__init__(f"Rate limit exceeded. Resets at {rate_limit.reset_at}")
