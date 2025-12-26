# src/extract_software_repos/validation.py
"""Validate software URLs by checking if resources actually exist."""

import asyncio
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests

from .async_validators import AsyncHTTPValidator
from .checkpoint import CheckpointManager
from .github_graphql import GitHubGraphQLValidator, RateLimitExceeded

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT = 10
DEFAULT_GIT_TIMEOUT = 10
USER_AGENT = "Extract-Software-Repos-Validator/1.0"


@dataclass
class ValidationResult:
    """Result of URL validation."""

    url: str
    url_type: str
    is_valid: bool
    method: str
    error: Optional[str] = None


@dataclass
class ValidationStats:
    """Statistics from URL validation."""

    total_urls: int = 0
    valid_urls: int = 0
    invalid_urls: int = 0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)

    def add_result(self, result: ValidationResult) -> None:
        """Record a validation result."""
        self.total_urls += 1

        if result.url_type not in self.by_type:
            self.by_type[result.url_type] = {"valid": 0, "invalid": 0}

        if result.is_valid:
            self.valid_urls += 1
            self.by_type[result.url_type]["valid"] += 1
        else:
            self.invalid_urls += 1
            self.by_type[result.url_type]["invalid"] += 1
            error_key = f"{result.url_type}:{result.error or 'unknown'}"
            self.errors_by_type[error_key] = self.errors_by_type.get(error_key, 0) + 1

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Total URLs: {self.total_urls:,}",
            f"Valid URLs: {self.valid_urls:,}",
            f"Invalid URLs: {self.invalid_urls:,}",
        ]
        if self.by_type:
            lines.append("By type:")
            for url_type, counts in sorted(self.by_type.items()):
                lines.append(f"  {url_type}: {counts['valid']:,} valid, {counts['invalid']:,} invalid")
        if self.errors_by_type:
            lines.append("Errors:")
            for error_type, count in sorted(self.errors_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"  {error_type}: {count:,}")
        return "\n".join(lines)


def validate_git_repo(url: str, timeout: int = DEFAULT_GIT_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate a git repository URL using git ls-remote."""
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--exit-code", "-h", url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True, "git_ls_remote", None
        elif result.returncode == 2:
            return False, "git_ls_remote", "not_found"
        else:
            stderr = result.stderr.lower()
            if "not found" in stderr or "does not exist" in stderr:
                return False, "git_ls_remote", "not_found"
            elif "permission denied" in stderr or "authentication" in stderr:
                return False, "git_ls_remote", "auth_required"
            else:
                return False, "git_ls_remote", "error"

    except subprocess.TimeoutExpired:
        return False, "git_ls_remote", "timeout"
    except FileNotFoundError:
        return False, "git_ls_remote", "git_not_installed"
    except Exception as e:
        logger.debug(f"Git validation error for {url}: {e}")
        return False, "git_ls_remote", "error"


def _extract_package_name(url: str, pattern: re.Pattern) -> Optional[str]:
    """Extract package name from URL using pattern."""
    match = pattern.search(url)
    if match:
        return match.group(1)
    return None


PYPI_PATTERN = re.compile(r"pypi\.org/project/([^/]+)")


def validate_pypi(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate PyPI package exists via API."""
    package = _extract_package_name(url, PYPI_PATTERN)
    if not package:
        return False, "api_check", "invalid_url"

    api_url = f"https://pypi.org/pypi/{package}/json"
    try:
        response = requests.head(api_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return True, "api_check", None
        elif response.status_code == 404:
            return False, "api_check", "not_found"
        else:
            return False, "api_check", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "api_check", "timeout"
    except requests.RequestException as e:
        logger.debug(f"PyPI validation error for {package}: {e}")
        return False, "api_check", "request_error"


NPM_PATTERN = re.compile(r"npmjs\.com/package/([^/]+)")


def validate_npm(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate npm package exists via registry API."""
    package = _extract_package_name(url, NPM_PATTERN)
    if not package:
        return False, "api_check", "invalid_url"

    api_url = f"https://registry.npmjs.org/{package}"
    try:
        response = requests.head(api_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if response.status_code == 200:
            return True, "api_check", None
        elif response.status_code == 404:
            return False, "api_check", "not_found"
        else:
            return False, "api_check", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "api_check", "timeout"
    except requests.RequestException as e:
        logger.debug(f"npm validation error for {package}: {e}")
        return False, "api_check", "request_error"


CRAN_PATTERN = re.compile(r"cran\.r-project\.org/(?:web/)?package[s]?[=/]([^/&]+)")


def validate_cran(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate CRAN package exists."""
    package = _extract_package_name(url, CRAN_PATTERN)
    if not package:
        return False, "api_check", "invalid_url"

    check_url = f"https://cran.r-project.org/package={package}"
    try:
        response = requests.head(
            check_url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "api_check", None
        elif response.status_code == 404:
            return False, "api_check", "not_found"
        else:
            return False, "api_check", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "api_check", "timeout"
    except requests.RequestException as e:
        logger.debug(f"CRAN validation error for {package}: {e}")
        return False, "api_check", "request_error"


BIOCONDUCTOR_PATTERN = re.compile(r"bioconductor\.org/packages/(?:release/|devel/)?(?:bioc|data|workflows)?/?([^/]+)")


def validate_bioconductor(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate Bioconductor package exists."""
    package = _extract_package_name(url, BIOCONDUCTOR_PATTERN)
    if not package:
        return False, "api_check", "invalid_url"

    check_url = f"https://bioconductor.org/packages/{package}"
    try:
        response = requests.head(
            check_url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "api_check", None
        elif response.status_code == 404:
            return False, "api_check", "not_found"
        else:
            return False, "api_check", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "api_check", "timeout"
    except requests.RequestException as e:
        logger.debug(f"Bioconductor validation error for {package}: {e}")
        return False, "api_check", "request_error"


def validate_zenodo(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate Zenodo record exists via HEAD request."""
    try:
        response = requests.head(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "http_head", None
        elif response.status_code == 404:
            return False, "http_head", "not_found"
        elif response.status_code == 410:
            return False, "http_head", "gone"
        else:
            return False, "http_head", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "http_head", "timeout"
    except requests.RequestException as e:
        logger.debug(f"Zenodo validation error for {url}: {e}")
        return False, "http_head", "request_error"


def validate_software_heritage(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate Software Heritage archive exists."""
    try:
        response = requests.head(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "http_head", None
        elif response.status_code == 404:
            return False, "http_head", "not_found"
        else:
            return False, "http_head", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "http_head", "timeout"
    except requests.RequestException as e:
        logger.debug(f"Software Heritage validation error for {url}: {e}")
        return False, "http_head", "request_error"


def validate_codeocean(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate CodeOcean capsule exists."""
    try:
        response = requests.head(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "http_head", None
        elif response.status_code == 404:
            return False, "http_head", "not_found"
        else:
            return False, "http_head", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "http_head", "timeout"
    except requests.RequestException as e:
        logger.debug(f"CodeOcean validation error for {url}: {e}")
        return False, "http_head", "request_error"


def validate_figshare(url: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> Tuple[bool, str, Optional[str]]:
    """Validate Figshare article exists."""
    try:
        response = requests.head(
            url, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        if response.status_code == 200:
            return True, "http_head", None
        elif response.status_code == 404:
            return False, "http_head", "not_found"
        else:
            return False, "http_head", f"http_{response.status_code}"
    except requests.Timeout:
        return False, "http_head", "timeout"
    except requests.RequestException as e:
        logger.debug(f"Figshare validation error for {url}: {e}")
        return False, "http_head", "request_error"


VALIDATORS: Dict[str, Callable[[str, int], Tuple[bool, str, Optional[str]]]] = {
    "github": validate_git_repo,
    "gitlab": validate_git_repo,
    "bitbucket": validate_git_repo,
    "codeberg": validate_git_repo,
    "pypi": validate_pypi,
    "npm": validate_npm,
    "cran": validate_cran,
    "bioconductor": validate_bioconductor,
    "zenodo": validate_zenodo,
    "software_heritage": validate_software_heritage,
    "codeocean": validate_codeocean,
    "figshare": validate_figshare,
}


def can_validate(url_type: str) -> bool:
    """Check if we have a validator for this URL type."""
    return url_type.lower() in VALIDATORS


def validate_url(url: str, url_type: str, timeout: int = DEFAULT_HTTP_TIMEOUT) -> ValidationResult:
    """Validate a single URL using the appropriate method."""
    url_type_lower = url_type.lower()
    validator = VALIDATORS.get(url_type_lower)

    if validator is None:
        return ValidationResult(
            url=url,
            url_type=url_type,
            is_valid=False,
            method="none",
            error="no_validator"
        )

    is_valid, method, error = validator(url, timeout)
    return ValidationResult(
        url=url,
        url_type=url_type,
        is_valid=is_valid,
        method=method,
        error=error
    )


def validate_urls_batch(
    urls: List[Dict[str, str]],
    max_workers: int = 10,
    timeout: int = DEFAULT_HTTP_TIMEOUT,
    progress_callback=None,
) -> List[ValidationResult]:
    """Validate a batch of URLs in parallel."""
    results: List[ValidationResult] = []
    total = len(urls)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(validate_url, item["url"], item["type"], timeout): item
            for item in urls
        }

        for future in as_completed(future_to_url):
            result = future.result()
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

    return results


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


class Validator:
    """Orchestrates validation across multiple validators with checkpointing."""

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
        unique_urls, url_to_records = deduplicate_urls(records)
        logger.info(f"Deduplicated: {len(records)} records -> {len(unique_urls)} unique URLs")

        cached = self.checkpoint.get_cached_urls()
        urls_to_validate = [u for u in unique_urls if u not in cached]
        logger.info(f"After cache filter: {len(urls_to_validate)} URLs to validate")

        self.results = {url: cached[url] for url in unique_urls if url in cached}

        if not urls_to_validate:
            return self.results

        categories = categorize_urls(urls_to_validate)

        if categories["github"]:
            self._validate_github(
                categories["github"],
                lambda c, t: progress_callback("GitHub", c, t) if progress_callback else None,
            )

        git_urls = categories["gitlab"] + categories["bitbucket"] + categories["codeberg"]
        if git_urls:
            self._validate_git_repos(
                git_urls,
                lambda c, t: progress_callback("Git repos", c, t) if progress_callback else None,
            )

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
            for url in urls:
                result = {"url": url, "valid": False, "method": "graphql", "error": "no_token"}
                self.results[url] = result
                self.checkpoint.save_result(url, False, "graphql", "no_token")
            return

        def save_batch(batch_results):
            for r in batch_results:
                result = {"url": r.url, "valid": r.valid, "method": "graphql", "error": r.error}
                self.results[r.url] = result
                self.checkpoint.save_result(r.url, r.valid, "graphql", r.error)

        try:
            asyncio.run(validator.validate_urls(urls, progress_callback, save_batch))

        except RateLimitExceeded as e:
            if self.wait_for_ratelimit:
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
