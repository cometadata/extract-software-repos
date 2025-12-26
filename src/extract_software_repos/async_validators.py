# src/extract_software_repos/async_validators.py
"""Async HTTP validators for package registries and archives."""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10
USER_AGENT = "Extract-Software-Repos-Validator/2.0"


# URL patterns for extracting package names
PYPI_PATTERN = re.compile(r"pypi\.org/project/([^/]+)")
NPM_PATTERN = re.compile(r"npmjs\.com/package/([^/]+)")
CRAN_PATTERN = re.compile(r"cran\.r-project\.org/(?:web/)?package[s]?[=/]([^/&]+)")
BIOCONDUCTOR_PATTERN = re.compile(
    r"bioconductor\.org/packages/(?:release/|devel/)?(?:bioc|data|workflows)?/?([^/]+)"
)


async def validate_pypi_async(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
    """Validate PyPI package exists."""
    match = PYPI_PATTERN.search(url)
    if not match:
        return False, "invalid_url"

    package = match.group(1)
    api_url = f"https://pypi.org/pypi/{package}/json"

    try:
        async with session.head(
            api_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status == 200:
                return True, None
            elif response.status == 404:
                return False, "not_found"
            else:
                return False, f"http_{response.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientError as e:
        logger.debug(f"PyPI validation error for {package}: {e}")
        return False, "request_error"


async def validate_npm_async(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
    """Validate npm package exists."""
    match = NPM_PATTERN.search(url)
    if not match:
        return False, "invalid_url"

    package = match.group(1)
    api_url = f"https://registry.npmjs.org/{package}"

    try:
        async with session.head(
            api_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status == 200:
                return True, None
            elif response.status == 404:
                return False, "not_found"
            else:
                return False, f"http_{response.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientError as e:
        logger.debug(f"npm validation error for {package}: {e}")
        return False, "request_error"


async def validate_cran_async(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
    """Validate CRAN package exists."""
    match = CRAN_PATTERN.search(url)
    if not match:
        return False, "invalid_url"

    package = match.group(1)
    check_url = f"https://cran.r-project.org/package={package}"

    try:
        async with session.head(
            check_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            if response.status == 200:
                return True, None
            elif response.status == 404:
                return False, "not_found"
            else:
                return False, f"http_{response.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientError as e:
        logger.debug(f"CRAN validation error for {package}: {e}")
        return False, "request_error"


async def validate_bioconductor_async(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
    """Validate Bioconductor package exists."""
    match = BIOCONDUCTOR_PATTERN.search(url)
    if not match:
        return False, "invalid_url"

    package = match.group(1)
    check_url = f"https://bioconductor.org/packages/{package}"

    try:
        async with session.head(
            check_url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            if response.status == 200:
                return True, None
            elif response.status == 404:
                return False, "not_found"
            else:
                return False, f"http_{response.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientError as e:
        logger.debug(f"Bioconductor validation error for {package}: {e}")
        return False, "request_error"


async def validate_http_head_async(
    url: str,
    session: aiohttp.ClientSession,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[bool, Optional[str]]:
    """Generic HTTP HEAD validation."""
    try:
        async with session.head(
            url,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            if response.status == 200:
                return True, None
            elif response.status == 404:
                return False, "not_found"
            elif response.status == 410:
                return False, "gone"
            else:
                return False, f"http_{response.status}"
    except asyncio.TimeoutError:
        return False, "timeout"
    except aiohttp.ClientError as e:
        logger.debug(f"HTTP validation error for {url}: {e}")
        return False, "request_error"


# Validator dispatch table
ASYNC_VALIDATORS = {
    "pypi": validate_pypi_async,
    "npm": validate_npm_async,
    "cran": validate_cran_async,
    "bioconductor": validate_bioconductor_async,
    "zenodo": validate_http_head_async,
    "figshare": validate_http_head_async,
    "codeocean": validate_http_head_async,
    "software_heritage": validate_http_head_async,
}


class AsyncHTTPValidator:
    """Validates URLs using async HTTP requests."""

    def __init__(
        self,
        max_concurrency: int = 100,
        per_host_limit: int = 20,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = 3,
    ):
        self.max_concurrency = max_concurrency
        self.per_host_limit = per_host_limit
        self.timeout = timeout
        self.max_retries = max_retries

    async def _validate_single(
        self,
        url: str,
        url_type: str,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """Validate a single URL with retry logic."""
        validator = ASYNC_VALIDATORS.get(url_type, validate_http_head_async)

        async with semaphore:
            for attempt in range(self.max_retries):
                valid, error = await validator(url, session, self.timeout)

                # Don't retry on definitive results
                if valid or error in ("not_found", "gone", "invalid_url"):
                    break

                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

            return {
                "url": url,
                "valid": valid,
                "method": "api_check" if url_type in ("pypi", "npm") else "http_head",
                "error": error,
            }

    async def validate_urls(
        self,
        urls: List[Dict[str, str]],
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """Validate multiple URLs concurrently.

        Args:
            urls: List of {"url": str, "type": str} dicts
            progress_callback: Optional callback(completed, total)

        Returns:
            List of validation result dicts
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        connector = aiohttp.TCPConnector(limit_per_host=self.per_host_limit)

        results: List[Dict[str, Any]] = []
        total = len(urls)
        completed = 0

        async with aiohttp.ClientSession(
            connector=connector,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            tasks = [
                self._validate_single(item["url"], item["type"], session, semaphore)
                for item in urls
            ]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                completed += 1

                if progress_callback and completed % 100 == 0:
                    progress_callback(completed, total)

        if progress_callback:
            progress_callback(total, total)

        return results
