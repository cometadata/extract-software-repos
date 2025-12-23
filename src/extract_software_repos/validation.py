# src/extract_software_repos/validation.py
"""Validate software URLs by checking if resources actually exist."""

import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests

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
