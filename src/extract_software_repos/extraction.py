# src/extract_software_repos/extraction.py
"""Extract software repository URLs from text."""

import html
import re
from typing import List, Set

PACKAGE_REGISTRY_PATTERNS = {
    "pypi": re.compile(
        r"(?:https?://)?pypi\.org/project/([a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
    "cran": re.compile(
        r"(?:https?://)?cran\.r-project\.org/package=([a-zA-Z0-9_.]+)",
        re.IGNORECASE
    ),
    "npm": re.compile(
        r"(?:https?://)?(?:www\.)?npmjs\.com/package/([a-zA-Z0-9@/._-]+)",
        re.IGNORECASE
    ),
    "conda": re.compile(
        r"(?:https?://)?anaconda\.org/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
        re.IGNORECASE
    ),
    "rubygems": re.compile(
        r"(?:https?://)?rubygems\.org/gems/([a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
    "cargo": re.compile(
        r"(?:https?://)?crates\.io/crates/([a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
    "packagist": re.compile(
        r"(?:https?://)?packagist\.org/packages/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
    "bioconductor": re.compile(
        r"(?:https?://)?bioconductor\.org/packages/([a-zA-Z0-9_.]+)",
        re.IGNORECASE
    ),
}

CODE_REPO_PATTERNS = {
    "github": re.compile(
        r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)(?:/[^\s)\"'<>]*)?",
        re.IGNORECASE
    ),
    "gitlab": re.compile(
        r"(?:https?://)?(?:www\.)?gitlab\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)(?:/[^\s)\"'<>]*)?",
        re.IGNORECASE
    ),
    "bitbucket": re.compile(
        r"(?:https?://)?(?:www\.)?bitbucket\.org/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)(?:/[^\s)\"'<>]*)?",
        re.IGNORECASE
    ),
    "sourceforge": re.compile(
        r"(?:https?://)?(?:www\.)?sourceforge\.net/projects/([a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
    "codeberg": re.compile(
        r"(?:https?://)?codeberg\.org/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)(?:/[^\s)\"'<>]*)?",
        re.IGNORECASE
    ),
}

SOFTWARE_ARCHIVE_PATTERNS = {
    "software_heritage": re.compile(
        r"(?:https?://)?archive\.softwareheritage\.org/([^\s)\"'<>]+)",
        re.IGNORECASE
    ),
    "codeocean": re.compile(
        r"(?:https?://)?codeocean\.com/capsule/([a-zA-Z0-9_-]+)",
        re.IGNORECASE
    ),
}

# Combined pattern for efficient single-pass extraction with Polars
# Uses non-capturing groups for alternation, capturing groups for extraction
COMBINED_URL_PATTERN = r"""
    (?:https?://)?(?:www\.)?
    (?:
        # Code repositories
        (?P<github>github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?)|
        (?P<gitlab>gitlab\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?)|
        (?P<bitbucket>bitbucket\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?)|
        (?P<sourceforge>sourceforge\.net/projects/[a-zA-Z0-9_-]+)|
        (?P<codeberg>codeberg\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:/[^\s)\"'<>]*)?)|
        # Package registries
        (?P<pypi>pypi\.org/project/[a-zA-Z0-9_-]+)|
        (?P<cran>cran\.r-project\.org/package=[a-zA-Z0-9_.]+)|
        (?P<npm>npmjs\.com/package/[a-zA-Z0-9@/._-]+)|
        (?P<conda>anaconda\.org/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)|
        (?P<rubygems>rubygems\.org/gems/[a-zA-Z0-9_-]+)|
        (?P<cargo>crates\.io/crates/[a-zA-Z0-9_-]+)|
        (?P<packagist>packagist\.org/packages/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)|
        (?P<bioconductor>bioconductor\.org/packages/[a-zA-Z0-9_.]+)|
        # Archives
        (?P<software_heritage>archive\.softwareheritage\.org/[^\s)\"'<>]+)|
        (?P<codeocean>codeocean\.com/capsule/[a-zA-Z0-9_-]+)
    )
"""

KNOWN_DOMAINS = [
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "codeberg.org", "pypi.org", "cran.r-project.org", "npmjs.com",
    "anaconda.org", "rubygems.org", "crates.io", "packagist.org",
    "bioconductor.org", "archive.softwareheritage.org", "codeocean.com",
]

EXCLUDED_PATHS = [
    "/wiki", "/issues", "/pull", "/pulls", "/discussions",
    "/projects", "/actions", "/security", "/pulse", "/graphs",
    "/settings", "/stargazers", "/watchers", "/network", "/forks",
]

DATA_EXTENSIONS = [
    ".csv", ".json", ".xml", ".xlsx", ".xls", ".tsv", ".dat",
    ".txt", ".md", ".rst", ".pdf", ".doc", ".docx",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".bmp",
    ".zip", ".tar", ".gz", ".tar.gz", ".rar", ".7z",
]

GITHUB_PAGES_PATTERN = re.compile(r"(?:https?://)?([a-zA-Z0-9_-]+)\.github\.io/", re.IGNORECASE)


def preprocess_text(text: str) -> str:
    """Preprocess text to handle URL variants from PDF extraction.

    Handles:
    - Spaces inserted in domains (github.co m -> github.com)
    - Spaces after slashes (github.com/ user -> github.com/user)
    - HTML entities (&#47; -> /)
    - Hyphenated line breaks (us-\\ner -> user)
    - Soft hyphens and zero-width characters

    Args:
        text: Raw text potentially containing mangled URLs.

    Returns:
        Preprocessed text with normalized URLs.
    """
    text = html.unescape(text)
    text = text.replace("\u00AD", "")  # Soft hyphen
    text = text.replace("\u200B", "")  # Zero-width space
    text = text.replace("\u200C", "")  # Zero-width non-joiner
    text = text.replace("\u200D", "")  # Zero-width joiner
    text = text.replace("\uFEFF", "")  # BOM

    # Fix hyphenated line breaks (word-\nrest -> wordrest)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Collapse spaces around slashes
    text = re.sub(r"\s*/\s*", "/", text)

    # Fix spaces in known domains
    for domain in KNOWN_DOMAINS:
        spaced_pattern = r"\s*".join(list(domain))
        text = re.sub(spaced_pattern, domain, text, flags=re.IGNORECASE)

    return text


def _normalize_path_segment(s: str) -> str:
    """Strip trailing non-alphanumeric chars except hyphen and underscore."""
    while s and s[-1] not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_':
        s = s[:-1]
    return s


def _strip_git_suffix(repo: str) -> str:
    """Remove .git suffix from repository name."""
    if repo.endswith('.git'):
        return repo[:-4]
    return repo


def _is_excluded_path(url_path: str) -> bool:
    """Check if URL path should be excluded."""
    path_lower = url_path.lower()
    path_parts = path_lower.split('/')

    for excluded in EXCLUDED_PATHS:
        excluded_segment = excluded.strip('/')
        if excluded_segment in path_parts:
            return True

    if "/blob/" in path_lower or "/raw/" in path_lower:
        for ext in DATA_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

    return False


def _is_github_pages(url: str) -> bool:
    """Check if URL is a GitHub Pages site."""
    return bool(GITHUB_PAGES_PATTERN.match(url))


def _normalize_package_url(registry: str, full_match: str, package: str) -> str:
    """Normalize a package registry URL."""
    package = _normalize_path_segment(package)
    if not package:
        return ""

    base_urls = {
        "pypi": f"https://pypi.org/project/{package}",
        "cran": f"https://cran.r-project.org/package={package}",
        "npm": f"https://www.npmjs.com/package/{package}",
        "conda": f"https://anaconda.org/{package}",
        "rubygems": f"https://rubygems.org/gems/{package}",
        "cargo": f"https://crates.io/crates/{package}",
        "packagist": f"https://packagist.org/packages/{package}",
        "bioconductor": f"https://bioconductor.org/packages/{package}",
    }
    return base_urls.get(registry, "")


def _normalize_repo_url(platform: str, full_match: str, repo_path: str) -> str:
    """Normalize a code repository URL, filtering excluded paths."""
    if platform == "github" and _is_github_pages(full_match):
        return ""

    if _is_excluded_path(full_match):
        return ""

    repo_path = _normalize_path_segment(repo_path)
    parts = repo_path.split("/")
    if len(parts) < 2:
        return ""

    user = _normalize_path_segment(parts[0])
    repo = _normalize_path_segment(parts[1])
    repo = _strip_git_suffix(repo)

    if not user or not repo:
        return ""
    if repo in ('.', '..'):
        return ""

    user_repo = f"{user}/{repo}"

    base_urls = {
        "github": f"https://github.com/{user_repo}",
        "gitlab": f"https://gitlab.com/{user_repo}",
        "bitbucket": f"https://bitbucket.org/{user_repo}",
        "sourceforge": f"https://sourceforge.net/projects/{user}",
        "codeberg": f"https://codeberg.org/{user_repo}",
    }
    return base_urls.get(platform, "")


def _normalize_archive_url(archive: str, full_match: str) -> str:
    """Normalize a software archive URL."""
    url = full_match.rstrip()
    url = _normalize_path_segment(url)
    if not url:
        return ""
    if not url.startswith("http"):
        url = f"https://{url}"
    return url.replace("http://", "https://")


def extract_base_repo(url: str) -> str:
    """Extract base repository identifier from URL for deduplication.

    Args:
        url: Full URL to extract base from.

    Returns:
        Normalized base repository string.
    """
    url_lower = url.lower()
    url_lower = re.sub(r"^https?://", "", url_lower)
    url_lower = re.sub(r"^www\.", "", url_lower)
    url_lower = url_lower.rstrip("/")

    repo_platforms = ["github.com", "gitlab.com", "bitbucket.org", "codeberg.org"]
    for platform in repo_platforms:
        if url_lower.startswith(platform):
            parts = url_lower.split("/")
            if len(parts) >= 3:
                return f"{parts[0]}/{parts[1]}/{parts[2]}"
            return url_lower

    if url_lower.startswith("sourceforge.net/projects/"):
        parts = url_lower.split("/")
        if len(parts) >= 3:
            return f"{parts[0]}/{parts[1]}/{parts[2]}"

    package_platforms = [
        "pypi.org/project/",
        "cran.r-project.org/package=",
        "npmjs.com/package/",
        "crates.io/crates/",
    ]
    for prefix in package_platforms:
        if prefix in url_lower:
            idx = url_lower.find(prefix)
            base = url_lower[idx:].split("/")
            if len(base) >= 3:
                return "/".join(base[:3])
            return url_lower[idx:]

    return url_lower


def is_duplicate(url: str, existing_identifiers: List[dict]) -> bool:
    """Check if URL is a duplicate of existing relatedIdentifiers.

    Args:
        url: New URL to check.
        existing_identifiers: List of existing relatedIdentifier objects.

    Returns:
        True if URL should be skipped as duplicate.
    """
    new_base = extract_base_repo(url)

    for identifier in existing_identifiers:
        if identifier.get("relatedIdentifierType") != "URL":
            continue
        existing_url = identifier.get("relatedIdentifier", "")
        existing_base = extract_base_repo(existing_url)
        if new_base == existing_base:
            return True

    return False


def extract_software_urls(text: str) -> List[str]:
    """Extract software URLs from text.

    Args:
        text: Text to search for URLs.

    Returns:
        List of normalized, deduplicated software URLs found.
    """
    text = preprocess_text(text)
    urls: Set[str] = set()
    seen_bases: Set[str] = set()

    for name, pattern in PACKAGE_REGISTRY_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_package_url(name, match.group(0), match.group(1))
            if url:
                base = extract_base_repo(url)
                if base not in seen_bases:
                    urls.add(url)
                    seen_bases.add(base)

    for name, pattern in CODE_REPO_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_repo_url(name, match.group(0), match.group(1))
            if url:
                base = extract_base_repo(url)
                if base not in seen_bases:
                    urls.add(url)
                    seen_bases.add(base)

    for name, pattern in SOFTWARE_ARCHIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_archive_url(name, match.group(0))
            if url:
                base = extract_base_repo(url)
                if base not in seen_bases:
                    urls.add(url)
                    seen_bases.add(base)

    return list(urls)


def extract_urls_with_types(text: str) -> List[dict]:
    """Extract software URLs from text with type classification.

    Args:
        text: Full text to search for URLs.

    Returns:
        List of dicts with 'url' and 'type' keys, deduplicated.
    """
    text = preprocess_text(text)
    seen_urls: set = set()
    results: List[dict] = []

    for name, pattern in PACKAGE_REGISTRY_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_package_url(name, match.group(0), match.group(1))
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({"url": url, "type": name})

    for name, pattern in CODE_REPO_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_repo_url(name, match.group(0), match.group(1))
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({"url": url, "type": name})

    for name, pattern in SOFTWARE_ARCHIVE_PATTERNS.items():
        for match in pattern.finditer(text):
            url = _normalize_archive_url(name, match.group(0))
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append({"url": url, "type": name})

    return results
