"""Markdown healing for PDF-extracted text."""

import re
import uuid
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ftfy
import mdformat
from markdown_it import MarkdownIt

# Pattern for normalizing bullets
BULLET_PATTERN = re.compile(r"(?m)^(\s*)([•●◦▪▫·]+)(\s*)")

# Patterns to protect URLs before word recovery
URL_PROTECTION_PATTERNS = [
    # Standard URLs with protocol
    re.compile(r'https?://[^\s<>\[\]()]+'),
    # Known domains without protocol
    re.compile(
        r'(?:github|gitlab|bitbucket|pypi|cran|npmjs|sourceforge|codeberg)'
        r'\.(?:com|org|io|net)[^\s<>\[\]()]*'
    ),
    # Generic domain/path pattern
    re.compile(r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}/[^\s<>\[\]()]+'),
]


def protect_urls(text: str) -> Tuple[str, Dict[str, str]]:
    """Replace URL-like patterns with placeholders.

    Args:
        text: Text potentially containing URLs.

    Returns:
        Tuple of (text with placeholders, mapping of placeholder -> original URL).
    """
    placeholders: Dict[str, str] = {}
    protected = text

    for pattern in URL_PROTECTION_PATTERNS:
        for match in pattern.finditer(protected):
            url = match.group(0)
            if url in placeholders.values():
                continue
            placeholder = f"__URL_PLACEHOLDER_{uuid.uuid4().hex[:8]}__"
            placeholders[placeholder] = url
            protected = protected.replace(url, placeholder, 1)

    return protected, placeholders


def restore_urls(text: str, placeholders: Dict[str, str]) -> str:
    """Restore original URLs from placeholders.

    Args:
        text: Text with placeholders.
        placeholders: Mapping of placeholder -> original URL.

    Returns:
        Text with URLs restored.
    """
    restored = text
    for placeholder, url in placeholders.items():
        restored = restored.replace(placeholder, url)
    return restored


def validate_input_quality(text: str) -> Tuple[bool, str]:
    """Check if text is well-formed enough to process.

    Args:
        text: Text to validate.

    Returns:
        Tuple of (is_valid, reason). Reason explains issue if invalid.
    """
    if not text or not text.strip():
        return False, "Empty text"

    lines = text.splitlines()
    if not lines:
        return False, "No content"

    max_reasonable_line = 1000
    long_lines = [i for i, ln in enumerate(lines) if len(ln) > max_reasonable_line]

    if long_lines and len(long_lines) / len(lines) > 0.20:
        longest = max(len(ln) for ln in lines)
        return False, f"Contains {len(long_lines)} extremely long lines (max: {longest} chars)"

    total_chars = len(text)
    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0

    if whitespace_ratio < 0.05:
        return False, f"Very low whitespace ratio ({whitespace_ratio:.1%}) - likely malformed"

    avg_line_length = total_chars / len(lines) if lines else 0
    if avg_line_length > 1000:
        return False, f"Unusually long average line length ({avg_line_length:.0f} chars)"

    return True, "OK"


def fix_hyphenation(text: str) -> str:
    """Rejoin words split by hyphenation across lines.

    Args:
        text: Text with potential hyphenated line breaks.

    Returns:
        Text with hyphenated words rejoined.
    """
    pattern = re.compile(r"([A-Za-z]{2,})-\s*[\r\n]+\s*([a-z][\w-]*)")
    return pattern.sub(r"\1\2", text)


def normalize_bullets(text: str) -> str:
    """Convert fancy bullet characters to standard markdown dashes.

    Args:
        text: Text with various bullet styles.

    Returns:
        Text with normalized bullet points.
    """
    return BULLET_PATTERN.sub(r"\1- ", text)


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines to maximum of two.

    Args:
        text: Text with potential excessive blank lines.

    Returns:
        Text with collapsed blank lines.
    """
    lines = text.splitlines()
    result: List[str] = []
    blank_run = 0

    for line in lines:
        if line.strip():
            blank_run = 0
            result.append(line.rstrip())
        else:
            blank_run += 1
            if blank_run <= 2:
                result.append("")

    return "\n".join(result)


def strip_repeated_lines(
    text: str, *, page_length_estimate: int, threshold: float, max_words: int
) -> Tuple[str, List[str]]:
    """Remove lines that repeat frequently (likely headers/footers).

    Args:
        text: Text to process.
        page_length_estimate: Approximate lines per page.
        threshold: Frequency threshold for removal.
        max_words: Maximum words in line to consider for removal.

    Returns:
        Tuple of (cleaned text, list of removed lines).
    """
    lines = text.splitlines()

    if len(lines) < page_length_estimate:
        return text, []

    normalized = [ln.strip() for ln in lines]

    freq = Counter(
        ln for ln in normalized
        if ln and len(ln) >= 4 and len(ln.split()) <= max_words
    )

    estimated_pages = max(1, len(lines) / float(page_length_estimate))
    to_remove = {
        ln for ln, count in freq.items()
        if count > 1 and count >= estimated_pages * threshold
    }

    if not to_remove:
        return text, []

    cleaned_lines = [
        original for original, norm in zip(lines, normalized)
        if norm not in to_remove
    ]

    return "\n".join(cleaned_lines), sorted(to_remove)


LIST_MARKER_PATTERN = re.compile(r"^(\d+\.|[A-Za-z]\.)\s")
UNORDERED_MARKER_PATTERN = re.compile(r"^[-*+]\s")


def is_section_header(line: str) -> bool:
    """Check if line appears to be a section header."""
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False

    common_headers = {
        'abstract', 'introduction', 'background', 'related work', 'methodology',
        'method', 'methods', 'approach', 'implementation', 'experiments',
        'results', 'discussion', 'conclusion', 'conclusions', 'references',
        'bibliography', 'acknowledgments', 'acknowledgements', 'appendix',
    }

    lower = stripped.lower()
    if lower in common_headers:
        return True

    if re.match(r'^\d+\.(\d+\.?)?\s+[A-Z]', stripped):
        return True

    if stripped.isupper() and len(stripped) < 50:
        return True

    return False


def is_complete_sentence(line: str) -> bool:
    """Check if line ends with sentence-ending punctuation."""
    stripped = line.strip()
    if not stripped:
        return False

    if stripped[-1] in '.!?':
        if stripped[-1] == '.' and len(stripped) < 10:
            return False
        return True

    return False


def merge_broken_lines(text: str) -> str:
    """Merge lines that were incorrectly broken during PDF extraction.

    Args:
        text: Text with potentially broken lines.

    Returns:
        Text with lines intelligently merged.
    """
    lines = text.splitlines()
    merged: List[str] = []
    buffer = ""
    in_code = False
    blank_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if buffer:
                merged.append(buffer)
                buffer = ""
            for _ in range(min(blank_count, 2)):
                merged.append("")
            blank_count = 0
            merged.append(stripped)
            in_code = not in_code
            continue

        if in_code:
            merged.append(line.rstrip())
            continue

        if not stripped:
            blank_count += 1
            continue

        is_list_item = bool(
            LIST_MARKER_PATTERN.match(stripped) or
            UNORDERED_MARKER_PATTERN.match(stripped)
        )
        starts_new_block = (
            is_list_item or
            stripped.startswith(">") or
            stripped.startswith("#")
        )

        preserve_blank = blank_count >= 2
        if blank_count == 1:
            if starts_new_block or is_section_header(stripped):
                preserve_blank = True
            elif buffer and is_complete_sentence(buffer):
                preserve_blank = True

        if preserve_blank:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append("")

        blank_count = 0

        if starts_new_block:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(stripped)
            continue

        if buffer:
            if buffer.endswith("-"):
                buffer = buffer[:-1] + stripped
            else:
                buffer = f"{buffer} {stripped}"
        else:
            buffer = stripped

    if buffer:
        merged.append(buffer)

    return "\n".join(merged)


def remove_footer_patterns(text: str) -> str:
    """Remove common page number and footer patterns."""
    patterns = [
        r"(?m)^\s*\d+\s*$",
        r"(?m)^\s*(?:Page|Pg\.?)\s*\d+(?:\s*of\s*\d+)?\s*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text


def format_markdown(text: str) -> str:
    """Format text as clean markdown."""
    try:
        return mdformat.text(text)
    except Exception:
        return text


def validate_markdown(text: str) -> List[str]:
    """Validate markdown and return any warnings."""
    warnings: List[str] = []

    if text.count("```") % 2 != 0:
        warnings.append("Unbalanced code fences")

    try:
        MarkdownIt().parse(text)
    except Exception as exc:
        warnings.append(f"Markdown parse issue: {exc}")

    return warnings


def heal_text(content: str) -> Tuple[str, List[str]]:
    """Heal malformed markdown text.

    Applies a series of fixes for common PDF extraction issues:
    - Encoding fixes (via ftfy)
    - Hyphenation repair
    - Line merging
    - Header/footer removal
    - Bullet normalization
    - Markdown formatting

    Args:
        content: Potentially malformed text content.

    Returns:
        Tuple of (healed_content, warnings). Warnings is empty list on clean success.
    """
    warnings: List[str] = []

    if not content or not content.strip():
        return "", ["Empty input"]

    is_valid, reason = validate_input_quality(content)
    if not is_valid:
        warnings.append(f"Input quality: {reason}")

    text, url_placeholders = protect_urls(content)
    text = ftfy.fix_text(text)

    text, removed_lines = strip_repeated_lines(
        text,
        page_length_estimate=50,
        threshold=0.8,
        max_words=12,
    )
    if removed_lines:
        warnings.append(f"Removed repeated lines: {', '.join(removed_lines[:3])}")

    text = remove_footer_patterns(text)
    text = fix_hyphenation(text)
    text = normalize_bullets(text)
    text = merge_broken_lines(text)
    text = collapse_blank_lines(text)
    text = restore_urls(text, url_placeholders)
    text = format_markdown(text)

    validation_warnings = validate_markdown(text)
    warnings.extend(validation_warnings)

    return text, warnings


def _heal_row(args: Tuple[str, str]) -> Tuple[str, str]:
    """Heal a single row. Worker function for multiprocessing."""
    content, content_field = args
    if content:
        healed, warnings = heal_text(content)
        return healed, "; ".join(warnings) if warnings else ""
    return "", ""


def heal_parquet_parallel(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    content_field: str = "content",
    workers: Optional[int] = None,
    chunk_size: int = 1000,
) -> dict:
    """Heal parquet file using parallel processing.

    Args:
        input_path: Path to input parquet file.
        output_path: Path to output parquet file.
        content_field: Column containing text to heal.
        workers: Number of parallel workers (default: CPU count).
        chunk_size: Rows per chunk for parallel processing.

    Returns:
        Stats dict with total, healed, warnings counts.
    """
    import polars as pl

    input_path = Path(input_path)
    output_path = Path(output_path)
    workers = workers or cpu_count()

    # Read full dataframe
    df = pl.read_parquet(input_path)
    total_rows = len(df)

    # Prepare data for parallel processing
    content_list = df[content_field].to_list()
    args_list = [(content, content_field) for content in content_list]

    # Process in parallel
    with Pool(workers) as pool:
        results = pool.map(_heal_row, args_list)

    # Extract results
    healed_content = [r[0] for r in results]
    warnings_list = [r[1] for r in results]

    # Update dataframe with healed content
    result = df.with_columns([
        pl.Series(content_field, healed_content),
        pl.Series("_heal_warnings", warnings_list),
    ])

    # Remove warnings column before writing (keep it internal)
    result_clean = result.drop("_heal_warnings")
    result_clean.write_parquet(output_path)

    # Compute stats
    warning_count = sum(1 for w in warnings_list if w)

    return {
        "total": total_rows,
        "healed": total_rows - warning_count,
        "warnings": warning_count,
    }
