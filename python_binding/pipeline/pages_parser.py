"""Parse 'pages' query parameter (e.g. '1-5,7,10-12') into 0-based indices."""
from __future__ import annotations


class PageSpecError(ValueError):
    """Raised when the pages parameter is malformed."""


def parse_pages_param(
    pages_str: str | None,
    total_in_doc: int,
    max_pages: int,
) -> list[int]:
    """Parse a 1-based page spec into a list of 0-based indices.

    Format: comma-separated tokens; each token is either "N" or "A-B"
    (1-based inclusive). Out-of-range pages are silently dropped.
    Result is deduplicated (preserving first-occurrence order) and
    truncated to `max_pages` entries.

    Raises:
        PageSpecError: on malformed tokens, descending ranges, or
            non-positive page numbers.
    """
    if pages_str is None or pages_str.strip() == "":
        effective_total = min(total_in_doc, max_pages)
        return list(range(effective_total))

    selected: list[int] = []
    seen: set[int] = set()
    for raw_token in pages_str.split(","):
        token = raw_token.strip()
        if token == "":
            continue
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise PageSpecError(f"Invalid range token: {token!r}")
            a_str, b_str = parts
            if not (a_str.isdigit() and b_str.isdigit()):
                raise PageSpecError(f"Non-numeric range: {token!r}")
            a, b = int(a_str), int(b_str)
            if a < 1 or b < 1:
                raise PageSpecError(f"Page numbers are 1-based: {token!r}")
            if a > b:
                raise PageSpecError(f"descending range: {token!r}")
            for p in range(a, b + 1):
                idx = p - 1
                if 0 <= idx < total_in_doc and idx not in seen:
                    seen.add(idx)
                    selected.append(idx)
        else:
            if not token.isdigit():
                raise PageSpecError(f"Non-numeric page: {token!r}")
            p = int(token)
            if p < 1:
                raise PageSpecError(f"Page numbers are 1-based: {token!r}")
            idx = p - 1
            if 0 <= idx < total_in_doc and idx not in seen:
                seen.add(idx)
                selected.append(idx)

    return selected[:max_pages]
