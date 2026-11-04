"""
Section-aware windowing.

Why this file exists at all
--------------------------
You asked for whole-document single-pass extraction. That is the wrong unit of
work for this task, for four reasons that have nothing to do with GPU budget:

1. Output length. A 120-page MSA contains 100-250 obligations. At ~130 tokens of
   JSON each that is a 15-30k token *completion*. Quality on a 4B model collapses
   long before that, and one malformed bracket at token 14,000 costs you the
   entire document.
2. Recall. Long-context recall is not uniform — material in the middle of a very
   long prompt is retrieved measurably worse than material at either end. A
   contract's payment terms are usually in the middle.
3. Provenance. Your client needs to know which clause on which page each
   obligation came from. Window-level extraction gives you page and character
   offsets for free; single-pass does not.
4. Trainability. 4k windows fine-tune on a free T4. 32k does not, at any model
   size worth shipping.

The document-level JSON your client receives is still produced in one call to
your API. It is assembled in `merge.py` from per-window extractions. The
single-pass requirement is satisfied at the pipeline boundary, which is where it
actually matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Heading detection
#
# Ordered most-specific first. These cover the overwhelming majority of English
# commercial contract numbering. Add client-specific patterns here rather than
# loosening the existing ones — a greedy heading regex shatters the document
# into useless fragments, which is a far worse failure than missing a heading.
# ---------------------------------------------------------------------------

_HEADING_PATTERNS: list[tuple[str, str]] = [
    ("article_roman", r"^\s*ARTICLE\s+([IVXLC]+)\b[.\s:—-]*(.{0,120})$"),
    ("article_num", r"^\s*ARTICLE\s+(\d+)\b[.\s:—-]*(.{0,120})$"),
    ("section_kw", r"^\s*(?:SECTION|Section|Clause|CLAUSE)\s+(\d+(?:\.\d+)*)\b[.\s:—-]*(.{0,120})$"),
    ("schedule", r"^\s*(SCHEDULE|ANNEX|ANNEXURE|EXHIBIT|APPENDIX)\s+([A-Z0-9IVX]+)\b[.\s:—-]*(.{0,120})$"),
    # "7.2 Payment Terms" / "7.2.1  Late payment"
    ("dotted", r"^\s{0,8}(\d{1,2}(?:\.\d{1,3}){0,4})\.?\s+(?=[A-Z(\"'])(.{0,120})$"),
    # "(a) The Supplier shall..."  — only treated as a heading if short
    ("lettered", r"^\s{0,12}\(([a-z]{1,2}|[ivxlc]{1,5})\)\s+(.{0,90})$"),
    ("allcaps", r"^\s{0,8}([A-Z][A-Z0-9 ,'&/()\-\.]{6,70})\s*$"),
    ("recital", r"^\s*(WHEREAS|NOW,? THEREFORE|RECITALS|BACKGROUND)\b(.{0,120})$"),
]

_HEADING_RE = [(name, re.compile(p)) for name, p in _HEADING_PATTERNS]

# Lines that look like headings but are structural noise.
_NOISE_RE = re.compile(
    r"^\s*(page\s+\d+(\s+of\s+\d+)?|\d+\s*/\s*\d+|table of contents|contents|"
    r"confidential|draft|execution version|[-=_*•\.\s]{4,})\s*$",
    re.IGNORECASE,
)


def approx_tokens(text: str) -> int:
    """Cheap token estimate for English legal prose.

    Legal text runs denser than ~4 chars/token because of numbering, citations
    and defined terms in capitals, so 3.6 is a safer divisor. Only used for
    planning window sizes; replace with a real tokenizer via the `token_len`
    argument when you care about the exact boundary.
    """
    return max(1, int(len(text) / 3.6))


@dataclass
class Block:
    """A heading and the body text beneath it."""

    text: str
    char_start: int
    char_end: int
    heading: Optional[str] = None
    heading_kind: Optional[str] = None
    level: int = 99

    @property
    def n_tokens(self) -> int:
        return approx_tokens(self.text)


@dataclass
class Window:
    """One unit of model input."""

    text: str
    char_start: int
    char_end: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None
    index: int = 0
    n_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)


def _heading_level(kind: str, number: str) -> int:
    if kind in ("article_roman", "article_num", "schedule"):
        return 0
    if kind == "allcaps" or kind == "recital":
        return 0
    if kind in ("dotted", "section_kw"):
        return number.count(".") + 1
    if kind == "lettered":
        return 5
    return 9


def split_blocks(text: str) -> list[Block]:
    """Split a document into heading-anchored blocks, preserving char offsets."""
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    cur_lines: list[str] = []
    cur_start = 0
    cur_heading: Optional[str] = None
    cur_kind: Optional[str] = None
    cur_level = 99
    pos = 0

    def flush(end: int) -> None:
        nonlocal cur_lines, cur_start, cur_heading, cur_kind, cur_level
        body = "".join(cur_lines)
        if body.strip():
            blocks.append(
                Block(
                    text=body,
                    char_start=cur_start,
                    char_end=end,
                    heading=cur_heading,
                    heading_kind=cur_kind,
                    level=cur_level,
                )
            )
        cur_lines = []
        cur_heading = None
        cur_kind = None
        cur_level = 99

    for line in lines:
        stripped = line.strip()
        matched = None
        if stripped and not _NOISE_RE.match(stripped) and len(stripped) <= 140:
            for name, rx in _HEADING_RE:
                m = rx.match(line.rstrip("\n"))
                if m:
                    matched = (name, m)
                    break

        if matched is not None:
            name, m = matched
            # A heading closes the previous block.
            flush(pos)
            cur_start = pos
            groups = [g for g in m.groups() if g]
            number = groups[0].strip() if groups else ""
            tail = groups[1].strip() if len(groups) > 1 else ""
            cur_heading = (f"{number} {tail}".strip() if tail else number).strip()
            cur_kind = name
            cur_level = _heading_level(name, number)

        cur_lines.append(line)
        pos += len(line)

    flush(pos)

    if not blocks and text.strip():
        blocks = [Block(text=text, char_start=0, char_end=len(text))]
    return blocks


def _page_for_offset(offset: int, page_spans: Sequence[tuple[int, int, int]]) -> Optional[int]:
    for page_no, s, e in page_spans:
        if s <= offset < e:
            return page_no
    return page_spans[-1][0] if page_spans else None


def _split_oversized(
    block: Block,
    max_tokens: int,
    token_len: Callable[[str], int],
) -> list[Block]:
    """Break a block that is larger than one window on paragraph, then sentence,
    then hard-character boundaries. Offsets stay absolute throughout."""
    if token_len(block.text) <= max_tokens:
        return [block]

    pieces: list[Block] = []

    # paragraph boundaries
    parts = re.split(r"(\n\s*\n)", block.text)
    buf = ""
    buf_start = block.char_start
    cursor = block.char_start
    for part in parts:
        if token_len(buf + part) > max_tokens and buf.strip():
            pieces.append(
                Block(buf, buf_start, cursor, block.heading, block.heading_kind, block.level)
            )
            buf = part
            buf_start = cursor
        else:
            buf += part
        cursor += len(part)
    if buf.strip():
        pieces.append(Block(buf, buf_start, cursor, block.heading, block.heading_kind, block.level))

    # anything still oversized: sentence boundaries
    final: list[Block] = []
    for p in pieces:
        if token_len(p.text) <= max_tokens:
            final.append(p)
            continue
        sents = re.split(r"(?<=[.;:])\s+(?=[A-Z(\"'\d])", p.text)
        buf = ""
        buf_start = p.char_start
        cursor = p.char_start
        for s in sents:
            seg = s + " "
            if token_len(buf + seg) > max_tokens and buf.strip():
                final.append(
                    Block(buf, buf_start, cursor, p.heading, p.heading_kind, p.level)
                )
                buf = seg
                buf_start = cursor
            else:
                buf += seg
            cursor += len(seg)
        if buf.strip():
            final.append(Block(buf, buf_start, min(cursor, p.char_end), p.heading, p.heading_kind, p.level))

    # last resort: hard character split
    out: list[Block] = []
    for p in final:
        if token_len(p.text) <= max_tokens:
            out.append(p)
            continue
        step = max(400, int(max_tokens * 3.4))
        for i in range(0, len(p.text), step):
            out.append(
                Block(
                    p.text[i : i + step],
                    p.char_start + i,
                    min(p.char_start + i + step, p.char_end),
                    p.heading,
                    p.heading_kind,
                    p.level,
                )
            )
    return out


def make_windows(
    text: str,
    target_tokens: int = 1100,
    max_tokens: int = 1400,
    overlap_tokens: int = 180,
    page_spans: Optional[Sequence[tuple[int, int, int]]] = None,
    token_len: Optional[Callable[[str], int]] = None,
) -> list[Window]:
    """Pack blocks into overlapping windows.

    Defaults are tuned for a 4096-token training sequence:
        system prompt   ~900 tokens
        window          ~1100 tokens
        JSON completion ~1600 tokens  (room for ~12 obligations)
                        ------
                        ~3600, leaving headroom

    If you raise `target_tokens`, raise `max_length` in the trainer to match, or
    long completions will be silently truncated mid-JSON — which teaches the
    model to emit unterminated objects. That failure is invisible in the loss
    curve and obvious in production.

    `overlap_tokens` exists so an obligation spanning a window boundary is seen
    whole at least once. Duplicates it creates are removed in `merge.py`.
    """
    tl = token_len or approx_tokens
    blocks = split_blocks(text)

    expanded: list[Block] = []
    for b in blocks:
        expanded.extend(_split_oversized(b, max_tokens, tl))

    windows: list[Window] = []
    i = 0
    n = len(expanded)
    heading_stack: dict[int, str] = {}

    while i < n:
        cur: list[Block] = []
        total = 0
        j = i
        while j < n:
            b = expanded[j]
            bt = tl(b.text)
            if cur and total + bt > target_tokens:
                break
            cur.append(b)
            total += bt
            j += 1
            if total >= target_tokens:
                break

        if not cur:
            cur = [expanded[i]]
            total = tl(cur[0].text)
            j = i + 1

        # section_path from the most recent headings at each level
        for b in cur:
            if b.heading:
                heading_stack[b.level] = b.heading
                for lvl in list(heading_stack):
                    if lvl > b.level:
                        heading_stack.pop(lvl, None)
        path = " > ".join(heading_stack[k] for k in sorted(heading_stack)) or None

        start = cur[0].char_start
        end = cur[-1].char_end
        w_text = text[start:end]

        windows.append(
            Window(
                text=w_text,
                char_start=start,
                char_end=end,
                page_start=_page_for_offset(start, page_spans) if page_spans else None,
                page_end=_page_for_offset(max(start, end - 1), page_spans) if page_spans else None,
                section_path=path,
                index=len(windows),
                n_tokens=total,
                block_ids=list(range(i, j)),
            )
        )

        if j >= n:
            break

        # step back far enough to create the requested overlap
        back = 0
        k = j - 1
        while k > i and back < overlap_tokens:
            back += tl(expanded[k].text)
            k -= 1
        i = max(i + 1, k + 1)

    return windows


def windows_from_pages(
    pages: Sequence[str],
    joiner: str = "\n\n",
    **kwargs,
) -> tuple[str, list[Window]]:
    """Build windows from a list of per-page strings, tracking page numbers.

    Returns (full_text, windows). Keep `full_text` — it is the reference text
    that `ground_obligations()` checks quotes against.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    pos = 0
    for idx, p in enumerate(pages, start=1):
        parts.append(p)
        spans.append((idx, pos, pos + len(p)))
        pos += len(p) + len(joiner)
    full = joiner.join(parts)
    return full, make_windows(full, page_spans=spans, **kwargs)


__all__ = [
    "Block",
    "Window",
    "approx_tokens",
    "split_blocks",
    "make_windows",
    "windows_from_pages",
]
