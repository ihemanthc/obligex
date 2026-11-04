"""
Window extractions -> one document-level JSON.

This is the step that makes "whole document in one pass" true from the client's
point of view. It runs in pure Python, costs nothing, and is deterministic —
which means it is testable, unlike anything the model does.

Three jobs:
  1. de-duplicate obligations seen in overlapping windows
  2. reconcile party ids across windows (P1 in window 3 is not P1 in window 9)
  3. produce stable, ordered obligation ids
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional, Sequence

from .chunking import Window
from .schema import (
    DocumentObligations,
    DocumentType,
    Obligation,
    Party,
    WindowExtraction,
    _norm,
    content_fingerprint,
    ground_obligations,
)


def _span_iou(a: Obligation, b: Obligation) -> float:
    """Character-level intersection-over-union of two source spans."""
    if None in (a.source.char_start, a.source.char_end, b.source.char_start, b.source.char_end):
        return 0.0
    a0, a1 = a.source.char_start, a.source.char_end
    b0, b1 = b.source.char_start, b.source.char_end
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union else 0.0


def _is_duplicate(a: Obligation, b: Obligation, iou_threshold: float = 0.5) -> bool:
    """Two extractions describe the same duty.

    Fingerprint equality catches the clean case. The IoU path catches the messy
    one: overlapping windows frequently yield quotes that start or end a few
    words apart, so the hashes differ while the obligation is plainly the same.
    Requiring type agreement on the IoU path stops a payment duty and an audit
    duty in one dense clause from being collapsed into one.
    """
    if content_fingerprint(a) == content_fingerprint(b):
        return True
    if a.obligation_type != b.obligation_type:
        return False
    if _span_iou(a, b) >= iou_threshold:
        return True
    # identical quote text after normalisation, offsets unknown
    if a.source.char_start is None or b.source.char_start is None:
        if _norm(a.source.quote) == _norm(b.source.quote) and _norm(a.obligor) == _norm(b.obligor):
            return True
    return False


def _pick_better(a: Obligation, b: Obligation) -> Obligation:
    """Keep the richer of two duplicate extractions.

    Preference order: *decisively* higher confidence, then more non-null fields,
    then the longer quote. The longer quote usually came from the window where
    the clause sat whole rather than clipped at an edge.

    The 0.15 margin is deliberate. Self-reported confidence from a 4B model is
    poorly calibrated, and anything under ~0.15 of separation is noise. A clause
    reference or a populated deadline is verifiable metadata and should not be
    discarded because the other copy came back 0.03 more confident. Only a real
    gap in confidence overrides field richness.
    """
    CONFIDENCE_MARGIN = 0.15
    if abs(a.confidence - b.confidence) >= CONFIDENCE_MARGIN:
        return a if a.confidence > b.confidence else b

    def richness(o: Obligation) -> int:
        d = o.model_dump(mode="json")
        score = 0
        for key in ("trigger", "obligee", "consequence_of_breach", "financial"):
            if d.get(key) not in (None, "", {}):
                score += 1
        t = d.get("timing") or {}
        for key in ("due_date", "relative_days", "relative_to", "recurrence", "deadline_verbatim"):
            if t.get(key) not in (None, ""):
                score += 1
        if d.get("source", {}).get("clause_reference"):
            score += 2
        return score

    ra, rb = richness(a), richness(b)
    if ra != rb:
        return a if ra > rb else b
    return a if len(a.source.quote) >= len(b.source.quote) else b


# ---------------------------------------------------------------------------
# Party reconciliation
# ---------------------------------------------------------------------------

_LEGAL_SUFFIX = re.compile(
    r"\b(private limited|pvt\.?\s*ltd\.?|limited|ltd\.?|llp|llc|inc\.?|incorporated|"
    r"corp\.?|corporation|gmbh|s\.?a\.?|b\.?v\.?|plc|co\.?)\b\.?",
    re.IGNORECASE,
)


def _party_key(p: Party) -> str:
    name = _LEGAL_SUFFIX.sub("", p.name)
    name = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def reconcile_parties(
    window_parties: Sequence[Sequence[Party]],
) -> tuple[list[Party], list[dict[str, str]]]:
    """Merge party lists across windows.

    Returns the canonical party list and, per window, a map from that window's
    local party_id to the canonical one. Without this remapping, obligor values
    are meaningless at document level: each window numbers parties from P1
    independently.
    """
    canonical: list[Party] = []
    by_key: dict[str, str] = {}
    role_votes: dict[str, Counter] = {}
    alias_votes: dict[str, Counter] = {}
    maps: list[dict[str, str]] = []

    for parties in window_parties:
        local: dict[str, str] = {}
        for p in parties:
            key = _party_key(p)
            if not key:
                continue
            if key not in by_key:
                pid = f"P{len(canonical) + 1}"
                by_key[key] = pid
                canonical.append(Party(party_id=pid, name=p.name.strip(), role=p.role, short_name=p.short_name))
                role_votes[pid] = Counter()
                alias_votes[pid] = Counter()
            pid = by_key[key]
            local[p.party_id] = pid
            role_votes[pid][p.role.value] += 1
            if p.short_name:
                alias_votes[pid][p.short_name.strip()] += 1
        maps.append(local)

    # majority vote on role and defined term
    for p in canonical:
        if role_votes[p.party_id]:
            p.role = type(p.role)(role_votes[p.party_id].most_common(1)[0][0])
        if alias_votes[p.party_id]:
            p.short_name = alias_votes[p.party_id].most_common(1)[0][0]

    return canonical, maps


def _remap_obligor(value: Optional[str], pmap: dict[str, str]) -> Optional[str]:
    if not value:
        return value
    v = value.strip()
    if v in pmap:
        return pmap[v]
    return v


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def merge_document(
    document_id: str,
    full_text: str,
    windows: Sequence[Window],
    extractions: Sequence[WindowExtraction],
    document_name: Optional[str] = None,
    iou_threshold: float = 0.5,
    drop_ungrounded: bool = True,
    min_confidence: float = 0.0,
) -> DocumentObligations:
    """Assemble the client-facing document JSON.

    `windows` and `extractions` must be the same length and in the same order.
    `full_text` must be the exact text the windows were cut from — quotes are
    re-grounded against it so that char offsets are document-absolute rather
    than window-relative.
    """
    if len(windows) != len(extractions):
        raise ValueError(
            f"windows ({len(windows)}) and extractions ({len(extractions)}) must align"
        )

    canonical_parties, pmaps = reconcile_parties([e.parties for e in extractions])

    collected: list[Obligation] = []
    n_ungrounded = 0
    n_low_conf = 0

    for w, extraction, pmap in zip(windows, extractions, pmaps):
        grounded, rejected = ground_obligations(
            extraction.obligations, w.text, drop_ungrounded=drop_ungrounded
        )
        n_ungrounded += len(rejected)

        for ob in grounded:
            if ob.confidence < min_confidence:
                n_low_conf += 1
                continue
            ob = ob.model_copy(deep=True)
            # window-relative -> document-absolute
            if ob.source.char_start is not None:
                ob.source.char_start += w.char_start
                ob.source.char_end = (ob.source.char_end or 0) + w.char_start
            if ob.source.page is None:
                ob.source.page = w.page_start
            ob.obligor = _remap_obligor(ob.obligor, pmap) or "unspecified"
            ob.obligee = _remap_obligor(ob.obligee, pmap)
            collected.append(ob)

    # de-duplicate
    deduped: list[Obligation] = []
    n_dupes = 0
    for ob in collected:
        hit = None
        for idx, kept in enumerate(deduped):
            if _is_duplicate(ob, kept, iou_threshold):
                hit = idx
                break
        if hit is None:
            deduped.append(ob)
        else:
            deduped[hit] = _pick_better(deduped[hit], ob)
            n_dupes += 1

    deduped.sort(
        key=lambda o: (
            o.source.char_start if o.source.char_start is not None else 1 << 60,
            o.source.quote[:40],
        )
    )
    for i, ob in enumerate(deduped, start=1):
        ob.obligation_id = f"OBL-{i:04d}"

    dt_votes = Counter(e.document_type.value for e in extractions if e.document_type)
    doc_type = DocumentType(dt_votes.most_common(1)[0][0]) if dt_votes else DocumentType.OTHER

    by_type = Counter(o.obligation_type.value for o in deduped)
    by_obligor = Counter(o.obligor for o in deduped)

    return DocumentObligations(
        document_id=document_id,
        document_name=document_name,
        document_type=doc_type,
        parties=canonical_parties,
        obligations=deduped,
        stats={
            "n_windows": len(windows),
            "n_obligations": len(deduped),
            "n_raw_extractions": len(collected) + n_ungrounded,
            "n_duplicates_merged": n_dupes,
            "n_ungrounded_dropped": n_ungrounded,
            "n_low_confidence_dropped": n_low_conf,
            "n_needs_review": sum(1 for o in deduped if o.confidence < 0.6),
            "by_type": dict(by_type),
            "by_obligor": dict(by_obligor),
            "doc_chars": len(full_text),
        },
    )


__all__ = ["merge_document", "reconcile_parties"]
