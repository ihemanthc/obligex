"""
Evaluation. Read this before you train anything.

An obligation extractor cannot be evaluated by loss, and it cannot be evaluated
by "does it return valid JSON" — constrained decoding guarantees that, which
makes schema validity a measure of your decoder, not your model. What you need
is three numbers that move independently:

  detection F1   did it find the right duties, matched by source span?
  strict F1      of those it found, did it get type / obligor / modality right?
  grounding rate what fraction of quotes actually exist in the document?

Grounding rate is the one to watch first. A model that scores well on detection
while quoting text that is not in the contract is unshippable in legal tech, and
this is the single most common way a fine-tuned extractor fails: it learns the
*shape* of a plausible clause and writes one.

Matching uses character-span IoU, not string equality on titles. Titles are
model prose and will never match exactly; spans are facts about the document.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .schema import (
    Obligation,
    WindowExtraction,
    _norm,
    find_span,
    parse_model_output,
)

KEY_FIELDS = ("obligation_type", "obligor", "modality", "binding_strength")


def _span(ob: Obligation, source_text: Optional[str]) -> Optional[tuple[int, int]]:
    if ob.source.char_start is not None and ob.source.char_end is not None:
        return (ob.source.char_start, ob.source.char_end)
    if source_text:
        return find_span(ob.source.quote, source_text)
    return None


def _iou(a: Optional[tuple[int, int]], b: Optional[tuple[int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def _text_overlap(a: str, b: str) -> float:
    """Token-level Jaccard. Fallback when neither side has offsets."""
    ta = set(_norm(a).split())
    tb = set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def match_obligations(
    pred: Sequence[Obligation],
    gold: Sequence[Obligation],
    source_text: Optional[str] = None,
    iou_threshold: float = 0.3,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy one-to-one matching by descending overlap.

    Returns (matches, unmatched_pred_idx, unmatched_gold_idx) where each match is
    (pred_idx, gold_idx, score). The threshold is deliberately loose: 0.3 IoU on
    a clause span means "they are talking about the same sentence", which is the
    right bar for detection. Field correctness is scored separately.
    """
    pairs: list[tuple[float, int, int]] = []
    pspans = [_span(p, source_text) for p in pred]
    gspans = [_span(g, source_text) for g in gold]

    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            s = _iou(pspans[i], gspans[j])
            if s == 0.0:
                s = _text_overlap(p.source.quote, g.source.quote) * 0.99
            if s >= iou_threshold:
                pairs.append((s, i, j))

    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for s, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, s))

    return (
        matches,
        [i for i in range(len(pred)) if i not in used_p],
        [j for j in range(len(gold)) if j not in used_g],
    )


def _field_value(ob: Obligation, name: str) -> str:
    v = getattr(ob, name)
    v = v.value if hasattr(v, "value") else v
    return _norm(str(v))


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvalReport:
    detection: Counts = field(default_factory=Counts)
    strict: Counts = field(default_factory=Counts)
    n_examples: int = 0
    n_parse_failures: int = 0
    n_schema_failures: int = 0
    n_pred_obligations: int = 0
    n_gold_obligations: int = 0
    n_grounded: int = 0
    field_correct: dict[str, int] = field(default_factory=dict)
    field_total: dict[str, int] = field(default_factory=dict)
    empty_gold_examples: int = 0
    empty_gold_false_positives: int = 0
    per_type: dict[str, Counts] = field(default_factory=dict)

    @property
    def grounding_rate(self) -> float:
        return self.n_grounded / self.n_pred_obligations if self.n_pred_obligations else 1.0

    @property
    def parse_rate(self) -> float:
        return 1.0 - (self.n_parse_failures / self.n_examples) if self.n_examples else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "parse_rate": round(self.parse_rate, 4),
            "n_parse_failures": self.n_parse_failures,
            "n_schema_failures": self.n_schema_failures,
            "grounding_rate": round(self.grounding_rate, 4),
            "n_pred_obligations": self.n_pred_obligations,
            "n_gold_obligations": self.n_gold_obligations,
            "detection": self.detection.as_dict(),
            "strict": self.strict.as_dict(),
            "field_accuracy": {
                k: round(self.field_correct.get(k, 0) / self.field_total[k], 4)
                for k in self.field_total
                if self.field_total[k]
            },
            "empty_gold_examples": self.empty_gold_examples,
            "empty_gold_false_positives": self.empty_gold_false_positives,
            "per_type_f1": {
                k: round(v.f1, 4) for k, v in sorted(self.per_type.items())
            },
        }

    def pretty(self) -> str:
        s = self.summary()
        lines = [
            "=" * 62,
            f"{'EVALUATION':^62}",
            "=" * 62,
            f"examples               {s['n_examples']}",
            f"parse rate             {s['parse_rate']:.1%}   ({s['n_parse_failures']} failures)",
            f"schema failures        {s['n_schema_failures']}",
            f"grounding rate         {s['grounding_rate']:.1%}   <-- watch this one",
            "",
            f"predicted obligations  {s['n_pred_obligations']}",
            f"gold obligations       {s['n_gold_obligations']}",
            "",
            "detection (span match)",
            f"  precision            {s['detection']['precision']:.3f}",
            f"  recall               {s['detection']['recall']:.3f}",
            f"  f1                   {s['detection']['f1']:.3f}",
            "",
            "strict (span + all key fields)",
            f"  precision            {s['strict']['precision']:.3f}",
            f"  recall               {s['strict']['recall']:.3f}",
            f"  f1                   {s['strict']['f1']:.3f}",
            "",
            "field accuracy on matched pairs",
        ]
        for k, v in s["field_accuracy"].items():
            lines.append(f"  {k:<20} {v:.3f}")
        lines += [
            "",
            f"empty-gold windows     {s['empty_gold_examples']}",
            f"  with false positives {s['empty_gold_false_positives']}"
            f"   <-- over-extraction on clean text",
            "",
            "per-type detection f1",
        ]
        for k, v in s["per_type_f1"].items():
            lines.append(f"  {k:<34} {v:.3f}")
        lines.append("=" * 62)
        return "\n".join(lines)


def evaluate(
    predictions: Iterable[str | WindowExtraction],
    golds: Iterable[WindowExtraction],
    sources: Optional[Iterable[str]] = None,
    iou_threshold: float = 0.3,
) -> EvalReport:
    """Score raw model completions (or parsed extractions) against gold.

    `sources` is the window text for each example. Supply it — without it the
    grounding rate cannot be computed, and grounding is the metric that tells
    you whether the model is inventing clauses.
    """
    rep = EvalReport()
    src_list = list(sources) if sources is not None else None

    for idx, (pred_raw, gold) in enumerate(zip(predictions, golds)):
        rep.n_examples += 1
        source_text = src_list[idx] if src_list and idx < len(src_list) else None

        if isinstance(pred_raw, WindowExtraction):
            pred = pred_raw
        else:
            try:
                pred = parse_model_output(pred_raw)
            except json.JSONDecodeError:
                rep.n_parse_failures += 1
                rep.n_gold_obligations += len(gold.obligations)
                rep.detection.fn += len(gold.obligations)
                rep.strict.fn += len(gold.obligations)
                continue
            except Exception:
                rep.n_schema_failures += 1
                rep.n_gold_obligations += len(gold.obligations)
                rep.detection.fn += len(gold.obligations)
                rep.strict.fn += len(gold.obligations)
                continue

        rep.n_pred_obligations += len(pred.obligations)
        rep.n_gold_obligations += len(gold.obligations)

        if source_text:
            for ob in pred.obligations:
                if find_span(ob.source.quote, source_text) is not None:
                    rep.n_grounded += 1
        else:
            rep.n_grounded += len(pred.obligations)

        if not gold.obligations:
            rep.empty_gold_examples += 1
            if pred.obligations:
                rep.empty_gold_false_positives += 1

        matches, fp_idx, fn_idx = match_obligations(
            pred.obligations, gold.obligations, source_text, iou_threshold
        )

        rep.detection.tp += len(matches)
        rep.detection.fp += len(fp_idx)
        rep.detection.fn += len(fn_idx)

        for j in fn_idx:
            t = gold.obligations[j].obligation_type.value
            rep.per_type.setdefault(t, Counts()).fn += 1
        for i in fp_idx:
            t = pred.obligations[i].obligation_type.value
            rep.per_type.setdefault(t, Counts()).fp += 1

        for i, j, _ in matches:
            p, g = pred.obligations[i], gold.obligations[j]
            all_ok = True
            for fname in KEY_FIELDS:
                rep.field_total[fname] = rep.field_total.get(fname, 0) + 1
                if _field_value(p, fname) == _field_value(g, fname):
                    rep.field_correct[fname] = rep.field_correct.get(fname, 0) + 1
                else:
                    all_ok = False
            gt = g.obligation_type.value
            if all_ok:
                rep.strict.tp += 1
                rep.per_type.setdefault(gt, Counts()).tp += 1
            else:
                rep.strict.fp += 1
                rep.strict.fn += 1
                rep.per_type.setdefault(gt, Counts()).tp += 1

    return rep


__all__ = ["evaluate", "EvalReport", "Counts", "match_obligations"]
