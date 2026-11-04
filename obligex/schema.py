"""
Obligation extraction schema — the single source of truth.

This module is imported by ALL THREE stages:
  1. dataset building  (teacher labels are validated against it)
  2. fine-tuning       (the system prompt is generated from it)
  3. serving           (vLLM constrained decoding uses its JSON Schema)

Keeping one definition is not tidiness, it is correctness. If the system prompt
used at inference differs by even a field name from the one used in training,
accuracy drops hard and silently. Import `SYSTEM_PROMPT` everywhere; never
retype it.

Python 3.10+, pydantic>=2.6
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from enum import Enum
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Controlled vocabularies
#
# Enums are what make a fine-tune worth doing. Free-text type fields drift
# ("Payment" / "payment" / "Payment Obligation") and are unusable downstream.
# Keep these lists SHORT and MUTUALLY EXCLUSIVE. Every value you add costs
# training examples; 12-16 classes is a practical ceiling for a 4B model.
# ---------------------------------------------------------------------------


class ObligationType(str, Enum):
    PAYMENT = "payment"
    DELIVERY = "delivery_or_performance"
    REPORTING = "reporting_or_notification"
    CONFIDENTIALITY = "confidentiality"
    DATA_PROTECTION = "data_protection_or_privacy"
    COMPLIANCE = "regulatory_compliance"
    INSURANCE = "insurance"
    AUDIT = "audit_or_inspection"
    IP = "intellectual_property"
    WARRANTY = "warranty_or_service_level"
    INDEMNITY = "indemnity"
    TERMINATION = "termination_or_renewal"
    GOVERNANCE = "governance_or_approval"
    RECORD_KEEPING = "record_keeping_or_retention"
    OTHER = "other"


class Modality(str, Enum):
    """The verbatim auxiliary verb that creates the duty.

    This field exists to stop the single most expensive error in obligation
    extraction: recording a RIGHT ("Customer may audit") as a DUTY. Forcing the
    model to commit to the modal verb it saw makes that error visible and
    filterable rather than buried in prose.
    """

    SHALL = "shall"
    MUST = "must"
    WILL = "will"
    IS_REQUIRED_TO = "is_required_to"
    AGREES_TO = "agrees_to"
    MAY = "may"
    SHOULD = "should"
    OTHER = "other"


class BindingStrength(str, Enum):
    MANDATORY = "mandatory"
    CONDITIONAL = "conditional"
    DISCRETIONARY = "discretionary"


class TimingType(str, Enum):
    ONE_TIME_FIXED_DATE = "one_time_fixed_date"
    ONE_TIME_RELATIVE = "one_time_relative"
    RECURRING = "recurring"
    ONGOING = "ongoing"
    ON_EVENT = "on_event"
    UNSPECIFIED = "unspecified"


class Recurrence(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"
    OTHER = "other"


class PartyRole(str, Enum):
    CUSTOMER = "customer_or_buyer"
    SUPPLIER = "supplier_or_vendor"
    LICENSOR = "licensor"
    LICENSEE = "licensee"
    EMPLOYER = "employer"
    EMPLOYEE = "employee"
    DISCLOSING = "disclosing_party"
    RECEIVING = "receiving_party"
    LANDLORD = "landlord"
    TENANT = "tenant"
    LENDER = "lender"
    BORROWER = "borrower"
    OTHER = "other"


class DocumentType(str, Enum):
    MSA = "master_services_agreement"
    SOW = "statement_of_work"
    NDA = "nda"
    LICENSE = "license_agreement"
    EMPLOYMENT = "employment_agreement"
    LEASE = "lease"
    PURCHASE_ORDER = "purchase_order"
    LOAN = "loan_or_credit_agreement"
    DPA = "data_processing_agreement"
    SLA = "service_level_agreement"
    AMENDMENT = "amendment_or_addendum"
    EMAIL = "email_correspondence"
    POLICY = "policy_or_procedure"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------


class SourceSpan(BaseModel):
    """Provenance. Non-optional by design.

    `quote` must be copied verbatim from the input window. That single
    constraint gives you a free, deterministic hallucination detector: if the
    quote is not a substring of the source text, the obligation is fabricated
    and you drop it without a human ever looking at it. See
    `ground_obligations()`.
    """

    model_config = {"extra": "forbid"}

    quote: str = Field(
        ...,
        min_length=10,
        max_length=1200,
        description=(
            "Verbatim text copied EXACTLY from the source window, containing "
            "the words that create the obligation. Never paraphrase. Never "
            "add ellipses. 10-1200 characters."
        ),
    )
    clause_reference: Optional[str] = Field(
        None,
        max_length=64,
        description=(
            "Clause or section number as printed in the document, e.g. "
            "'Section 7.2(b)', 'Art. 4.1', 'Clause 12'. null if unnumbered."
        ),
    )
    page: Optional[int] = Field(
        None, ge=1, le=10000, description="1-based page number, null if unknown."
    )

    # char offsets are filled in by code, not by the model. The model cannot
    # count characters reliably and asking it to try wastes tokens and invites
    # confident wrong numbers.
    char_start: Optional[int] = Field(None, ge=0)
    char_end: Optional[int] = Field(None, ge=0)


class Money(BaseModel):
    model_config = {"extra": "forbid"}

    amount: Optional[float] = Field(
        None, description="Numeric amount with no separators, e.g. 50000.0"
    )
    currency: Optional[str] = Field(
        None,
        max_length=8,
        description="ISO-4217 code where determinable: INR, USD, EUR, GBP.",
    )
    verbatim: Optional[str] = Field(
        None,
        max_length=200,
        description="The amount exactly as written, e.g. 'Rs. 50,00,000/-'.",
    )
    is_cap: bool = Field(
        False,
        description="true if this is a ceiling/limit rather than a sum payable.",
    )


class Timing(BaseModel):
    model_config = {"extra": "forbid"}

    timing_type: TimingType = TimingType.UNSPECIFIED
    due_date: Optional[str] = Field(
        None,
        description="Absolute date as YYYY-MM-DD, only if stated or trivially derivable.",
    )
    relative_days: Optional[int] = Field(
        None,
        ge=0,
        le=36500,
        description="Number of days, for deadlines like 'within 30 days'.",
    )
    relative_to: Optional[str] = Field(
        None,
        max_length=200,
        description="The anchor event, e.g. 'the Effective Date', 'receipt of invoice'.",
    )
    recurrence: Optional[Recurrence] = None
    deadline_verbatim: Optional[str] = Field(
        None,
        max_length=300,
        description="The deadline exactly as written, e.g. 'within thirty (30) days'.",
    )

    @field_validator("due_date")
    @classmethod
    def _iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v in (None, ""):
            return None
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"due_date must be YYYY-MM-DD, got {v!r}")
        return v


class Party(BaseModel):
    model_config = {"extra": "forbid"}

    party_id: str = Field(
        ..., max_length=8, description="Short stable handle: P1, P2, P3."
    )
    name: str = Field(
        ..., min_length=1, max_length=300, description="Legal name as written."
    )
    role: PartyRole = PartyRole.OTHER
    short_name: Optional[str] = Field(
        None,
        max_length=120,
        description="Defined term used for this party, e.g. 'the Supplier'.",
    )


class Obligation(BaseModel):
    model_config = {"extra": "forbid"}

    obligation_id: str = Field(
        "",
        max_length=32,
        description=(
            "Leave as empty string. Assigned deterministically after extraction "
            "so that IDs are stable across re-runs."
        ),
    )
    title: str = Field(
        ...,
        min_length=4,
        max_length=120,
        description=(
            "Imperative one-line summary, no trailing period, e.g. "
            "'Pay invoices within 30 days of receipt'."
        ),
    )
    description: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description=(
            "Self-contained statement of who must do what, by when, under what "
            "condition. Must be readable with no access to the contract. Do not "
            "copy the clause verbatim here — that belongs in source.quote."
        ),
    )
    obligation_type: ObligationType
    obligor: str = Field(
        ...,
        max_length=300,
        description=(
            "The party that owes the duty: a party_id (P1), its defined term, "
            "'both_parties', or 'unspecified'."
        ),
    )
    obligee: Optional[str] = Field(
        None,
        max_length=300,
        description="The party owed the duty, same convention. null if not stated.",
    )
    modality: Modality
    binding_strength: BindingStrength
    trigger: Optional[str] = Field(
        None,
        max_length=400,
        description="Condition that activates the duty, e.g. 'upon receipt of a valid invoice'.",
    )
    timing: Timing = Field(default_factory=Timing)
    financial: Optional[Money] = None
    consequence_of_breach: Optional[str] = Field(
        None, max_length=600, description="Stated penalty, interest, or remedy. null if none."
    )
    source: SourceSpan
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Your calibrated confidence that this is a genuine, correctly-typed "
            "obligation. Use <0.6 when the clause is ambiguous."
        ),
    )

    @field_validator("title")
    @classmethod
    def _clean_title(cls, v: str) -> str:
        return v.strip().rstrip(".")


class WindowExtraction(BaseModel):
    """What the model returns for ONE window of text. This is the training target."""

    model_config = {"extra": "forbid"}

    obligations: list[Obligation] = Field(
        default_factory=list,
        description="All obligations found in this window. Empty list if none.",
    )
    parties: list[Party] = Field(
        default_factory=list,
        description="Parties identifiable from this window only. Empty list if none.",
    )
    document_type: DocumentType = DocumentType.OTHER


class DocumentObligations(BaseModel):
    """Document-level result, assembled in code by merging window extractions.

    The model never produces this object directly — see `merge.py`. This is what
    your API returns to the client.
    """

    model_config = {"extra": "forbid"}

    schema_version: str = SCHEMA_VERSION
    document_id: str
    document_name: Optional[str] = None
    document_type: DocumentType = DocumentType.OTHER
    parties: list[Party] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic IDs
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    """Aggressive normalisation for comparison only, never for storage."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[\s ]+", " ", text)
    return text.strip().lower()


def content_fingerprint(ob: Obligation) -> str:
    """Stable hash of an obligation's identity.

    Deliberately excludes title, description, confidence and timing: those are
    model-generated prose that wobbles between runs. Two extractions of the same
    clause by the same obligor of the same type ARE the same obligation, even if
    the model worded the summary differently. This is what makes cross-window
    deduplication and re-run stability work.
    """
    basis = "|".join(
        [
            _norm(ob.source.quote)[:400],
            _norm(ob.obligor),
            ob.obligation_type.value,
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]


def assign_ids(obligations: Iterable[Obligation], doc_id: str = "") -> list[Obligation]:
    """Assign OBL-#### ids in document order, stashing the fingerprint."""
    out: list[Obligation] = []
    for i, ob in enumerate(obligations, start=1):
        ob = ob.model_copy()
        ob.obligation_id = f"OBL-{i:04d}"
        out.append(ob)
    return out


# ---------------------------------------------------------------------------
# Span grounding — the free hallucination filter
# ---------------------------------------------------------------------------


def find_span(quote: str, source_text: str) -> Optional[tuple[int, int]]:
    """Locate `quote` in `source_text`, tolerating whitespace/punctuation drift.

    Returns (start, end) offsets into the ORIGINAL source_text, or None.

    Three passes, cheapest first. The fuzzy pass matters more than it looks:
    PDF extraction routinely injects line breaks mid-sentence and models
    silently normalise them, so an exact-match-only check would reject a large
    share of perfectly good obligations.
    """
    if not quote or not source_text:
        return None

    # 1. exact
    idx = source_text.find(quote)
    if idx >= 0:
        return (idx, idx + len(quote))

    # 2. whitespace-insensitive, via a map from normalised to original offsets
    orig_idx: list[int] = []
    buf: list[str] = []
    prev_space = True
    for i, ch in enumerate(source_text):
        if ch.isspace() or ch == " ":
            if prev_space:
                continue
            buf.append(" ")
            orig_idx.append(i)
            prev_space = True
        else:
            buf.append(ch)
            orig_idx.append(i)
            prev_space = False
    flat = "".join(buf)
    orig_idx.append(len(source_text))

    q = re.sub(r"[\s ]+", " ", quote).strip()
    j = flat.find(q)
    if j >= 0:
        return (orig_idx[j], orig_idx[min(j + len(q), len(orig_idx) - 1)])

    # 3. case- and punctuation-insensitive
    def squash(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", s).lower())

    sq_src = []
    sq_map = []
    for i, ch in enumerate(source_text):
        c = squash(ch)
        if c:
            sq_src.append(c)
            sq_map.append(i)
    sq_text = "".join(sq_src)
    sq_q = squash(q)
    if sq_q and len(sq_q) >= 12:
        k = sq_text.find(sq_q)
        if k >= 0:
            start = sq_map[k]
            end_i = min(k + len(sq_q) - 1, len(sq_map) - 1)
            return (start, sq_map[end_i] + 1)

    return None


def ground_obligations(
    obligations: Iterable[Obligation],
    source_text: str,
    drop_ungrounded: bool = True,
) -> tuple[list[Obligation], list[Obligation]]:
    """Attach char offsets; split into (grounded, ungrounded).

    Call this on every teacher label before training and on every model output
    before returning it to a client. An ungrounded quote means the model wrote
    text that is not in the document.
    """
    kept: list[Obligation] = []
    rejected: list[Obligation] = []
    for ob in obligations:
        span = find_span(ob.source.quote, source_text)
        if span is None:
            rejected.append(ob)
            if not drop_ungrounded:
                kept.append(ob)
            continue
        ob = ob.model_copy(deep=True)
        ob.source.char_start, ob.source.char_end = span
        kept.append(ob)
    return kept, rejected


# ---------------------------------------------------------------------------
# JSON Schema for constrained decoding
# ---------------------------------------------------------------------------

_STRIP_KEYS = {
    # xgrammar/guidance either ignore or choke on these. Enforce them in
    # pydantic after generation instead of inside the grammar, where a single
    # unsupported keyword can fail schema compilation for the whole request.
    "pattern",
    "format",
    "default",
    "examples",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minItems",
    "maxItems",
    "title",
}


def _inline_refs(node: Any, defs: dict[str, Any], depth: int = 0) -> Any:
    """Resolve $ref/$defs into a single self-contained schema tree."""
    if depth > 24:
        return {"type": "object"}
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            target = defs.get(name, {"type": "object"})
            return _inline_refs(target, defs, depth + 1)
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in ("$defs", "$schema"):
                continue
            if k in _STRIP_KEYS:
                continue
            out[k] = _inline_refs(v, defs, depth + 1)
        return out
    if isinstance(node, list):
        return [_inline_refs(v, defs, depth + 1) for v in node]
    return node


def guided_schema(strip_descriptions: bool = True) -> dict[str, Any]:
    """JSON Schema for `WindowExtraction`, safe for vLLM structured outputs.

    Pass to vLLM as:
        StructuredOutputsParams(json=guided_schema())          # offline
        extra_body={"structured_outputs": {"json": guided_schema()}}   # server

    NOTE: `guided_json` was removed in vLLM v0.12.0. Do not use it.
    """
    raw = WindowExtraction.model_json_schema()
    defs = raw.get("$defs", {})
    schema = _inline_refs(raw, defs)

    if strip_descriptions:
        # Descriptions are instructions for the model and belong in the prompt.
        # Carrying them into the grammar inflates compile time for no benefit.
        def drop_desc(n: Any) -> Any:
            if isinstance(n, dict):
                return {k: drop_desc(v) for k, v in n.items() if k != "description"}
            if isinstance(n, list):
                return [drop_desc(v) for v in n]
            return n

        schema = drop_desc(schema)

    return schema


# ---------------------------------------------------------------------------
# The prompt. Shared verbatim between training and serving.
# ---------------------------------------------------------------------------

_FIELD_DOC = """\
obligations[]           list, may be empty
  title                 imperative one-liner, <=120 chars, no trailing period
  description           self-contained: who must do what, by when, under what condition
  obligation_type       {obligation_types}
  obligor               party that owes the duty: "P1".."P9", its defined term,
                        "both_parties", or "unspecified"
  obligee               party owed the duty, same convention, or null
  modality              {modalities}
  binding_strength      mandatory | conditional | discretionary
  trigger               condition that activates the duty, or null
  timing.timing_type    {timing_types}
  timing.due_date       "YYYY-MM-DD" or null
  timing.relative_days  integer or null        e.g. 30 for "within 30 days"
  timing.relative_to    anchor event or null   e.g. "the Effective Date"
  timing.recurrence     {recurrences} or null
  timing.deadline_verbatim  deadline as written, or null
  financial             {{amount, currency, verbatim, is_cap}} or null
  consequence_of_breach stated penalty/remedy, or null
  source.quote          VERBATIM text from the window that creates the duty
  source.clause_reference  section number as printed, or null
  source.page           1-based page number, or null
  confidence            0.0-1.0
parties[]               list, may be empty
  party_id              "P1", "P2", ...
  name                  legal name as written
  short_name            defined term, e.g. "the Supplier", or null
  role                  {party_roles}
document_type           {document_types}\
"""


def _enum_list(enum_cls: type[Enum]) -> str:
    return " | ".join(m.value for m in enum_cls)


TEACHER_SYSTEM_PROMPT = f"""\
You extract contractual obligations from a window of a document and return JSON.

An OBLIGATION is a duty that a specific party is required or permitted to
perform. Extract one entry per distinct duty.

DO extract:
- duties created by shall / must / will / is required to / agrees to
- duties that are conditional ("if X, Supplier shall Y") — set binding_strength
  to "conditional" and record the condition in trigger
- rights and permissions created by "may" — set modality "may" and
  binding_strength "discretionary"
- payment terms, deadlines, notice periods, reporting and audit duties,
  insurance and compliance requirements, confidentiality and data-protection duties

DO NOT extract:
- definitions, recitals, headings, tables of contents, signature blocks
- statements of fact, background, or governing-law declarations that impose no duty
- boilerplate that creates no action ("This Agreement is governed by the laws of X")
- the same duty twice; if a clause imposes two distinct duties, emit two entries

HARD RULES
1. source.quote MUST be copied character-for-character from the window below.
   Never paraphrase it, never shorten it with "...", never fix its typos or
   line breaks. A quote that is not literally present will be rejected.
2. Put the duty in your own words in `description`. Put the document's words in
   `source.quote`. Do not swap them.
3. If the window contains no obligations, return {{"obligations": [], "parties": [], "document_type": "..."}}.
   Inventing obligations is far worse than returning none.
4. Record the modal verb you actually saw in `modality`. Do not upgrade "may"
   to "shall".
5. Leave obligation_id as "".
6. Output ONE JSON object and nothing else. No markdown fences, no commentary.

FIELDS
{_FIELD_DOC.format(
    obligation_types=_enum_list(ObligationType),
    modalities=_enum_list(Modality),
    timing_types=_enum_list(TimingType),
    recurrences=_enum_list(Recurrence),
    party_roles=_enum_list(PartyRole),
    document_types=_enum_list(DocumentType),
)}\
"""

# ---------------------------------------------------------------------------
# The student prompt.
#
# Why there are two prompts
# -------------------------
# TEACHER_SYSTEM_PROMPT above is ~1050 tokens. It has to be: a frontier model
# labelling your data has never seen this schema and needs every rule and enum
# spelled out.
#
# Your fine-tuned model is in the opposite position. After a few thousand
# examples the schema, the enum vocabularies and the verbatim-quote rule live in
# the weights. Re-sending 1050 tokens of instructions on every call spends 26%
# of a 4096-token budget telling the model something it already knows, and that
# budget is the scarcest thing you have on a T4.
#
# So: label with the long prompt, train and serve with the short one. The short
# prompt is what goes into every training example and every inference request,
# and the two MUST be byte-identical — a fine-tune conditioned on one system
# prompt degrades sharply when served with another. That is why this is a
# constant in shared code and not a string typed into a notebook.
#
# Do not shorten this further after training has started. Changing it
# invalidates your adapter.
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = (
    "Extract contractual obligations from the window as JSON matching the "
    "obligation schema. source.quote must be copied verbatim from the window. "
    "Return {\"obligations\":[],\"parties\":[],\"document_type\":\"...\"} if the "
    "window imposes no duties. Output one JSON object only."
)

# The train/serve contract. Import this, never retype it.
SYSTEM_PROMPT = STUDENT_SYSTEM_PROMPT


def build_user_prompt(
    window_text: str,
    doc_name: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    section_path: str | None = None,
) -> str:
    """Render the user turn. Must be identical in training and serving."""
    head = []
    if doc_name:
        head.append(f"document: {doc_name}")
    if page_start is not None:
        head.append(
            f"pages: {page_start}" + (f"-{page_end}" if page_end and page_end != page_start else "")
        )
    if section_path:
        head.append(f"section: {section_path}")
    meta = ("\n".join(head) + "\n\n") if head else ""
    return f"{meta}<window>\n{window_text}\n</window>\n\nExtract obligations as JSON."


def build_messages(
    window_text: str,
    target: WindowExtraction | None = None,
    system: str = SYSTEM_PROMPT,
    **meta: Any,
) -> list[dict[str, str]]:
    """Build the chat turns.

    `system` defaults to STUDENT_SYSTEM_PROMPT. Pass TEACHER_SYSTEM_PROMPT only
    when calling the labelling model in stage 1.
    """
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(window_text, **meta)},
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": dumps_target(target)})
    return msgs


def dumps_target(extraction: WindowExtraction) -> str:
    """Serialise a training target.

    Compact separators and no indentation: indentation is pure token cost in
    every single training example and every single generation. On a 4B model at
    ~50 obligations/doc this alone is a double-digit percentage of throughput.
    """
    return json.dumps(
        extraction.model_dump(mode="json", exclude_none=False),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def parse_model_output(text: str) -> WindowExtraction:
    """Parse a model completion into a validated WindowExtraction.

    Tolerates the two failure modes that survive even constrained decoding:
    markdown fences, and trailing prose after the object.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start = t.find("{")
        if start < 0:
            raise
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i, ch in enumerate(t[start:], start=start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            raise
        obj = json.loads(t[start:end])
    return WindowExtraction.model_validate(obj)


__all__ = [
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "STUDENT_SYSTEM_PROMPT",
    "TEACHER_SYSTEM_PROMPT",
    "ObligationType",
    "Modality",
    "BindingStrength",
    "TimingType",
    "Recurrence",
    "PartyRole",
    "DocumentType",
    "SourceSpan",
    "Money",
    "Timing",
    "Party",
    "Obligation",
    "WindowExtraction",
    "DocumentObligations",
    "assign_ids",
    "content_fingerprint",
    "find_span",
    "ground_obligations",
    "guided_schema",
    "build_messages",
    "build_user_prompt",
    "dumps_target",
    "parse_model_output",
    "ValidationError",
]
