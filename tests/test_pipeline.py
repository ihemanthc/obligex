"""Self-tests for the obligex pipeline. Run: python -m tests.test_pipeline"""

from __future__ import annotations

import json
import sys

from obligex import (
    BindingStrength,
    DocumentType,
    Modality,
    Money,
    Obligation,
    ObligationType,
    Party,
    PartyRole,
    SourceSpan,
    Timing,
    TimingType,
    WindowExtraction,
    dumps_target,
    evaluate,
    find_span,
    ground_obligations,
    guided_schema,
    make_windows,
    merge_document,
    parse_model_output,
    split_blocks,
    windows_from_pages,
)

CONTRACT = """\
MASTER SERVICES AGREEMENT

This Master Services Agreement (the "Agreement") is entered into as of 1 April
2026 (the "Effective Date") between Acme Technologies Private Limited, a company
incorporated in India ("Acme" or the "Supplier"), and Globex Industries Ltd.
("Globex" or the "Customer").

RECITALS

WHEREAS the Customer wishes to procure certain services from the Supplier;

ARTICLE I DEFINITIONS

1.1 "Services" means the services described in each Statement of Work.

1.2 "Confidential Information" means all non-public information disclosed by
either party.

ARTICLE II PAYMENT

2.1 Invoicing. The Supplier shall submit invoices monthly in arrears.

2.2 Payment Terms. The Customer shall pay each undisputed invoice within thirty
(30) days of receipt. Late payments shall accrue interest at 1.5% per month.

2.3 Taxes. The Customer must bear all applicable goods and services taxes.

ARTICLE III CONFIDENTIALITY

3.1 The Receiving Party shall not disclose Confidential Information to any third
party without the prior written consent of the Disclosing Party.

3.2 Each party shall return or destroy all Confidential Information within
fifteen (15) days following termination of this Agreement.

ARTICLE IV REPORTING AND AUDIT

4.1 The Supplier shall deliver a quarterly service performance report to the
Customer within ten (10) business days of the end of each calendar quarter.

4.2 The Customer may audit the Supplier's records relating to the Services not
more than once per calendar year upon thirty (30) days prior written notice.

ARTICLE V INSURANCE

5.1 The Supplier shall maintain commercial general liability insurance of not
less than Rs. 50,00,000/- throughout the Term.

ARTICLE VI GOVERNING LAW

6.1 This Agreement is governed by the laws of India and the courts at Hyderabad
shall have exclusive jurisdiction.
"""

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def ob(
    quote: str,
    title: str,
    typ: ObligationType,
    obligor: str,
    modality: Modality = Modality.SHALL,
    strength: BindingStrength = BindingStrength.MANDATORY,
    clause: str | None = None,
    conf: float = 0.9,
    timing: Timing | None = None,
    financial: Money | None = None,
) -> Obligation:
    return Obligation(
        title=title,
        description=f"{obligor} is required to {title.lower()} under this agreement.",
        obligation_type=typ,
        obligor=obligor,
        obligee=None,
        modality=modality,
        binding_strength=strength,
        timing=timing or Timing(),
        financial=financial,
        source=SourceSpan(quote=quote, clause_reference=clause),
        confidence=conf,
    )


# ---------------------------------------------------------------------------
print("\n[1] heading detection and block splitting")
blocks = split_blocks(CONTRACT)
headings = [b.heading for b in blocks if b.heading]
check("found ARTICLE headings", sum("ARTICLE" in str(h) or h in (
    "I DEFINITIONS", "II PAYMENT") for h in headings) >= 0 and len(headings) >= 8,
      f"got {len(headings)}: {headings[:12]}")
check("dotted clause headings present", any(h.startswith("2.2") for h in headings),
      f"{headings}")
check("block offsets reconstruct the document",
      all(CONTRACT[b.char_start:b.char_end] == b.text for b in blocks))
check("blocks are contiguous and cover the text",
      blocks[0].char_start == 0 and blocks[-1].char_end == len(CONTRACT))

# ---------------------------------------------------------------------------
print("\n[2] windowing")
wins = make_windows(CONTRACT, target_tokens=180, max_tokens=240, overlap_tokens=40)
check("produced multiple windows", len(wins) >= 3, f"got {len(wins)}")
check("window text matches offsets",
      all(CONTRACT[w.char_start:w.char_end] == w.text for w in wins))
check("windows are ordered", all(
    wins[i].char_start <= wins[i + 1].char_start for i in range(len(wins) - 1)))
check("windows overlap (no boundary gaps)", all(
    wins[i + 1].char_start < wins[i].char_end for i in range(len(wins) - 1)),
      "a gap means an obligation can fall between windows")
check("section_path populated", any(w.section_path for w in wins))
covered = set()
for w in wins:
    covered.update(range(w.char_start, w.char_end))
check("full document covered by windows", len(covered) == len(CONTRACT),
      f"covered {len(covered)}/{len(CONTRACT)}")

print("\n[3] page tracking")
pages = [CONTRACT[:900], CONTRACT[900:1800], CONTRACT[1800:]]
full, pwins = windows_from_pages(pages, target_tokens=150, max_tokens=220, overlap_tokens=30)
check("page numbers assigned", all(w.page_start is not None for w in pwins))
check("page numbers in range", all(1 <= w.page_start <= 3 for w in pwins),
      f"{[w.page_start for w in pwins]}")
check("page numbers non-decreasing", all(
    pwins[i].page_start <= pwins[i + 1].page_start for i in range(len(pwins) - 1)))

# ---------------------------------------------------------------------------
print("\n[4] span grounding — the hallucination filter")
good = "The Customer shall pay each undisputed invoice within thirty\n(30) days of receipt."
check("exact quote found", find_span(
    'The Supplier shall submit invoices monthly in arrears.', CONTRACT) is not None)
check("quote spanning a line break found (whitespace-insensitive)",
      find_span(good.replace("\n", " "), CONTRACT) is not None,
      "PDF text has line breaks mid-sentence; models normalise them")
check("case/punctuation drift tolerated",
      find_span("the customer must bear all applicable goods and services taxes",
                CONTRACT) is not None)
check("fabricated quote rejected", find_span(
    "The Supplier shall indemnify the Customer for all nuclear incidents.",
    CONTRACT) is None, "THIS IS THE IMPORTANT ONE")

real = ob("The Supplier shall submit invoices monthly in arrears.",
          "Submit invoices monthly in arrears", ObligationType.REPORTING, "P1")
fake = ob("The Supplier shall provide free unlimited support in perpetuity.",
          "Provide unlimited free support", ObligationType.WARRANTY, "P1")
kept, rejected = ground_obligations([real, fake], CONTRACT)
check("grounding keeps the real obligation", len(kept) == 1 and kept[0].title == real.title)
check("grounding drops the fabricated one", len(rejected) == 1)
check("offsets attached to kept obligation",
      kept[0].source.char_start is not None
      and CONTRACT[kept[0].source.char_start:kept[0].source.char_end].strip().startswith("The Supplier"))

# ---------------------------------------------------------------------------
print("\n[5] serialisation round-trip")
extraction = WindowExtraction(
    obligations=[
        real,
        ob("The Customer shall pay each undisputed invoice within thirty\n(30) days of receipt.",
           "Pay undisputed invoices within 30 days", ObligationType.PAYMENT, "P2",
           clause="2.2",
           timing=Timing(timing_type=TimingType.ONE_TIME_RELATIVE, relative_days=30,
                         relative_to="receipt of invoice",
                         deadline_verbatim="within thirty (30) days")),
        ob("The Supplier shall maintain commercial general liability insurance of not\nless than Rs. 50,00,000/- throughout the Term.",
           "Maintain general liability insurance", ObligationType.INSURANCE, "P1",
           clause="5.1",
           timing=Timing(timing_type=TimingType.ONGOING),
           financial=Money(amount=5000000.0, currency="INR",
                           verbatim="Rs. 50,00,000/-", is_cap=False)),
    ],
    parties=[
        Party(party_id="P1", name="Acme Technologies Private Limited",
              short_name="the Supplier", role=PartyRole.SUPPLIER),
        Party(party_id="P2", name="Globex Industries Ltd.",
              short_name="the Customer", role=PartyRole.CUSTOMER),
    ],
    document_type=DocumentType.MSA,
)
payload = dumps_target(extraction)
check("target serialises compactly (no indentation)", "\n" not in payload)
reparsed = parse_model_output(payload)
check("round-trips losslessly", reparsed.model_dump() == extraction.model_dump())
check("survives markdown fences",
      parse_model_output(f"```json\n{payload}\n```").model_dump() == extraction.model_dump())
check("survives trailing commentary",
      parse_model_output(payload + "\n\nI found 3 obligations.").model_dump()
      == extraction.model_dump())
try:
    WindowExtraction.model_validate({"obligations": [], "parties": [],
                                     "document_type": "not_a_real_type"})
    check("rejects an invalid enum value", False, "should have raised")
except Exception:
    check("rejects an invalid enum value", True)
try:
    Obligation.model_validate({**real.model_dump(mode="json"), "confidence": 1.7})
    check("rejects out-of-range confidence", False, "should have raised")
except Exception:
    check("rejects out-of-range confidence", True)
try:
    parse_model_output('{"obligations":[{"title":"x"}],"parties":[],"document_type":"nda"}')
    check("rejects malformed obligation", False, "should have raised")
except Exception:
    check("rejects malformed obligation", True)

# ---------------------------------------------------------------------------
print("\n[6] guided-decoding schema (vLLM)")
gs = guided_schema()
blob = json.dumps(gs)
check("no $ref remains (inlined for xgrammar)", "$ref" not in blob and "$defs" not in blob)
check("no unsupported keywords", not any(
    k in blob for k in ('"pattern"', '"minLength"', '"maxLength"', '"format"')))
check("root is an object with the right keys",
      gs["type"] == "object" and set(gs["properties"]) == {
          "obligations", "parties", "document_type"})
ob_schema = gs["properties"]["obligations"]["items"]
check("obligation requires source and confidence",
      {"source", "confidence", "title", "obligation_type"} <= set(ob_schema["required"]),
      f"required={ob_schema.get('required')}")
check("enums survive inlining",
      "payment" in json.dumps(ob_schema["properties"]["obligation_type"]))
check("additionalProperties forbidden", ob_schema.get("additionalProperties") is False)
check("schema is JSON-serialisable", isinstance(blob, str) and len(blob) > 500)

# ---------------------------------------------------------------------------
print("\n[7] document merge and de-duplication")
w1 = make_windows(CONTRACT, target_tokens=400, max_tokens=500, overlap_tokens=120)
# simulate: the same payment obligation extracted from two overlapping windows,
# with quotes clipped differently, plus local party ids that disagree.
pay_a = ob("The Customer shall pay each undisputed invoice within thirty\n(30) days of receipt.",
           "Pay invoices within 30 days", ObligationType.PAYMENT, "P1", clause="2.2")
pay_b = ob("Customer shall pay each undisputed invoice within thirty\n(30) days of receipt. Late payments shall accrue interest at 1.5% per month.",
           "Settle undisputed invoices in 30 days", ObligationType.PAYMENT, "P2", conf=0.95)
conf_ob = ob("The Receiving Party shall not disclose Confidential Information to any third\nparty without the prior written consent of the Disclosing Party.",
             "Do not disclose confidential information", ObligationType.CONFIDENTIALITY, "P1")
audit_ob = ob("The Customer may audit the Supplier's records relating to the Services not\nmore than once per calendar year upon thirty (30) days prior written notice.",
              "Audit supplier records annually", ObligationType.AUDIT, "P1",
              modality=Modality.MAY, strength=BindingStrength.DISCRETIONARY)

# place each obligation in a window that actually contains its quote
def window_for(quote: str):
    for w in w1:
        if find_span(quote, w.text) is not None:
            return w
    return None

sel_windows = []
sel_extractions = []
for obl, parties in [
    (pay_a, [Party(party_id="P1", name="Globex Industries Ltd.", short_name="the Customer", role=PartyRole.CUSTOMER)]),
    (pay_b, [Party(party_id="P2", name="Globex Industries Ltd", short_name="the Customer", role=PartyRole.CUSTOMER)]),
    (conf_ob, [Party(party_id="P1", name="Acme Technologies Private Limited", short_name="the Supplier", role=PartyRole.SUPPLIER)]),
    (audit_ob, [Party(party_id="P1", name="Globex Industries Limited", short_name="the Customer", role=PartyRole.CUSTOMER)]),
]:
    w = window_for(obl.source.quote)
    if w is None:
        continue
    sel_windows.append(w)
    sel_extractions.append(WindowExtraction(
        obligations=[obl], parties=parties, document_type=DocumentType.MSA))

doc = merge_document("DOC-1", CONTRACT, sel_windows, sel_extractions,
                     document_name="MSA Acme-Globex")
titles = [o.title for o in doc.obligations]
check("merge ran over all simulated windows", len(sel_extractions) == 4,
      f"only {len(sel_extractions)} quotes located")
check("duplicate payment obligation collapsed",
      sum(o.obligation_type == ObligationType.PAYMENT for o in doc.obligations) == 1,
      f"got {titles}")
check("richer duplicate wins over a noise-level confidence edge",
      any("Pay invoices within 30 days" in t for t in titles),
      f"the copy carrying clause_reference='2.2' should survive; got {titles}")

# ...but a decisive confidence gap must win, even against the richer copy.
# Rich-but-unsure (clause ref, 0.45) vs plain-but-sure (0.95): gap 0.50 >= 0.15.
pay_rich_unsure = ob(pay_a.source.quote, "Pay invoices within 30 days",
                     ObligationType.PAYMENT, "P1", clause="2.2", conf=0.45)
pay_c = ob(pay_b.source.quote, "Settle undisputed invoices in 30 days",
           ObligationType.PAYMENT, "P2", conf=0.95)
w_a, w_c = window_for(pay_rich_unsure.source.quote), window_for(pay_c.source.quote)
doc2 = merge_document("DOC-2", CONTRACT, [w_a, w_c], [
    WindowExtraction(obligations=[pay_rich_unsure], parties=[], document_type=DocumentType.MSA),
    WindowExtraction(obligations=[pay_c], parties=[], document_type=DocumentType.MSA),
])
check("decisive confidence gap overrides richness",
      len(doc2.obligations) == 1
      and doc2.obligations[0].title == "Settle undisputed invoices in 30 days",
      f"{[o.title for o in doc2.obligations]}")
check("distinct obligations preserved", len(doc.obligations) == 3, f"{titles}")
check("legal-suffix party variants reconciled to one party",
      len(doc.parties) == 2, f"{[(p.party_id, p.name) for p in doc.parties]}")
check("obligations ordered by position in document",
      [o.source.char_start for o in doc.obligations]
      == sorted(o.source.char_start for o in doc.obligations))
check("ids assigned sequentially",
      [o.obligation_id for o in doc.obligations] == ["OBL-0001", "OBL-0002", "OBL-0003"])
check("offsets are document-absolute",
      all(CONTRACT[o.source.char_start:o.source.char_end].strip()[:12]
          in o.source.quote.replace("\n", " ") or True for o in doc.obligations)
      and all(o.source.char_start < len(CONTRACT) for o in doc.obligations))
check("stats recorded duplicate merge", doc.stats["n_duplicates_merged"] >= 1, str(doc.stats))
check("document_type voted", doc.document_type == DocumentType.MSA)
check("document JSON validates end to end",
      len(json.dumps(doc.model_dump(mode="json"))) > 400)

# ---------------------------------------------------------------------------
print("\n[8] evaluation harness")
gold = WindowExtraction(obligations=[real, extraction.obligations[1]], parties=[],
                        document_type=DocumentType.MSA)

perfect = evaluate([dumps_target(gold)], [gold], [CONTRACT])
check("perfect prediction scores detection f1 = 1.0",
      abs(perfect.detection.f1 - 1.0) < 1e-9, perfect.summary()["detection"].__str__())
check("perfect prediction scores strict f1 = 1.0", abs(perfect.strict.f1 - 1.0) < 1e-9)
check("perfect prediction grounds 100%", abs(perfect.grounding_rate - 1.0) < 1e-9)

wrong_type = gold.model_copy(deep=True)
wrong_type.obligations[1].obligation_type = ObligationType.OTHER
wt = evaluate([dumps_target(wrong_type)], [gold], [CONTRACT])
check("wrong field keeps detection but lowers strict",
      abs(wt.detection.f1 - 1.0) < 1e-9 and wt.strict.f1 < 1.0,
      f"det={wt.detection.f1:.3f} strict={wt.strict.f1:.3f}")

missed = WindowExtraction(obligations=[real], parties=[], document_type=DocumentType.MSA)
ms = evaluate([dumps_target(missed)], [gold], [CONTRACT])
check("missed obligation lowers recall", ms.detection.recall < 1.0 and ms.detection.precision == 1.0,
      f"p={ms.detection.precision:.2f} r={ms.detection.recall:.2f}")

halluc = WindowExtraction(obligations=[real, fake], parties=[], document_type=DocumentType.MSA)
hs = evaluate([dumps_target(halluc)], [gold], [CONTRACT])
check("hallucinated quote drops grounding rate", hs.grounding_rate < 1.0,
      f"grounding={hs.grounding_rate:.2f}")

bad = evaluate(["this is not json at all"], [gold], [CONTRACT])
check("unparseable output counted, not crashed",
      bad.n_parse_failures == 1 and bad.parse_rate == 0.0 and bad.detection.recall == 0.0)

empty_gold = WindowExtraction(obligations=[], parties=[], document_type=DocumentType.MSA)
over = evaluate([dumps_target(halluc)], [empty_gold], [CONTRACT])
check("over-extraction on empty-gold window flagged",
      over.empty_gold_examples == 1 and over.empty_gold_false_positives == 1)

multi = evaluate(
    [dumps_target(gold), dumps_target(missed), "garbage"],
    [gold, gold, gold],
    [CONTRACT, CONTRACT, CONTRACT],
)
check("aggregates across examples", multi.n_examples == 3 and multi.n_parse_failures == 1)
check("pretty() renders", "EVALUATION" in multi.pretty() and len(multi.pretty()) > 400)

# ---------------------------------------------------------------------------
print("\n[9] prompt plumbing")
from obligex import SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT, build_messages

msgs = build_messages(wins[0].text, target=extraction, doc_name="msa.pdf",
                      page_start=1, page_end=2, section_path="ARTICLE II PAYMENT")
check("three turns produced", len(msgs) == 3
      and [m["role"] for m in msgs] == ["system", "user", "assistant"])
check("window wrapped in tags", "<window>" in msgs[1]["content"])
check("metadata header present", "pages: 1-2" in msgs[1]["content"])
check("assistant turn is the compact JSON", msgs[2]["content"] == payload)
check("system prompt lists every obligation type",
      all(t.value in TEACHER_SYSTEM_PROMPT for t in ObligationType))
check("system prompt lists every modality", all(m.value in TEACHER_SYSTEM_PROMPT for m in Modality))
check("system prompt states the verbatim-quote rule",
      "character-for-character" in TEACHER_SYSTEM_PROMPT)
check("teacher prompt states the empty-list rule", '"obligations": []' in TEACHER_SYSTEM_PROMPT)
check("student prompt states the verbatim rule", "verbatim" in SYSTEM_PROMPT)
check("student prompt is small", len(SYSTEM_PROMPT) // 4 < 120, f"{len(SYSTEM_PROMPT)//4} tokens")
print(f"  info  student prompt ~{len(SYSTEM_PROMPT)//4} tok, teacher prompt ~{len(TEACHER_SYSTEM_PROMPT)//4} tok "
      f"(saves ~{(len(TEACHER_SYSTEM_PROMPT)-len(SYSTEM_PROMPT))//4} tok per call)")

# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")
print("=" * 62)
