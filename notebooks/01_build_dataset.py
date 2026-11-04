# %% [markdown]
# # 01 — Build the obligation dataset
#
# You have raw client documents and no labels. This notebook turns them into a
# supervised dataset. **This is the stage that decides whether the whole project
# works.** The fine-tune in notebook 02 is mechanical by comparison — a weekend
# of careful labelling beats a month of hyperparameter search.
#
# Pipeline:
#
# ```
# PDFs / .eml / .msg / .docx
#         |
#         v  text + page offsets
#   section-aware windows (~1100 tokens, overlapping)
#         |
#         v  teacher model, 2 passes
#   candidate obligations
#         |
#         v  span grounding (drops fabrications)
#         v  self-consistency (agreement -> silver, disagreement -> review queue)
#   silver labels + human review queue
#         |
#         v  stratified split
#   train.jsonl / val.jsonl / gold.jsonl
# ```
#
# ## Read this before you run anything
#
# **Your client's documents are confidential.** Sending them to a third-party
# API to generate labels may breach the engagement terms you signed, and
# "it was only for training data" is not a defence. Two options below:
#
# - **Option A — API teacher.** Best labels, fastest. Only if you have written
#   permission, or you are labelling public contracts (CUAD, EDGAR) to
#   bootstrap.
# - **Option B — local open teacher.** Qwen3-32B-AWQ on a rented GPU under your
#   own control. Roughly 3-5 points worse on hard clauses in our experience, and
#   the data never leaves your infrastructure.
#
# Start with CUAD (CC BY 4.0, 510 commercial contracts) to validate the pipeline
# end to end before you touch a single client file. You will find half a dozen
# bugs on public data for free.

# %%
# Mount Drive so parsed text and labels survive a Colab disconnect. Teacher
# labelling costs real money; losing it to a dropped runtime twice is a rite of
# passage you can skip.
from pathlib import Path

try:
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/obligex")
except Exception:
    ROOT = Path("./obligex_work")

RAW = ROOT / "raw"           # put your source documents here
TEXT = ROOT / "text"         # extracted text, one .json per document
LABELS = ROOT / "labels"     # teacher output, one .json per document
DATA = ROOT / "data"         # final jsonl splits
for d in (RAW, TEXT, LABELS, DATA):
    d.mkdir(parents=True, exist_ok=True)

print(f"workspace: {ROOT}")
print(f"put documents in: {RAW}")

# %%
!pip install -q pymupdf extract-msg python-docx anthropic pydantic rapidfuzz

# %%
# Install the shared package. Replace with your own git URL once you have one —
# the whole point of obligex/ is that notebooks 01, 02 and 03 import the SAME
# schema and prompts. Retyping the system prompt into a notebook is how you get
# a model that scores 0.9 in eval and 0.4 in production.
import os
import sys

PKG = ROOT / "obligex"
if not PKG.exists():
    # !git clone https://github.com/<you>/obligex.git {ROOT}/obligex_repo
    # Or: upload obligex/ to Drive and point at it.
    print(f"Place the obligex package at {PKG}")
sys.path.insert(0, str(ROOT))

from obligex import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    DocumentType,
    WindowExtraction,
    build_user_prompt,
    dumps_target,
    ground_obligations,
    guided_schema,
    parse_model_output,
    windows_from_pages,
)

print("schema loaded; teacher prompt is", len(TEACHER_SYSTEM_PROMPT) // 4, "tokens")

# %% [markdown]
# ## 1. Document parsing
#
# Keep page boundaries. They cost nothing to track and your client will ask
# "where did this come from?" about every single obligation.
#
# A note on PDF quality: **a scanned PDF produces zero text and no error.** The
# audit cell below catches that. Do not skip it — silently training on 40 empty
# documents is a failure mode that looks exactly like a learning-rate problem.

# %%
import email
import email.policy
import json
import re
import unicodedata


def clean_text(t: str) -> str:
    """Normalise without destroying the offsets grounding depends on.

    Deliberately conservative: NFKC, de-hyphenate across line breaks, collapse
    runs of blank lines. It does NOT collapse all whitespace, because the window
    text and the grounding reference text must stay byte-identical.
    """
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)        # word- \n wrap
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return t


def pages_from_pdf(path: Path) -> list[str]:
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        return [clean_text(p.get_text("text")) for p in doc]


def pages_from_docx(path: Path) -> list[str]:
    import docx

    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [clean_text("\n".join(parts))]


def pages_from_eml(path: Path) -> list[str]:
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    head = "\n".join(
        f"{k}: {msg.get(k)}" for k in ("From", "To", "Cc", "Date", "Subject") if msg.get(k)
    )
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()
    return [clean_text(f"{head}\n\n{body}")]


def pages_from_msg(path: Path) -> list[str]:
    import extract_msg

    m = extract_msg.Message(str(path))
    head = f"From: {m.sender}\nTo: {m.to}\nDate: {m.date}\nSubject: {m.subject}"
    return [clean_text(f"{head}\n\n{m.body or ''}")]


PARSERS = {
    ".pdf": pages_from_pdf,
    ".docx": pages_from_docx,
    ".eml": pages_from_eml,
    ".msg": pages_from_msg,
    ".txt": lambda p: [clean_text(p.read_text(errors="replace"))],
    ".md": lambda p: [clean_text(p.read_text(errors="replace"))],
}


def parse_document(path: Path) -> dict | None:
    fn = PARSERS.get(path.suffix.lower())
    if fn is None:
        return None
    try:
        pages = fn(path)
    except Exception as e:
        return {"document_id": path.stem, "name": path.name, "error": f"{type(e).__name__}: {e}",
                "pages": [], "n_chars": 0}
    return {
        "document_id": path.stem,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "pages": pages,
        "n_pages": len(pages),
        "n_chars": sum(len(p) for p in pages),
    }


docs = []
skipped = []
for p in sorted(RAW.rglob("*")):
    if not p.is_file():
        continue
    rec = parse_document(p)
    if rec is None:
        skipped.append((p.name, "unsupported extension"))
        continue
    docs.append(rec)
    (TEXT / f"{rec['document_id']}.json").write_text(json.dumps(rec, ensure_ascii=False))

print(f"parsed {len(docs)} documents, skipped {len(skipped)}")

# %%
# PARSING AUDIT — do not proceed past this cell with failures.
import statistics

empty = [d for d in docs if d["n_chars"] < 200]
errored = [d for d in docs if d.get("error")]
tiny_pages = [d for d in docs if d["n_pages"] and d["n_chars"] / d["n_pages"] < 150]

print(f"documents            {len(docs)}")
if docs:
    chars = [d["n_chars"] for d in docs]
    print(f"chars  median {int(statistics.median(chars)):>8,}   max {max(chars):>8,}")
    print(f"total chars          {sum(chars):,}  (~{sum(chars)//3600:,}k tokens)")
print(f"\nPARSE ERRORS         {len(errored)}")
for d in errored[:10]:
    print(f"  {d['name']}: {d['error']}")
print(f"\nEMPTY (<200 chars)   {len(empty)}   <-- almost certainly scanned PDFs")
for d in empty[:10]:
    print(f"  {d['name']}  ({d['n_chars']} chars, {d['n_pages']} pages)")
print(f"\nSPARSE (<150 ch/pg)  {len(tiny_pages)}  <-- partial OCR or image-heavy")
for d in tiny_pages[:5]:
    print(f"  {d['name']}  ({d['n_chars'] // max(1, d['n_pages'])} chars/page)")

if empty or errored:
    print("\n" + "!" * 70)
    print("Scanned documents need OCR before they are usable. Options:")
    print("  ocrmypdf --redo-ocr in.pdf out.pdf        (good, local, free)")
    print("  pytesseract page-by-page                  (slower, more control)")
    print("Training on empty documents teaches the model to return [] always.")
    print("!" * 70)

# %% [markdown]
# ## 2. Windowing
#
# `target_tokens=1100` is matched to a 4096-token training sequence:
#
# | part | tokens |
# |---|---|
# | student system prompt | ~66 |
# | window text | ~1100 |
# | JSON completion (~12 obligations) | ~1600 |
# | metadata + chat template | ~60 |
# | **total** | **~2830**, comfortable headroom |
#
# If you change `target_tokens` here, change `MAX_LEN` in notebook 02 to match.
# A completion truncated mid-JSON trains the model to emit unterminated objects,
# and that bug is invisible in the loss curve.

# %%
TARGET_TOKENS = 1100
MAX_TOKENS = 1400
OVERLAP_TOKENS = 180

all_windows = []
for d in docs:
    if d["n_chars"] < 200:
        continue
    full_text, wins = windows_from_pages(
        d["pages"],
        target_tokens=TARGET_TOKENS,
        max_tokens=MAX_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )
    for w in wins:
        all_windows.append(
            {
                "document_id": d["document_id"],
                "document_name": d["name"],
                "window_index": w.index,
                "text": w.text,
                "char_start": w.char_start,
                "char_end": w.char_end,
                "page_start": w.page_start,
                "page_end": w.page_end,
                "section_path": w.section_path,
                "n_tokens": w.n_tokens,
            }
        )

print(f"{len(all_windows)} windows from {len({w['document_id'] for w in all_windows})} documents")
if all_windows:
    tk = sorted(w["n_tokens"] for w in all_windows)
    print(f"window tokens  p50 {tk[len(tk)//2]}  p95 {tk[int(len(tk)*0.95)]}  max {tk[-1]}")
    print(f"windows/document  mean {len(all_windows)/max(1,len(docs)):.1f}")

# %% [markdown]
# ## 3. Teacher labelling
#
# ### Why two passes
#
# One pass gives you labels. Two passes at `temperature=0.3` give you labels
# *plus a free quality signal*: where the two runs agree on an obligation, the
# label is probably right and goes to `train`. Where they disagree, it goes to a
# human review queue. This routes your scarce review time at the examples that
# actually need it instead of spreading it uniformly.
#
# The cost is 2x on labelling, which is small next to a human hour.

# %%
TEACHER_MODE = "api"          # "api" or "local"
TEACHER_MODEL = "claude-sonnet-4-5"
N_PASSES = 2
TEMPERATURE = 0.3
MAX_WINDOWS = None            # set to e.g. 200 for a cheap pilot run

if TEACHER_MODE == "api":
    from google.colab import userdata  # type: ignore

    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")

# %%
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def label_window_api(window: dict, temperature: float) -> dict | None:
    """One teacher call. Returns a parsed, validated WindowExtraction as dict."""
    import anthropic

    client = anthropic.Anthropic()
    user = build_user_prompt(
        window["text"],
        doc_name=window["document_name"],
        page_start=window["page_start"],
        page_end=window["page_end"],
        section_path=window["section_path"],
    )
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=TEACHER_MODEL,
                max_tokens=8000,
                temperature=temperature,
                system=TEACHER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text
            return parse_model_output(raw).model_dump(mode="json")
        except Exception as e:
            if attempt == 3:
                print(f"  giving up on {window['document_id']}#{window['window_index']}: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def label_window_local(window: dict, temperature: float) -> dict | None:
    """Local vLLM teacher. Constrained decoding means output is always valid.

    Serve the teacher first, on your own hardware:
        vllm serve Qwen/Qwen3-32B-AWQ \\
            --max-model-len 8192 \\
            --structured-outputs-config.backend xgrammar
    """
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
    user = build_user_prompt(
        window["text"],
        doc_name=window["document_name"],
        page_start=window["page_start"],
        page_end=window["page_end"],
        section_path=window["section_path"],
    )
    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen3-32B-AWQ",
            temperature=temperature,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            # vLLM >= 0.12: guided_json was REMOVED. This is the current field.
            extra_body={"structured_outputs": {"json": guided_schema()}},
        )
        return parse_model_output(resp.choices[0].message.content).model_dump(mode="json")
    except Exception as e:
        print(f"  local teacher failed on {window['document_id']}: {e}")
        return None


label_fn = label_window_api if TEACHER_MODE == "api" else label_window_local

targets = all_windows[:MAX_WINDOWS] if MAX_WINDOWS else all_windows
print(f"labelling {len(targets)} windows x {N_PASSES} passes = {len(targets)*N_PASSES} calls")

# %%
# Resumable labelling. Re-running this cell after a disconnect picks up where it
# stopped instead of paying for everything twice.
LABEL_FILE = LABELS / "teacher_passes.jsonl"
done: set[tuple[str, int, int]] = set()
if LABEL_FILE.exists():
    for line in LABEL_FILE.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((r["document_id"], r["window_index"], r["pass"]))
    print(f"resuming: {len(done)} calls already recorded")

jobs = [
    (w, p)
    for w in targets
    for p in range(N_PASSES)
    if (w["document_id"], w["window_index"], p) not in done
]
print(f"{len(jobs)} calls to make")

written = 0
with LABEL_FILE.open("a") as fh:
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(label_fn, w, TEMPERATURE if p > 0 else 0.0): (w, p)
            for w, p in jobs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            w, p = futs[fut]
            result = fut.result()
            if result is None:
                continue
            fh.write(
                json.dumps(
                    {
                        "document_id": w["document_id"],
                        "window_index": w["window_index"],
                        "pass": p,
                        "extraction": result,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            written += 1
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  ({written} written)")

print(f"done: {written} new results in {LABEL_FILE}")

# %% [markdown]
# ## 4. Grounding + self-consistency
#
# Two filters, in this order:
#
# 1. **Grounding.** Any obligation whose `source.quote` is not present in the
#    window is dropped. The teacher wrote text that is not in the document.
#    This is deterministic, costs nothing, and typically removes 2-8% of
#    candidates.
# 2. **Self-consistency.** Obligations found by both passes are high-confidence
#    silver. Found by one pass only, they go to the review queue.

# %%
from collections import Counter, defaultdict

from obligex.merge import _is_duplicate  # same matcher used at merge time
from obligex.schema import Obligation

win_by_key = {(w["document_id"], w["window_index"]): w for w in all_windows}
passes: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
for line in LABEL_FILE.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    passes[(r["document_id"], r["window_index"])][r["pass"]] = r["extraction"]

stats = Counter()
silver: list[dict] = []
review: list[dict] = []

for key, by_pass in passes.items():
    window = win_by_key.get(key)
    if window is None:
        continue

    grounded_passes: list[list[Obligation]] = []
    doc_types: list[str] = []
    parties_seen: list[dict] = []

    for p_idx in sorted(by_pass):
        try:
            ext = WindowExtraction.model_validate(by_pass[p_idx])
        except Exception:
            stats["schema_invalid"] += 1
            continue
        kept, rejected = ground_obligations(ext.obligations, window["text"])
        stats["candidates"] += len(ext.obligations)
        stats["ungrounded_dropped"] += len(rejected)
        grounded_passes.append(kept)
        doc_types.append(ext.document_type.value)
        parties_seen.extend(p.model_dump(mode="json") for p in ext.parties)

    if not grounded_passes:
        continue

    base = grounded_passes[0]
    others = grounded_passes[1:]

    agreed: list[Obligation] = []
    contested: list[Obligation] = []
    for ob in base:
        if not others:
            agreed.append(ob)
            continue
        hits = sum(any(_is_duplicate(ob, o2) for o2 in other) for other in others)
        (agreed if hits == len(others) else contested).append(ob)

    # obligations only the later passes found are contested by definition
    for other in others:
        for ob in other:
            if not any(_is_duplicate(ob, a) for a in agreed) and not any(
                _is_duplicate(ob, c) for c in contested
            ):
                contested.append(ob)

    stats["agreed"] += len(agreed)
    stats["contested"] += len(contested)

    dt = Counter(doc_types).most_common(1)[0][0] if doc_types else DocumentType.OTHER.value
    uniq_parties = {(p["name"], p["role"]): p for p in parties_seen}

    record = {
        **{k: window[k] for k in (
            "document_id", "document_name", "window_index", "text",
            "page_start", "page_end", "section_path")},
        "document_type": dt,
    }
    silver.append({
        **record,
        "obligations": [o.model_dump(mode="json") for o in agreed],
        "parties": list(uniq_parties.values()),
        "n_contested": len(contested),
    })
    if contested:
        review.append({
            **record,
            "agreed": [o.model_dump(mode="json") for o in agreed],
            "contested": [o.model_dump(mode="json") for o in contested],
        })

print(f"candidate obligations     {stats['candidates']}")
print(f"  dropped, ungrounded     {stats['ungrounded_dropped']} "
      f"({stats['ungrounded_dropped']/max(1,stats['candidates']):.1%})  <-- teacher hallucinations")
print(f"  schema-invalid passes   {stats['schema_invalid']}")
print(f"agreed across passes      {stats['agreed']}")
print(f"contested (1 pass only)   {stats['contested']}")
print(f"\nsilver windows            {len(silver)}")
print(f"windows needing review    {len(review)}")

# %%
# REVIEW QUEUE. Open this CSV, fix what is wrong, and you have a gold set.
#
# Review the contested rows first — they are where the teacher was unsure, which
# is where your model will be unsure. Budget 150-250 reviewed windows before you
# train. That is roughly one focused day and it is the difference between a
# number you can quote to a client and a number you cannot.
import csv

review_csv = DATA / "review_queue.csv"
with review_csv.open("w", newline="", encoding="utf-8") as fh:
    wr = csv.writer(fh)
    wr.writerow([
        "document_id", "window_index", "page", "section", "status",
        "title", "type", "obligor", "modality", "quote",
        "VERDICT(keep/drop/fix)", "CORRECTED_TYPE", "CORRECTED_OBLIGOR", "NOTES",
    ])
    for r in review:
        for status, group in (("agreed", r["agreed"]), ("CONTESTED", r["contested"])):
            for o in group:
                wr.writerow([
                    r["document_id"], r["window_index"], r["page_start"],
                    (r["section_path"] or "")[:60], status,
                    o["title"], o["obligation_type"], o["obligor"], o["modality"],
                    o["source"]["quote"][:300].replace("\n", " "),
                    "", "", "", "",
                ])
print(f"wrote {review_csv}")
print("Fill the VERDICT column, then load it back to build gold.jsonl.")

# %% [markdown]
# ## 5. Negative examples
#
# **The single most important cell in this notebook.**
#
# If every training example contains obligations, the model learns that its job
# is to produce obligations — and it will produce them for a table of contents,
# a signature block and a definitions section. Recall looks fine. Precision
# collapses on real documents, where most windows contain no duties at all.
#
# Target: **25-35% of training examples with `obligations: []`.** Keep them
# representative — definitions, recitals, signature pages, schedules of
# addresses, boilerplate governing-law clauses.

# %%
empties = [s for s in silver if not s["obligations"]]
positives = [s for s in silver if s["obligations"]]
share = len(empties) / max(1, len(silver))

print(f"windows with obligations  {len(positives)}")
print(f"windows with none         {len(empties)}  ({share:.1%} of total)")

TARGET_NEG_SHARE = 0.30
if share < 0.20:
    print(f"\n!! Only {share:.1%} negatives. Below 20% the model over-extracts.")
    print("   Label more boilerplate: definitions, recitals, signature pages,")
    print("   notices/addresses schedules, governing law, counterparts clauses.")
elif share > 0.55:
    keep_n = int(len(positives) * TARGET_NEG_SHARE / (1 - TARGET_NEG_SHARE))
    print(f"\n!! {share:.1%} negatives is too many — the model learns to answer [].")
    print(f"   Downsampling empties to {keep_n}.")
else:
    print(f"\nOK: {share:.1%} negatives is in the healthy 20-55% band.")

import random

random.seed(13)
if share > 0.55:
    keep_n = int(len(positives) * TARGET_NEG_SHARE / (1 - TARGET_NEG_SHARE))
    empties = random.sample(empties, min(keep_n, len(empties)))
balanced = positives + empties
random.shuffle(balanced)
print(f"\nbalanced set: {len(balanced)} windows, "
      f"{len(empties)/max(1,len(balanced)):.1%} negative")

# %% [markdown]
# ## 6. Splits
#
# **Split by document, never by window.** Overlapping windows from one contract
# share text; if window 7 is in train and window 8 is in val, you are evaluating
# on text the model has memorised and your val score is a fiction. This mistake
# is easy to make and inflates numbers by 10-20 points.

# %%
doc_ids = sorted({s["document_id"] for s in balanced})
random.seed(7)
random.shuffle(doc_ids)

n = len(doc_ids)
n_gold = max(1, int(n * 0.15))
n_val = max(1, int(n * 0.10))
gold_docs = set(doc_ids[:n_gold])
val_docs = set(doc_ids[n_gold:n_gold + n_val])
train_docs = set(doc_ids[n_gold + n_val:])

assert not (gold_docs & val_docs) and not (gold_docs & train_docs) and not (val_docs & train_docs)


def to_example(s: dict) -> dict:
    """One training row: the window, its metadata, and the JSON target."""
    target = WindowExtraction.model_validate(
        {
            "obligations": s["obligations"],
            "parties": s["parties"],
            "document_type": s["document_type"],
        }
    )
    return {
        "document_id": s["document_id"],
        "document_name": s["document_name"],
        "window_index": s["window_index"],
        "text": s["text"],
        "page_start": s["page_start"],
        "page_end": s["page_end"],
        "section_path": s["section_path"],
        "target_json": dumps_target(target),
        "n_obligations": len(s["obligations"]),
    }


splits = {
    "train": [to_example(s) for s in balanced if s["document_id"] in train_docs],
    "val": [to_example(s) for s in balanced if s["document_id"] in val_docs],
    "gold": [to_example(s) for s in balanced if s["document_id"] in gold_docs],
}

for name, rows in splits.items():
    out = DATA / f"{name}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_docs = len({r["document_id"] for r in rows})
    n_obl = sum(r["n_obligations"] for r in rows)
    n_neg = sum(1 for r in rows if r["n_obligations"] == 0)
    print(f"{name:<6} {len(rows):>5} windows  {n_docs:>3} docs  "
          f"{n_obl:>5} obligations  {n_neg/max(1,len(rows)):>5.1%} negative")

print(f"\nwritten to {DATA}")
print("\ngold.jsonl is still SILVER until you review it. Apply review_queue.csv")
print("verdicts to it before quoting any number to a client.")

# %%
# FINAL AUDIT. Four ways this dataset can be quietly broken.
import collections

train = splits["train"]
print("=" * 64)

# 1. target length vs sequence budget
lens = sorted(len(r["target_json"]) // 3 for r in train)
if lens:
    print(f"target length (approx tokens)  p50 {lens[len(lens)//2]}  "
          f"p95 {lens[int(len(lens)*0.95)]}  max {lens[-1]}")
    over = sum(1 for l in lens if l > 1800)
    print(f"  targets over 1800 tokens: {over}  "
          f"{'<-- raise MAX_LEN in nb 02 or lower TARGET_TOKENS' if over else 'OK'}")

# 2. class balance
types = collections.Counter(
    o["obligation_type"]
    for r in train
    for o in json.loads(r["target_json"])["obligations"]
)
print("\nobligation types in train:")
for t, c in types.most_common():
    print(f"  {t:<34} {c:>5}")
rare = [t for t, c in types.items() if c < 20]
if rare:
    print(f"  !! under 20 examples, will not learn: {rare}")
from obligex import ObligationType as _OT
absent = [t.value for t in _OT if t.value not in types]
if absent:
    print(f"  !! zero examples: {absent}")
    print("     Either label some, or remove them from the enum — an enum value")
    print("     with no training data is a value the model will mispredict into.")

# 3. leakage check
for a, b in (("train", "val"), ("train", "gold"), ("val", "gold")):
    da = {r["document_id"] for r in splits[a]}
    db = {r["document_id"] for r in splits[b]}
    overlap = da & db
    print(f"\n{a}/{b} document overlap: {len(overlap)} "
          f"{'OK' if not overlap else '!! LEAKAGE: ' + str(list(overlap)[:5])}")

# 4. every target must re-validate and every quote must still ground
bad_schema = bad_ground = 0
for r in train:
    try:
        ext = WindowExtraction.model_validate(json.loads(r["target_json"]))
    except Exception:
        bad_schema += 1
        continue
    _, rej = ground_obligations(ext.obligations, r["text"])
    bad_ground += len(rej)
print(f"\ntargets failing schema validation: {bad_schema}  {'OK' if not bad_schema else '!! FIX'}")
print(f"quotes failing to ground:          {bad_ground}  {'OK' if not bad_ground else '!! FIX'}")
print("=" * 64)
print("Proceed to 02_finetune_t4.ipynb" if not (bad_schema or bad_ground)
      else "Fix the above before training.")
