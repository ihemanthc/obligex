# %% [markdown]
# # 03 — Evaluate, then serve on vLLM
#
# Two halves. The first produces numbers you can defend to a client. The second
# is the production path.
#
# ## What to measure, and why loss is not it
#
# Training loss tells you the model learned *a* distribution. It cannot tell you
# whether the obligations are real. Three metrics that move independently:
#
# | metric | question | failure it catches |
# |---|---|---|
# | **grounding rate** | do the quotes exist in the document? | invented clauses |
# | **detection F1** | did it find the right duties? | missed / spurious obligations |
# | **strict F1** | are type, obligor, modality right? | plausible but mislabelled |
#
# **Look at grounding rate first.** A model that scores 0.85 detection while
# quoting text that is not in the contract is unshippable in legal tech, and
# this is the characteristic failure of a fine-tuned extractor: it learns the
# *shape* of a plausible clause and writes one. Detection F1 cannot see it,
# because a fabricated quote that happens to overlap a real span still matches.
#
# Note that schema validity is **not** on this list. With constrained decoding
# it is 100% by construction — it measures your decoder, not your model. Which
# is why the eval below runs **unconstrained**.

# %%
!pip install -q unsloth
import unsloth  # noqa: F401  <- first, always
from unsloth import FastLanguageModel

import json
import os
import sys
from pathlib import Path

import torch

try:
    from google.colab import drive, userdata  # type: ignore

    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/obligex")
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    ROOT = Path("./obligex_work")

DATA = ROOT / "data"
OUT = ROOT / "runs"

# %%
# Install / locate the shared obligex package.
#
# All three notebooks import the SAME schema and prompts. Retyping the system
# prompt into a notebook is how you get a model that scores 0.9 in eval and 0.4
# in production, so this cell exists to make the import reliable rather than
# convenient.
#
# Code lives on local Colab disk, NOT Drive: imports off a Drive FUSE mount are
# slow, __pycache__ writes there are flaky, and nesting the clone inside the
# Drive workspace is what produces the "unknown location" namespace-package
# error. Drive holds data; /content holds code.
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ihemanthc/obligex.git"
CLONE_DIR = Path("/content/obligex_repo")      # local disk, distinct name

if CLONE_DIR.exists():
    subprocess.run(["git", "-C", str(CLONE_DIR), "pull", "--quiet"], check=False)
    print(f"repo present at {CLONE_DIR} (pulled)")
else:
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", REPO_URL, str(CLONE_DIR)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(
            f"clone failed: {r.stderr.strip()}\n"
            f"If the repo is private, use a token URL or upload obligex/ to Drive "
            f"and set SEARCH_ROOTS below to its parent."
        )
    print(f"cloned to {CLONE_DIR}")

# Find the directory CONTAINING the package (the dir whose child is
# obligex/__init__.py). Do not assume the clone root is it — a repo cloned into
# a directory also called 'obligex' nests one level deeper, and putting the
# wrong level on sys.path yields a namespace package that imports with
# __file__ = None and raises "cannot import name ... (unknown location)".
SEARCH_ROOTS = [CLONE_DIR, Path("/content"), Path("/content/drive/MyDrive")]

pkg_parent = None
for root in SEARCH_ROOTS:
    if not root.exists():
        continue
    for init in sorted(root.glob("**/obligex/__init__.py")):
        if "__pycache__" in str(init) or ".git" in str(init).split("/"):
            continue
        pkg_parent = init.parent.parent
        break
    if pkg_parent:
        break

if pkg_parent is None:
    raise SystemExit(
        "Could not find obligex/__init__.py under: "
        + ", ".join(str(r) for r in SEARCH_ROOTS)
    )

print(f"package dir   {pkg_parent / 'obligex'}")
print(f"sys.path entry {pkg_parent}")

# Drop any stale/namespace 'obligex' already bound in this kernel. Without this
# a failed import stays cached and keeps failing after you fix the path, which
# looks exactly like the fix not working.
for name in [m for m in sys.modules if m == "obligex" or m.startswith("obligex.")]:
    del sys.modules[name]

sys.path.insert(0, str(pkg_parent))

from obligex import (  # noqa: E402
    SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    DocumentType,
    ObligationType,
    WindowExtraction,
    build_user_prompt,
    dumps_target,
    ground_obligations,
    guided_schema,
    parse_model_output,
    windows_from_pages,
)
import obligex  # noqa: E402

# A real package has a __file__; a namespace package does not. Assert it, so the
# failure is one clear line instead of a confusing ImportError further down.
assert obligex.__file__ is not None, (
    f"'obligex' resolved to a NAMESPACE package at {list(obligex.__path__)} — "
    "that directory has no __init__.py. sys.path is pointing one level too high; "
    "it must point at the PARENT of the package directory."
)

print(f"\nobligex {obligex.__version__} loaded from {obligex.__file__}")
print(f"  student prompt {len(SYSTEM_PROMPT) // 4:>5} tokens  (train + serve)")
print(f"  teacher prompt {len(TEACHER_SYSTEM_PROMPT) // 4:>5} tokens  (labelling only)")

from obligex import evaluate  # noqa: E402

# %%
HF_REPO = "your-username/obligex-qwen3-4b"
ADAPTER = str(OUT / "adapter")          # or HF_REPO + "-adapter"
MAX_LEN = 4096

# %%
def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


gold_rows = load_jsonl(DATA / "gold.jsonl")
print(f"gold: {len(gold_rows)} windows from "
      f"{len({r['document_id'] for r in gold_rows})} documents")
print(f"  with obligations: {sum(1 for r in gold_rows if r['n_obligations'] > 0)}")
print(f"  without:          {sum(1 for r in gold_rows if r['n_obligations'] == 0)}")
print(f"  total obligations: {sum(r['n_obligations'] for r in gold_rows)}")

if len(gold_rows) < 60:
    print("\n!! A gold set under ~60 windows gives F1 with a confidence interval")
    print("   wide enough to hide a 10-point regression. Label more before you")
    print("   make decisions from these numbers.")

# %% [markdown]
# ## Generate predictions
#
# Greedy (`do_sample=False`). Extraction is not a creative task and sampling
# only adds variance you will mistake for model quality.

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)


def predict(rows: list[dict], max_new_tokens: int = 2048) -> list[str]:
    preds = []
    for i, row in enumerate(rows, 1):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    row["text"],
                    doc_name=row.get("document_name"),
                    page_start=row.get("page_start"),
                    section_path=row.get("section_path"),
                ),
            },
        ]
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        preds.append(tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}")
    return preds


import time  # noqa: E402

t0 = time.time()
predictions = predict(gold_rows)
print(f"\n{len(predictions)} predictions in {(time.time()-t0)/60:.1f} min")
(OUT / "gold_predictions.json").write_text(json.dumps(predictions))

# %%
golds = [WindowExtraction.model_validate(json.loads(r["target_json"])) for r in gold_rows]
sources = [r["text"] for r in gold_rows]

report = evaluate(predictions, golds, sources)
print(report.pretty())

(OUT / "eval_report.json").write_text(json.dumps(report.summary(), indent=2))

# %%
# How to read what you just got.
s = report.summary()
print("=" * 66)
if s["grounding_rate"] < 0.95:
    print(f"GROUNDING {s['grounding_rate']:.1%} — FIX THIS FIRST.")
    print("  The model is inventing quotes. Causes, in order of likelihood:")
    print("   1. Teacher labels contained ungrounded quotes -> re-run the")
    print("      grounding filter in notebook 01; it should drop them all.")
    print("   2. Trained too long on too little data (memorising clause shapes)")
    print("      -> fewer epochs, or more documents.")
    print("   3. Targets paraphrase in source.quote -> that is a labelling bug,")
    print("      the teacher prompt forbids it.")
else:
    print(f"grounding {s['grounding_rate']:.1%} — good.")

d, st = s["detection"], s["strict"]
if d["recall"] < 0.70:
    print(f"\nRECALL {d['recall']:.2f} — missing obligations.")
    print("  Check per-type F1 below for which kinds. Usually one of:")
    print("   - a type with <20 training examples (see nb 01 class balance)")
    print("   - obligations stated without a modal verb ('Payment is due...')")
    print("   - duties buried in tables or schedules")
if d["precision"] < 0.70:
    print(f"\nPRECISION {d['precision']:.2f} — over-extracting.")
    print("  Almost always too few negative examples. Target 25-35% empty")
    print("  windows in train; check the number nb 01 printed.")
if s["empty_gold_examples"]:
    fp_rate = s["empty_gold_false_positives"] / s["empty_gold_examples"]
    print(f"\nno-obligation windows: {fp_rate:.1%} produced spurious obligations")
    if fp_rate > 0.15:
        print("  >15% is too high. Add negative examples and retrain.")
if d["f1"] - st["f1"] > 0.20:
    print(f"\nSTRICT-vs-DETECTION GAP {d['f1']-st['f1']:.2f} — finds the right")
    print("  clauses, labels them wrong. Field accuracy above shows which field.")
    print("  If obligation_type: your enum has overlapping categories — merge the")
    print("  confusable pair. If obligor: parties are not being resolved; consider")
    print("  putting the party list in the user prompt as given context.")
print("=" * 66)

# %% [markdown]
# ## Baseline comparison
#
# Do not skip this. The only number that justifies the fine-tune is the delta
# against the base model zero-shot with the full teacher prompt. If the gap is
# small, your effort belongs in prompt and schema design, not training.

# %%
from obligex import TEACHER_SYSTEM_PROMPT  # noqa: E402

del model
torch.cuda.empty_cache()

base_model, base_tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(base_model)

base_preds = []
subset = gold_rows[:40]          # 40 is enough to see the gap
for i, row in enumerate(subset, 1):
    msgs = [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},   # full instructions
        {"role": "user", "content": build_user_prompt(row["text"],
                                                      doc_name=row.get("document_name"))},
    ]
    ids = base_tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        out = base_model.generate(input_ids=ids, max_new_tokens=2048, do_sample=False,
                                  pad_token_id=base_tok.eos_token_id)
    base_preds.append(base_tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
    if i % 10 == 0:
        print(f"  {i}/{len(subset)}")

base_report = evaluate(base_preds, golds[:len(subset)], sources[:len(subset)])
ft_report = evaluate(predictions[:len(subset)], golds[:len(subset)], sources[:len(subset)])

print(f"\n{'metric':<22}{'base 0-shot':>14}{'fine-tuned':>14}{'delta':>10}")
print("-" * 60)
for label, key in [("parse rate", "parse_rate"), ("grounding", "grounding_rate")]:
    b, f = base_report.summary()[key], ft_report.summary()[key]
    print(f"{label:<22}{b:>14.3f}{f:>14.3f}{f-b:>+10.3f}")
for label, grp, key in [
    ("detection precision", "detection", "precision"),
    ("detection recall", "detection", "recall"),
    ("detection F1", "detection", "f1"),
    ("strict F1", "strict", "f1"),
]:
    b = base_report.summary()[grp][key]
    f = ft_report.summary()[grp][key]
    print(f"{label:<22}{b:>14.3f}{f:>14.3f}{f-b:>+10.3f}")
print("-" * 60)
print(f"prompt tokens/call    {len(TEACHER_SYSTEM_PROMPT)//4:>14}"
      f"{len(SYSTEM_PROMPT)//4:>14}{len(SYSTEM_PROMPT)//4 - len(TEACHER_SYSTEM_PROMPT)//4:>+10}")

# %%
# Type confusion matrix. Tells you which enum values to merge.
from collections import Counter  # noqa: E402

from obligex.evaluate import match_obligations  # noqa: E402

confusion: Counter = Counter()
for pred_raw, gold, src in zip(predictions, golds, sources):
    try:
        pred = parse_model_output(pred_raw)
    except Exception:
        continue
    matches, _, _ = match_obligations(pred.obligations, gold.obligations, src)
    for i, j, _ in matches:
        p = pred.obligations[i].obligation_type.value
        g = gold.obligations[j].obligation_type.value
        if p != g:
            confusion[(g, p)] += 1

print("most confused type pairs (gold -> predicted):")
for (g, p), c in confusion.most_common(12):
    print(f"  {c:>4}x  {g}  ->  {p}")
if confusion:
    print("\nA pair confused >15 times is usually two categories a human cannot")
    print("separate either. Merging them raises strict F1 and loses nothing a")
    print("client cares about.")

# %%
# Worst failures, read them. Twenty minutes here beats a day of tuning.
rows_by_loss = []
for idx, (pred_raw, gold, src) in enumerate(zip(predictions, golds, sources)):
    try:
        pred = parse_model_output(pred_raw)
    except Exception:
        rows_by_loss.append((99, idx, "UNPARSEABLE", 0, 0))
        continue
    matches, fp, fn = match_obligations(pred.obligations, gold.obligations, src)
    rows_by_loss.append((len(fp) + len(fn), idx, "", len(fp), len(fn)))

rows_by_loss.sort(reverse=True)
for err, idx, note, n_fp, n_fn in rows_by_loss[:5]:
    row = gold_rows[idx]
    print("=" * 66)
    print(f"{row['document_id']}#{row['window_index']}  errors={err} "
          f"(spurious={n_fp} missed={n_fn}) {note}")
    print(f"section: {row.get('section_path')}")
    print(f"--- window (first 400 chars) ---\n{row['text'][:400]}")
    print(f"--- gold ({len(golds[idx].obligations)}) ---")
    for o in golds[idx].obligations:
        print(f"  [{o.obligation_type.value}] {o.title}")
    print("--- predicted ---")
    try:
        for o in parse_model_output(predictions[idx]).obligations:
            print(f"  [{o.obligation_type.value}] {o.title}")
    except Exception as e:
        print(f"  {type(e).__name__}: {predictions[idx][:200]}")

# %% [markdown]
# # Serving on vLLM
#
# ```bash
# pip install vllm
#
# vllm serve your-username/obligex-qwen3-4b \
#   --served-model-name obligex \
#   --max-model-len 8192 \
#   --gpu-memory-utilization 0.90 \
#   --structured-outputs-config.backend xgrammar
# ```
#
# Two things that will bite you:
#
# 1. **`guided_json` was removed in vLLM v0.12.0.** The field is now
#    `structured_outputs`. Likewise `guided_decoding_backend` must be dropped
#    from requests entirely — the backend is a launch flag now.
# 2. **Serve with the exact `SYSTEM_PROMPT` you trained on.** It is imported
#    from `obligex` below for precisely this reason.
#
# Constrained decoding is belt and braces here. Your fine-tune should already
# emit valid JSON; the grammar guarantees it even on an out-of-distribution
# document, which is when you most need the guarantee.

# %%
# Offline / batch path.
OFFLINE = """
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from obligex import SYSTEM_PROMPT, build_user_prompt, guided_schema

llm = LLM(model="your-username/obligex-qwen3-4b", max_model_len=8192)

params = SamplingParams(
    temperature=0.0,
    max_tokens=2048,
    structured_outputs=StructuredOutputsParams(json=guided_schema()),
)

prompts = [
    llm.get_tokenizer().apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": build_user_prompt(w.text, page_start=w.page_start)}],
        tokenize=False, add_generation_prompt=True,
    )
    for w in windows
]
outputs = llm.generate(prompts, params)
"""
print(OFFLINE)

# %%
# Server path, and the full document pipeline. This is the function behind your
# API endpoint: bytes in, one DocumentObligations JSON out.
#
# This is where "whole document in one pass" is delivered. The client makes one
# call and receives one JSON for the entire contract. The windowing is an
# implementation detail they never see — and it is what buys you clause-level
# provenance, which a true single-pass model could not give you.
PIPELINE = '''
import asyncio, json
from openai import AsyncOpenAI
from obligex import (SYSTEM_PROMPT, build_user_prompt, guided_schema,
                     parse_model_output, windows_from_pages, merge_document)

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="none")
SCHEMA = guided_schema()
SEM = asyncio.Semaphore(16)

async def extract_window(w, doc_name):
    async with SEM:
        r = await client.chat.completions.create(
            model="obligex",
            temperature=0.0,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(
                    w.text, doc_name=doc_name, page_start=w.page_start,
                    page_end=w.page_end, section_path=w.section_path)},
            ],
            # vLLM >= 0.12. NOT guided_json.
            extra_body={"structured_outputs": {"json": SCHEMA}},
        )
    return parse_model_output(r.choices[0].message.content)

async def extract_document(pages: list[str], document_id: str, document_name: str):
    """One call per document. Returns the client-facing JSON."""
    full_text, windows = windows_from_pages(pages, target_tokens=1100,
                                            max_tokens=1400, overlap_tokens=180)
    extractions = await asyncio.gather(
        *(extract_window(w, document_name) for w in windows)
    )
    doc = merge_document(
        document_id=document_id,
        full_text=full_text,
        windows=windows,
        extractions=list(extractions),
        document_name=document_name,
        min_confidence=0.35,     # tune on your gold set
    )
    return doc.model_dump(mode="json")

# FastAPI endpoint
# from fastapi import FastAPI, UploadFile
# app = FastAPI()
#
# @app.post("/extract")
# async def extract(file: UploadFile):
#     pages = pages_from_pdf_bytes(await file.read())   # see notebook 01
#     return await extract_document(pages, document_id=file.filename,
#                                   document_name=file.filename)
'''
print(PIPELINE)

# %% [markdown]
# ## Production checklist
#
# Things that are not optional once a client is relying on this.
#
# **Correctness**
# - Re-ground every quote server-side before returning it. `merge_document`
#   does this, but assert it at the API boundary too — it is your last line of
#   defence against a fabricated clause reaching a lawyer.
# - Return `char_start` / `char_end` / `page` with every obligation so the UI
#   can highlight the source text. Reviewers trust what they can click.
# - Route `confidence < 0.6` to human review rather than hiding it. A visible
#   review queue is what makes the system defensible.
#
# **Operations**
# - Pin the model by revision, not by tag: `obligex-qwen3-4b@<commit-sha>`.
#   A silent re-push changes your client's output.
# - Log the schema version with every result. When the schema changes you need
#   to know which rows were produced under which contract.
# - Keep notebook 01's grounding filter in the request path. It is pure Python
#   and catches the failure that matters most.
#
# **Retraining**
# - Every human correction in the review queue is a gold example. Feed them
#   back; corrections on the model's own errors are worth far more per example
#   than fresh teacher labels.
# - Re-evaluate on the *same frozen gold set* each time, and never train on it.
#   The moment gold leaks into train, your metrics stop meaning anything and you
#   will not notice.
#
# **And the standing caveat**
# - This extracts and structures text. It does not give legal advice, and the
#   output needs review by someone qualified. Say so in your UI.
