# obligex — fine-tuning an open LLM to extract contract obligations as JSON

Takes client documents (PDF, DOCX, `.eml`, `.msg`, text), extracts every
contractual obligation, validates it against the source text, and returns one
structured JSON document per file.

Built for: **free Colab T4 → Hugging Face → self-hosted vLLM.**

---

## Read this first: the two things you asked for that conflict

You specified **whole-document single-pass** extraction and a **free T4**. Those
cannot both hold, and the reason is hardware, not tuning:

A T4 is 16 GB and Turing-class (sm75) — **no bfloat16, no FlashAttention-2.**
Fine-tuning at 32k context is out of reach on it at any model size worth
shipping. But single-pass is also the wrong architecture here, for reasons that
survive buying a bigger GPU:

| | why single-pass fails |
|---|---|
| **Output length** | A 120-page MSA holds 100-250 obligations ≈ 15-30k tokens of JSON in one completion. One bad bracket at token 14,000 loses the document. |
| **Recall** | Long-context retrieval is not uniform. Payment terms live in the middle, which is the weakest region. |
| **Provenance** | Your client will ask "which clause?" about every obligation. Windows give page + character offsets for free. |
| **Trainability** | 4k windows train on a free T4. 32k does not. |

**What this repo does instead:** section-aware overlapping windows → per-window
extraction → deterministic merge and de-duplication. Your client makes **one API
call and gets one JSON for the whole contract.** Single-pass is satisfied at the
pipeline boundary, which is where it actually matters — and you gain
clause-level citation, which a true single-pass model could not give you.

---

## Why Qwen3-4B-Instruct-2507

Not a guess. The closest public analogue of your task — extracting 41 CUAD
clause types from commercial contracts with a 4B Qwen3-Instruct QLoRA — reaches
**detection F1 0.900 / strict F1 0.678, beating zero-shot GPT-4.1 and zero-shot
Qwen3-8B** on strict full-span coverage. A tuned 4B beats a far larger zero-shot
model on narrow extraction, and 4B is what fits in 16 GB.

**Do not reach for Qwen3.5.** Unsloth explicitly advises against 4-bit QLoRA on
that family (larger-than-normal quantisation error), and its recommended bf16
LoRA path needs hardware a T4 does not have.

Upgrade path, once the pipeline works:

| GPU | model | seq len | ~time / 4k examples |
|---|---|---|---|
| T4 (free) | Qwen3-4B 4-bit | 4096 | 3-5 h |
| L4 (Pro) | Qwen3-8B 4-bit | 8192 | 2-3 h |
| A100 40GB | Qwen3-14B 4-bit | 16384 | 1.5-2 h |

Train the 4B first regardless. Every bug you will hit is a data bug, and data
bugs are cheaper to find on free hardware.

---

## Layout

```
obligex/
  schema.py      Pydantic schema, both system prompts, span grounding,
                 vLLM-safe JSON Schema export        <- single source of truth
  chunking.py    heading detection, section-aware overlapping windows
  merge.py       cross-window de-duplication, party reconciliation
  evaluate.py    detection / strict F1, grounding rate, type confusion
notebooks/
  01_build_dataset.ipynb    raw docs -> train/val/gold JSONL
  02_finetune_t4.ipynb      QLoRA on a free T4 -> push to Hub
  03_eval_and_serve.ipynb   metrics, baseline delta, vLLM serving
tests/
  test_pipeline.py          61 checks, all passing
```

All three notebooks import `obligex`. **Never retype the system prompt into a
notebook** — a fine-tune conditioned on one prompt string degrades sharply when
served with another, and this is the most common way a working model breaks in
production.

```bash
pip install 'pydantic>=2.6'
python -m tests.test_pipeline     # 61 checks
```

---

## The order of work

### 1. Data (notebook 01) — this is where the project is won or lost

You have raw documents and no labels, so the first deliverable is a labelled
dataset, not a trained model. A careful weekend here beats a month of
hyperparameter search.

**Before client files, run the whole pipeline on CUAD** (510 commercial
contracts, CC BY 4.0). You will find half a dozen bugs on public data for free.

**Confidentiality.** Sending client documents to a third-party API to generate
labels may breach the terms you signed, and "it was only training data" is not a
defence. Notebook 01 supports both an API teacher and a local Qwen3-32B-AWQ
teacher on hardware you control (roughly 3-5 points worse on hard clauses; the
data never leaves your infrastructure).

Three mechanisms in notebook 01 that are not optional:

- **Span grounding.** Every obligation must quote the source verbatim. If the
  quote is not in the text, it is fabricated and dropped automatically. A free,
  deterministic hallucination filter — it typically removes 2-8% of teacher
  output, and it is the single highest-value line of code in the repo.
- **Two teacher passes.** Where passes agree → silver training label. Where they
  disagree → human review queue. This aims your scarce review time at the
  examples that need it. Budget **150-250 reviewed windows** before training;
  that is the difference between a number you can quote a client and one you
  cannot.
- **Negative examples.** Target **25-35% of training windows with
  `obligations: []`.** Skip this and the model learns its job is to produce
  obligations, then does so for tables of contents and signature blocks. Recall
  looks fine; precision collapses on real documents.

Splits are **by document, never by window** — overlapping windows share text,
and window-level splitting inflates scores by 10-20 points.

### 2. Fine-tune (notebook 02)

QLoRA, r=32, all attention **and MLP** projections (attention-only is a common
shortcut that measurably hurts format adherence), loss on the assistant turn
only, fp16 with hard gradient clipping because Turing has no bf16.

Three guards worth knowing about:

- **Sequence-length audit before training.** Anything over `MAX_LEN` is
  truncated from the right — mid-JSON — which explicitly teaches the model to
  emit unterminated objects. This never appears in the loss curve. It appears as
  unparseable output in production.
- **A hard assertion that response-masking worked.** `train_on_responses_only`
  fails *silently* on wrong marker strings: every label becomes -100, loss is
  exactly 0.0, and training "succeeds" having learned nothing.
- **Two prompts, deliberately.** The teacher prompt is ~1059 tokens because a
  frontier model has never seen your schema. The student prompt is ~66 tokens
  because after training the schema lives in the weights. **Saves 993 tokens per
  call — 24% of a 4096 budget — reclaimed for contract text.**

### 3. Evaluate and serve (notebook 03)

**Look at grounding rate before anything else.** A model scoring 0.85 detection
while quoting text absent from the contract is unshippable in legal tech, and
detection F1 cannot see the problem — a fabricated quote that overlaps a real
span still matches.

Schema validity is deliberately *not* a headline metric: under constrained
decoding it is 100% by construction, so it measures your decoder, not your
model. The eval therefore runs **unconstrained**.

Notebook 03 also computes the **base-model zero-shot delta**. If that gap is
small, your effort belongs in prompt and schema design rather than training —
better to learn that from a table than after three weeks.

---

## API facts that break copy-pasted code

Verified against current docs, November 2026. Every one of these appears wrong
in tutorials still ranking on search.

| | |
|---|---|
| **vLLM** | `guided_json` was **removed in v0.12.0**. Use `structured_outputs`: `StructuredOutputsParams(json=schema)` offline, `extra_body={"structured_outputs": {"json": schema}}` on the server. Drop `guided_decoding_backend` from requests — it is a launch flag now. |
| **TRL** | `SFTConfig(max_length=...)`, renamed from `max_seq_length`. `SFTTrainer(processing_class=tokenizer)`, not `tokenizer=`. |
| **Unsloth** | `import unsloth` **before** transformers / peft / trl. It monkey-patches them on import and patches only apply to not-yet-imported modules. Wrong order = no error, just an OOM later at a length that should have fit. |
| **Qwen3** | `-Instruct-2507` is non-thinking, so no `<think>` suppression needed. Hybrid Qwen3 checkpoints *do* need `enable_thinking=False`, or the template injects scaffolding your targets lack. |

---

## Schema shape

```json
{
  "obligations": [{
    "obligation_id": "OBL-0007",
    "title": "Pay undisputed invoices within 30 days",
    "description": "Globex must pay each undisputed invoice within 30 days of receipt.",
    "obligation_type": "payment",
    "obligor": "P2", "obligee": "P1",
    "modality": "shall", "binding_strength": "mandatory",
    "trigger": "receipt of a valid undisputed invoice",
    "timing": {"timing_type": "one_time_relative", "relative_days": 30,
               "relative_to": "receipt of invoice",
               "deadline_verbatim": "within thirty (30) days"},
    "financial": null,
    "consequence_of_breach": "Interest accrues at 1.5% per month.",
    "source": {"quote": "The Customer shall pay each undisputed invoice within thirty (30) days of receipt.",
               "clause_reference": "2.2", "page": 4,
               "char_start": 8812, "char_end": 8897},
    "confidence": 0.93
  }],
  "parties": [{"party_id": "P2", "name": "Globex Industries Ltd.",
               "short_name": "the Customer", "role": "customer_or_buyer"}],
  "document_type": "master_services_agreement"
}
```

Two design decisions worth defending:

**`modality` is a required field** so that a *right* ("Customer **may** audit")
cannot be silently recorded as a *duty*. Forcing the model to commit to the
modal verb it saw makes that error visible and filterable instead of buried in
prose. It is the most expensive error class in obligation extraction.

**`source.quote` is required and verbatim**, which is what makes the free
hallucination filter possible at every stage — label-time, train-time and
request-time.

Enums are short on purpose. Every value you add costs training examples; a value
with no training data is a value the model will mispredict into. Notebook 01
flags any enum value with fewer than 20 examples.

---

## Not legal advice

This extracts and structures text. It does not interpret contracts, and output
requires review by a qualified person. Say so in your UI.

Data: CUAD v1 © The Atticus Project, CC BY 4.0 (Hendrycks, Burns, Chen, Ball,
NeurIPS 2021). Base model Apache 2.0.
