# %% [markdown]
# # 02 — QLoRA fine-tune on a free Colab T4, then push to the Hub
#
# **Runtime → Change runtime type → T4 GPU** before you start.
#
# ## What a T4 actually constrains
#
# A T4 is Turing (sm75), 16 GB. Two consequences that shape every setting below:
#
# - **No bfloat16.** Turing has no bf16 units. You must train in fp16, which is
#   numerically touchier — hence the conservative learning rate and the
#   gradient-norm clipping below. Every tutorial that sets `bf16=True` was
#   written for an A100 and will fail here.
# - **No FlashAttention-2.** FA2 requires Ampere or newer. Unsloth's own kernels
#   carry the memory saving instead, which is why we use Unsloth rather than
#   plain `peft` + `transformers`.
#
# ## Why Qwen3-4B-Instruct-2507
#
# Not a guess. A 4B Qwen3-Instruct QLoRA fine-tune on the closest public analogue
# of your task — extracting 41 CUAD clause types from commercial contracts —
# reaches **detection F1 0.900 / strict F1 0.678, beating zero-shot GPT-4.1 and
# zero-shot Qwen3-8B** on strict full-span coverage. A well-tuned 4B beats a much
# larger zero-shot model on narrow extraction, and 4B is what fits here.
#
# One important caveat if you are tempted by newer Qwen: **do not use 4-bit QLoRA
# on the Qwen3.5 family.** Unsloth explicitly advises against it — those models
# show larger-than-normal quantisation error — and their recommended bf16 LoRA
# path needs hardware a T4 does not have. Qwen3-4B-Instruct-2507 is the right
# choice *for this GPU*.
#
# ## The upgrade path, costed
#
# | GPU | model | seq len | ~time for 4k examples |
# |---|---|---|---|
# | T4 free | Qwen3-4B 4-bit | 4096 | 3-5 h |
# | L4 (Pro) | Qwen3-8B 4-bit | 8192 | 2-3 h |
# | A100 40GB | Qwen3-14B 4-bit | 16384 | 1.5-2 h |
#
# An A100 for three hours is a few dollars. If this is going to a paying client,
# train the 14B — but only *after* the 4B pipeline works end to end, because
# every bug you will hit is a data bug, and data bugs are cheaper to find on a T4.

# %%
# Unsloth MUST be imported before transformers / peft / trl. It monkey-patches
# them on import, and patches only apply to modules not yet loaded. An import
# in the wrong order costs you the speed and memory savings silently — no error,
# just OOM later at a sequence length that should have fit.
!pip install -q unsloth
!pip install -q --no-deps --upgrade "trl>=0.12" "peft>=0.13" accelerate bitsandbytes

import unsloth  # noqa: F401  <- FIRST. Do not move.
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

import torch

print("torch", torch.__version__)
print("gpu  ", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"vram  {props.total_memory / 1e9:.1f} GB   sm_{props.major}{props.minor}")

IS_TURING = props.major == 7
USE_FP16 = IS_TURING or not torch.cuda.is_bf16_supported()
print(f"\ndtype: {'fp16 (Turing — no bf16)' if USE_FP16 else 'bf16'}")
if IS_TURING:
    print("T4 detected: MAX_LEN capped at 4096, 4B model, batch size 1.")

# %%
import json
import os
from pathlib import Path

try:
    from google.colab import drive, userdata  # type: ignore

    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/obligex")
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    ROOT = Path("./obligex_work")

DATA = ROOT / "data"
OUT = ROOT / "runs"
OUT.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(ROOT))
from obligex import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

# This is the train/serve contract. It is imported, never retyped. If you find
# yourself pasting a system prompt into this notebook, stop — you are about to
# train a model that is conditioned on a string your server does not send.
print("student system prompt:", len(SYSTEM_PROMPT) // 4, "tokens")
print(repr(SYSTEM_PROMPT[:90]) + " ...")

# %% [markdown]
# ## Config

# %%
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
MAX_LEN = 4096 if IS_TURING else 8192

# LoRA. r=32 is the sweet spot for structured extraction: r=8/16 underfits the
# enum vocabularies, r=64 adds VRAM and training time for no measurable gain on
# a task this constrained.
LORA_R = 32
LORA_ALPHA = 32          # alpha == r keeps the effective scale at 1.0
LORA_DROPOUT = 0.0       # Unsloth's fused path is optimised for 0

LR = 1e-4                # conservative: fp16 + structured output is loss-spiky
EPOCHS = 3
BATCH = 1                # 4096 tokens on 16 GB leaves no room for more
GRAD_ACCUM = 8           # effective batch 8
WARMUP_RATIO = 0.05
SEED = 3407

HF_REPO = "your-username/obligex-qwen3-4b"   # <-- change this

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,            # None = auto: fp16 on T4/V100, bf16 on Ampere+
    load_in_4bit=True,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    # All attention AND MLP projections. Attention-only LoRA is a common
    # shortcut that measurably hurts structured generation — the MLP layers are
    # where format and vocabulary adherence live.
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",   # the setting that makes 4096 fit
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\ntrainable {trainable/1e6:.1f}M / {total/1e9:.2f}B = {100*trainable/total:.3f}%")

# %% [markdown]
# ## Data
#
# Rendering happens once, here, with the tokenizer's own chat template. The
# three turns are system (the short student prompt), user (metadata + window),
# assistant (compact JSON).

# %%
from datasets import Dataset  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def render(row: dict) -> dict:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                row["text"],
                doc_name=row.get("document_name"),
                page_start=row.get("page_start"),
                page_end=row.get("page_end"),
                section_path=row.get("section_path"),
            ),
        },
        {"role": "assistant", "content": row["target_json"]},
    ]
    # Qwen3-*-Instruct-2507 is non-thinking, so there is no <think> block to
    # suppress. Hybrid Qwen3 checkpoints DO need enable_thinking=False or the
    # template injects thinking scaffolding your targets do not contain.
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    return {"text": text}


train_rows = load_jsonl(DATA / "train.jsonl")
val_rows = load_jsonl(DATA / "val.jsonl")
print(f"train {len(train_rows)}   val {len(val_rows)}")

train_ds = Dataset.from_list(train_rows).map(render, desc="render train")
val_ds = Dataset.from_list(val_rows).map(render, desc="render val")

print("\n--- one rendered example (truncated) ---")
print(train_ds[0]["text"][:700])
print("   ...")
print(train_ds[0]["text"][-400:])

# %%
# SEQUENCE LENGTH AUDIT — run this, read it, then decide.
#
# Any example longer than MAX_LEN is TRUNCATED FROM THE RIGHT, which cuts the
# JSON target mid-object. You are then explicitly teaching the model to emit
# unterminated JSON. This does not show up in the loss curve. It shows up as
# unparseable output in production.
lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in train_ds["text"]]
lens_sorted = sorted(lens)
n = len(lens_sorted)
over = sum(1 for l in lens if l > MAX_LEN)

print(f"MAX_LEN = {MAX_LEN}")
print(f"  p50 {lens_sorted[n//2]}")
print(f"  p90 {lens_sorted[int(n*0.90)]}")
print(f"  p95 {lens_sorted[int(n*0.95)]}")
print(f"  p99 {lens_sorted[int(n*0.99)]}")
print(f"  max {lens_sorted[-1]}")
print(f"\nexamples over MAX_LEN: {over} / {n}  ({over/n:.2%})")

if over:
    print("\n" + "!" * 70)
    print(f"{over} examples WILL BE TRUNCATED mid-JSON. Pick one:")
    print("  a) lower TARGET_TOKENS in notebook 01 and re-window (best)")
    print("  b) drop these examples (fine if under ~2%)")
    print("  c) raise MAX_LEN (needs a bigger GPU than a T4)")
    print("!" * 70)
    # (b), applied: safer to drop than to train on broken targets.
    keep = [i for i, l in enumerate(lens) if l <= MAX_LEN]
    train_ds = train_ds.select(keep)
    print(f"\ndropped {over}; training on {len(train_ds)} examples")
else:
    print("\nOK — nothing truncated.")

# %% [markdown]
# ## Trainer
#
# Two API details that break most copy-pasted code:
#
# - `SFTConfig(max_length=...)` — TRL renamed this from `max_seq_length`.
# - `SFTTrainer(processing_class=tokenizer)` — `tokenizer=` is the old,
#   removed form.

# %%
from trl import SFTConfig, SFTTrainer  # noqa: E402

args = SFTConfig(
    output_dir=str(OUT / "qwen3-4b-obligations"),
    dataset_text_field="text",
    max_length=MAX_LEN,
    packing=False,                      # packing corrupts response-only masking
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    optim="adamw_8bit",                 # saves ~1.5 GB vs adamw_torch
    weight_decay=0.01,
    max_grad_norm=0.3,                  # fp16 safety: clip hard
    fp16=USE_FP16,
    bf16=not USE_FP16,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,         # NOT tokenizer=
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=args,
)

# %%
# Mask the prompt; train only on the assistant turn.
#
# Without this the model spends most of its gradient learning to reproduce the
# system prompt and the contract text — which it will never need to generate.
#
# The marker strings are Qwen's ChatML. If they are wrong, every label becomes
# -100, the loss is exactly 0.0, and training "succeeds" while learning nothing.
# The assertion below is not optional.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

trainer = train_on_responses_only(
    trainer,
    instruction_part=INSTRUCTION_PART,
    response_part=RESPONSE_PART,
)

sample = trainer.train_dataset[0]
labels = sample["labels"]
ids = sample["input_ids"]
supervised = [(i, t) for i, t in zip(labels, ids) if i != -100]

print(f"tokens total      {len(labels)}")
print(f"tokens supervised {len(supervised)}  ({len(supervised)/len(labels):.1%})")

assert supervised, (
    "ALL LABELS ARE -100 — the marker strings do not match this chat template. "
    "Training would run and learn nothing. Print tokenizer.chat_template and fix "
    "INSTRUCTION_PART / RESPONSE_PART."
)
decoded = tokenizer.decode([t for _, t in supervised])
print(f"\nsupervised text begins: {decoded[:120]!r}")
print(f"supervised text ends:   {decoded[-80:]!r}")
assert decoded.lstrip().startswith("{"), (
    f"Supervised span should start at the JSON target but starts with "
    f"{decoded.lstrip()[:60]!r}. The response_part marker is misaligned."
)
frac = len(supervised) / len(labels)
if frac > 0.75:
    print("\n!! >75% supervised — masking probably did not apply. Check markers.")
else:
    print("\nMasking verified: loss is computed on the JSON target only.")

# %%
import time  # noqa: E402

t0 = time.time()
stats = trainer.train()
mins = (time.time() - t0) / 60
print(f"\ntrained in {mins:.1f} min")
print(f"final train loss {stats.training_loss:.4f}")
print(f"peak vram        {torch.cuda.max_memory_reserved()/1e9:.2f} GB")

# %%
# Loss sanity check. For this task, expect train loss to land roughly in
# 0.05-0.25. Interpret the extremes:
#
#   loss == 0.0 exactly  -> masking is broken (see the assertion above)
#   loss < 0.02          -> memorisation; too many epochs or duplicated windows
#   loss > 0.8 and flat  -> LR too low, or targets inconsistent with each other
#                           (two labellers disagreeing on the same clause type)
#   val >> train, rising -> overfitting; stop at the best checkpoint
hist = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
tr = [(h["step"], h["loss"]) for h in hist if "loss" in h]
ev = [(h["step"], h["eval_loss"]) for h in hist if "eval_loss" in h]
print(f"train loss: {tr[0][1]:.4f} (step {tr[0][0]}) -> {tr[-1][1]:.4f} (step {tr[-1][0]})")
if ev:
    print(f"val   loss: {ev[0][1]:.4f} -> {ev[-1][1]:.4f}")
    best = min(ev, key=lambda x: x[1])
    print(f"best val {best[1]:.4f} at step {best[0]}"
          + ("  <-- val rose after this; prefer this checkpoint"
             if best[0] != ev[-1][0] else ""))

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 3.5))
    plt.plot(*zip(*tr), label="train", lw=1)
    if ev:
        plt.plot(*zip(*ev), label="val", marker="o", ms=3)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
except Exception:
    pass

# %% [markdown]
# ## Smoke test
#
# Before saving anything, make the model do the job once. Constrained decoding
# is NOT in play here — this is raw generation, which is exactly what you want
# to see. A fine-tune that needs a grammar to produce valid JSON has not learned
# the format; it is being forced into it, and it will fight the grammar on
# content too.

# %%
from obligex import parse_model_output, ground_obligations  # noqa: E402

FastLanguageModel.for_inference(model)

gold_rows = load_jsonl(DATA / "gold.jsonl")
probe = [r for r in gold_rows if r["n_obligations"] > 0][:3]
probe += [r for r in gold_rows if r["n_obligations"] == 0][:1]

for row in probe:
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
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        input_ids=inputs,
        max_new_tokens=2048,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

    print("=" * 66)
    print(f"{row['document_id']}#{row['window_index']}   gold={row['n_obligations']} obligations")
    try:
        ext = parse_model_output(raw)
        kept, rej = ground_obligations(ext.obligations, row["text"])
        print(f"  parsed OK: {len(ext.obligations)} predicted, "
              f"{len(kept)} grounded, {len(rej)} FABRICATED")
        for o in ext.obligations[:4]:
            flag = "" if any(k.title == o.title for k in kept) else "  <-- NOT IN DOCUMENT"
            print(f"    [{o.obligation_type.value}] {o.title}{flag}")
    except Exception as e:
        print(f"  PARSE FAILED: {type(e).__name__}: {e}")
        print(f"  raw: {raw[:300]}")

# %% [markdown]
# ## Save and push
#
# Three artefacts, because you said self-hosted vLLM but plans change:
#
# | artefact | size | use |
# |---|---|---|
# | LoRA adapter | ~130 MB | version control, swap at runtime |
# | merged 16-bit | ~8 GB | **vLLM — load this one** |
# | GGUF q4_k_m | ~2.5 GB | llama.cpp / Ollama fallback |
#
# vLLM can serve a bare adapter with `--enable-lora`, but a merged model avoids
# the per-request adapter overhead and is simpler to pin a version of. Merge.

# %%
run_cfg = {
    "base_model": BASE_MODEL,
    "max_len": MAX_LEN,
    "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
    "lr": LR,
    "epochs": EPOCHS,
    "effective_batch": BATCH * GRAD_ACCUM,
    "precision": "fp16" if USE_FP16 else "bf16",
    "n_train": len(train_ds),
    "n_val": len(val_ds),
    "final_train_loss": float(stats.training_loss),
    "system_prompt": SYSTEM_PROMPT,     # the contract. Serve with this exact string.
    "instruction_part": INSTRUCTION_PART,
    "response_part": RESPONSE_PART,
    "seed": SEED,
}
(OUT / "run_config.json").write_text(json.dumps(run_cfg, indent=2))
print(json.dumps(run_cfg, indent=2)[:600])

# %%
# 1. adapter (fast, do this first so a later failure costs nothing)
model.save_pretrained(str(OUT / "adapter"))
tokenizer.save_pretrained(str(OUT / "adapter"))
model.push_to_hub(HF_REPO + "-adapter", token=os.environ["HF_TOKEN"])
tokenizer.push_to_hub(HF_REPO + "-adapter", token=os.environ["HF_TOKEN"])
print("adapter pushed")

# %%
# 2. merged 16-bit for vLLM.
#
# On a T4 this is the step most likely to fail: merging needs the full fp16
# model in RAM, and Colab's free tier gives ~12 GB. If it OOMs, push the adapter
# (done above) and merge on the serving box instead:
#
#     from peft import AutoPeftModelForCausalLM
#     m = AutoPeftModelForCausalLM.from_pretrained("you/obligex-...-adapter",
#                                                  torch_dtype="float16")
#     m.merge_and_unload().save_pretrained("merged")
model.push_to_hub_merged(
    HF_REPO,
    tokenizer,
    save_method="merged_16bit",
    token=os.environ["HF_TOKEN"],
)
print(f"merged model pushed to {HF_REPO}")

# %%
# 3. GGUF (optional). Builds llama.cpp from source — 10-20 min.
try:
    model.push_to_hub_gguf(
        HF_REPO + "-gguf",
        tokenizer,
        quantization_method=["q4_k_m", "q8_0"],
        token=os.environ["HF_TOKEN"],
    )
    print("gguf pushed")
except Exception as e:
    print(f"gguf export skipped: {e}")

# %%
# Model card. Write down the eval numbers from notebook 03 when you have them —
# an undocumented extraction model is unauditable, and in legal tech you will be
# asked how it was measured.
card = f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags: [obligation-extraction, contract-analysis, structured-output, legal-nlp, qwen3, unsloth]
---

# obligex — contract obligation extraction (Qwen3-4B QLoRA)

Extracts contractual obligations from a window of a document and returns JSON
conforming to the `obligex` schema.

## Serving

Load the **merged** model and enforce the schema with constrained decoding:

```bash
vllm serve {HF_REPO} \\
  --max-model-len 8192 \\
  --structured-outputs-config.backend xgrammar
```

```python
from obligex import SYSTEM_PROMPT, build_user_prompt, guided_schema
# vLLM >= 0.12 removed guided_json. Use structured_outputs.
extra_body={{"structured_outputs": {{"json": guided_schema()}}}}
```

**The system prompt is part of the model.** Serve with the exact
`SYSTEM_PROMPT` string from `obligex` — a different prompt degrades accuracy
sharply.

## Training

| | |
|---|---|
| base | `{BASE_MODEL}` |
| method | QLoRA 4-bit, r={LORA_R}, alpha={LORA_ALPHA} |
| modules | all attention + MLP projections |
| seq len | {MAX_LEN} |
| lr / sched | {LR} cosine, {WARMUP_RATIO:.0%} warmup |
| epochs | {EPOCHS} |
| effective batch | {BATCH * GRAD_ACCUM} |
| precision | {"fp16" if USE_FP16 else "bf16"} |
| loss | assistant turn only |
| train / val | {len(train_ds)} / {len(val_ds)} windows |
| final train loss | {stats.training_loss:.4f} |

Labels were generated by a teacher model over two passes, filtered by **span
grounding** (any obligation whose `source.quote` is absent from the source text
was discarded) and by cross-pass agreement. Splits are **by document**, not by
window.

## Evaluation

Fill in from `03_eval_and_serve.ipynb`:

| metric | value |
|---|---|
| detection F1 | |
| strict F1 | |
| grounding rate | |
| false positives on no-obligation windows | |

## Limitations

- Trained on English-language commercial contracts; other jurisdictions and
  languages are untested.
- Operates on ~1100-token windows. Document-level output requires the
  `obligex.merge` step; the model does not see a whole contract at once.
- Not legal advice. Output requires review by a qualified person.
"""
(OUT / "README.md").write_text(card)

from huggingface_hub import HfApi  # noqa: E402

HfApi().upload_file(
    path_or_fileobj=str(OUT / "README.md"),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    token=os.environ["HF_TOKEN"],
)
print("model card pushed")
print("\nNext: 03_eval_and_serve.ipynb")
