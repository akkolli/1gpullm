"""
CORE benchmark evaluation.

Adapted from Karpathy's nanochat (https://github.com/karpathy/nanochat),
which itself implements DCLM CORE (https://arxiv.org/abs/2406.11794) ==
the 22 low-variance tasks from the MosaicML Eval Gauntlet v0.3.

Per example we render N prompts (one per choice for MC/schema, or just the
prompt-without-continuation for LM), tokenize them, then score:

  - multiple_choice: pick the choice with the lowest mean per-token loss
                     (length-normalized log-likelihood under teacher forcing)
  - schema:          same scoring, common span is the suffix
  - language_modeling: teacher-forced argmax over the continuation must equal
                       every gold continuation token (single forward pass).

Aligns with nanochat's scoring so CORE numbers are directly comparable.
Common-prefix length (not strict prefix equality) is used to delimit the
continuation, which is necessary for byte-level BPE — the boundary token
between the prompt and its continuation often differs between the two
encodings (e.g. encode("...the ") vs encode("...the apple") diverge at
" apple" being a single merged token).

Per-task accuracy is centered against a published random baseline:
  centered = (acc - r) / (1 - r)
CORE = unweighted mean of the 22 centered scores.

Reference value: GPT-2 124M scores ~0.114, GPT-2 1.5B scores ~0.257.
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile

import torch
import torch.nn.functional as F
import yaml
from jinja2 import Template
from tokenizers import Tokenizer

from llm import LLM, LLMConfig

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR = "eval_bundle"
TOKENIZER_PATH = "tokenizer/tokenizer.json"


# -------------- prompt rendering ---------------

_MC_TPL = Template(
    "{%- for example in fewshot_examples -%}"
    "{{ example.query }}{{ continuation_delimiter }}"
    "{{ example.choices[example.gold] }}\n\n"
    "{% endfor -%}"
    "{{ item.query }}{{ continuation_delimiter }}{{ choice }}"
)

_SCHEMA_TPL = Template(
    "{%- for example in fewshot_examples -%}"
    "{{ example.context_options[example.gold] }}{{ continuation_delimiter }}"
    "{{ example.continuation }}\n\n"
    "{% endfor -%}"
    "{{ context }}{{ continuation_delimiter }}{{ item.continuation }}"
)

_LM_TPL = Template(
    "{%- for example in fewshot_examples -%}"
    "{{ example.context | trim }}{{ continuation_delimiter }}"
    "{{ example.continuation }}\n\n"
    "{% endfor -%}"
    "{{ item.context | trim }}{{ continuation_delimiter }}"
    "{% if include_continuation %}{{ item.continuation }}{% endif %}"
)


def render_prompts_mc(item, delim, fewshot):
    return [
        _MC_TPL.render(
            choice=c, fewshot_examples=fewshot, continuation_delimiter=delim, item=item
        )
        for c in item["choices"]
    ]


def render_prompts_schema(item, delim, fewshot):
    return [
        _SCHEMA_TPL.render(
            context=ctx, fewshot_examples=fewshot, continuation_delimiter=delim, item=item
        )
        for ctx in item["context_options"]
    ]


def render_prompts_lm(item, delim, fewshot):
    p_without = _LM_TPL.render(
        include_continuation=False,
        fewshot_examples=fewshot,
        continuation_delimiter=delim,
        item=item,
    ).strip()
    p_with = _LM_TPL.render(
        include_continuation=True,
        fewshot_examples=fewshot,
        continuation_delimiter=delim,
        item=item,
    )
    return [p_without, p_with]


def find_common_length(seqs, direction="left"):
    min_len = min(len(s) for s in seqs)
    indices = range(min_len) if direction == "left" else range(-1, -min_len - 1, -1)
    for i, idx in enumerate(indices):
        tok = seqs[0][idx]
        if not all(s[idx] == tok for s in seqs):
            return i
    return min_len


# -------------- tokenizer wrapper ---------------


class TokWrapper:
    """Calls our HF BPE tokenizer with special-tokens disabled.

    The trained tokenizer has a TemplateProcessor that appends <|endoftext|>
    after every encoded string. That breaks the common-prefix/suffix
    detection across choices, so we bypass it via add_special_tokens=False
    and then prepend a single EOT manually as a "start of document" marker
    (analogous to how nanochat prepends BOS).
    """

    def __init__(self, path):
        self.tk = Tokenizer.from_file(path)
        self.eot_id = self.tk.token_to_id("<|endoftext|>")
        if self.eot_id is None:
            raise RuntimeError("tokenizer is missing <|endoftext|>")

    def encode_many(self, texts, prepend_id=None):
        encs = self.tk.encode_batch(texts, add_special_tokens=False)
        out = []
        for e in encs:
            ids = list(e.ids)
            if prepend_id is not None:
                ids = [prepend_id] + ids
            out.append(ids)
        return out


# -------------- model forward + batching ---------------


@torch.no_grad()
def forward_model(model, input_ids):
    """Returns (losses, argmax preds). losses[:, -1] is nan (no target)."""
    B, S = input_ids.shape
    logits = model(input_ids)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        logits.reshape(B * S, -1),
        targets.reshape(B * S),
        reduction="none",
    ).view(B, S)
    losses[:, -1] = float("nan")
    preds = logits.argmax(dim=-1)
    return losses, preds


def stack_pad(seqs, pad_id):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def crop_to_max_len(tokens, starts, ends, max_seq_len):
    """Left-truncate each row to fit max_seq_len, shifting the answer-span
    indices accordingly. If the answer span itself can't fit (s would be < 1),
    we flag the row by returning s=None for that row — caller treats it as
    a wrong answer."""
    new_t, new_s, new_e = [], [], []
    for t, s, e in zip(tokens, starts, ends):
        if len(t) > max_seq_len:
            crop = len(t) - max_seq_len
            t = t[-max_seq_len:]
            s -= crop
            e -= crop
        if s < 1 or e <= s:
            new_t.append(t)
            new_s.append(None)
            new_e.append(None)
        else:
            new_t.append(t)
            new_s.append(s)
            new_e.append(e)
    return new_t, new_s, new_e


# -------------- per-example eval ---------------


@torch.no_grad()
def evaluate_example(idx, model, tk, data, device, task_meta, max_seq_len):
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    delim = task_meta["continuation_delimiter"]

    fewshot = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        avail = [i for i in range(len(data)) if i != idx]
        if len(avail) >= num_fewshot:
            fewshot = [data[i] for i in rng.sample(avail, num_fewshot)]

    if task_type == "language_modeling":
        return _eval_lm(model, tk, item, delim, fewshot, device, max_seq_len)

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, delim, fewshot)
        tokens = tk.encode_many(prompts, prepend_id=tk.eot_id)
        ans_start = find_common_length(tokens, "left")
        starts = [ans_start] * len(prompts)
        ends = [len(t) for t in tokens]
    elif task_type == "schema":
        prompts = render_prompts_schema(item, delim, fewshot)
        tokens = tk.encode_many(prompts, prepend_id=tk.eot_id)
        suf_len = find_common_length(tokens, "right")
        ends = [len(t) for t in tokens]
        starts = [e - suf_len for e in ends]
    else:
        raise ValueError(f"unsupported task type: {task_type}")

    tokens, starts, ends = crop_to_max_len(tokens, starts, ends, max_seq_len)
    if all(s is None for s in starts):
        return False  # answer span never fits — give up

    input_ids = stack_pad(tokens, pad_id=tk.eot_id).to(device)
    losses, _ = forward_model(model, input_ids)

    mean_losses = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        if s is None:
            mean_losses.append(float("inf"))
            continue
        m = losses[i, s - 1 : e - 1].mean().item()
        mean_losses.append(float("inf") if m != m else m)  # nan -> inf
    pred_idx = mean_losses.index(min(mean_losses))
    return pred_idx == item["gold"]


@torch.no_grad()
def _eval_lm(model, tk, item, delim, fewshot, device, max_seq_len):
    """LM-task scoring: teacher-forced argmax over the gold continuation in a
    single forward pass. Common-prefix length handles BPE boundary tokens
    where encode(prompt) is not a strict prefix of encode(prompt+continuation).
    """
    prompt_without, prompt_with = render_prompts_lm(item, delim, fewshot)
    t_without, t_with = tk.encode_many([prompt_without, prompt_with], prepend_id=tk.eot_id)

    common = find_common_length([t_without, t_with], "left")
    if common == 0 or common >= len(t_with):
        return False

    # left-truncate to fit the model's context, shifting `common` along
    if len(t_with) > max_seq_len:
        crop = len(t_with) - max_seq_len
        t_with = t_with[crop:]
        common -= crop
        if common < 1:
            return False

    input_ids = torch.tensor([t_with], dtype=torch.long, device=device)
    _, preds = forward_model(model, input_ids)
    # preds[t] = argmax of logits at position t, predicting token at t+1.
    # Continuation tokens are at positions [common, len(t_with)); their
    # predictions come from logits at positions [common-1, len(t_with)-1).
    pred_continuation = preds[0, common - 1 : len(t_with) - 1].tolist()
    gold_continuation = t_with[common:]
    return pred_continuation == gold_continuation


def evaluate_task(model, tk, data, device, task_meta, max_seq_len):
    n = len(data)
    correct = 0
    t0 = time.time()
    for idx in range(n):
        if evaluate_example(idx, model, tk, data, device, task_meta, max_seq_len):
            correct += 1
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(
                f"    [{idx + 1}/{n}] acc={correct / (idx + 1):.4f} "
                f"@ {rate:.1f} ex/s",
                end="\r",
                flush=True,
            )
    return correct / n if n > 0 else 0.0


# -------------- bundle download ---------------


def ensure_eval_bundle():
    if os.path.exists(EVAL_BUNDLE_DIR):
        return
    print(f"[bundle] downloading {EVAL_BUNDLE_URL}")
    with urllib.request.urlopen(EVAL_BUNDLE_URL) as resp:
        data = resp.read()
    print(f"[bundle] unzipping ({len(data) / 1e6:.1f}MB)")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(".")
    if not os.path.exists(EVAL_BUNDLE_DIR):
        raise RuntimeError(f"unzip didn't produce {EVAL_BUNDLE_DIR}/")


# -------------- top-level orchestrator ---------------


def evaluate_core(model, tk, device, max_seq_len, max_per_task=-1):
    ensure_eval_bundle()
    config_path = os.path.join(EVAL_BUNDLE_DIR, "core.yaml")
    data_base = os.path.join(EVAL_BUNDLE_DIR, "eval_data")
    meta_path = os.path.join(EVAL_BUNDLE_DIR, "eval_meta_data.csv")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered = {}
    print(f"[core] {len(tasks)} tasks, max_per_task={max_per_task}")
    for task in tasks:
        label = task["label"]
        meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }
        path = os.path.join(data_base, meta["dataset_uri"])
        with open(path) as f:
            data = [json.loads(line) for line in f]
        # match nanochat: shuffle with fixed seed, then truncate
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        t0 = time.time()
        print(
            f"[core] {label} ({meta['num_fewshot']}-shot {meta['task_type']}, "
            f"n={len(data)})..."
        )
        acc = evaluate_task(model, tk, data, device, meta, max_seq_len)
        r = random_baselines[label] / 100.0
        c = (acc - r) / (1.0 - r) if r < 1.0 else 0.0
        results[label] = acc
        centered[label] = c
        print(
            f"  -> acc={acc:.4f}  centered={c:.4f}  ({time.time() - t0:.1f}s)        "
        )

    core = sum(centered.values()) / len(centered)
    return {"results": results, "centered_results": centered, "core_metric": core}


def write_results_csv(out, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
        for label in out["results"]:
            f.write(
                f"{label:<35}, {out['results'][label]:<10.6f}, "
                f"{out['centered_results'][label]:<10.6f}\n"
            )
        f.write(f"{'CORE':<35}, {'':<10}, {out['core_metric']:<10.6f}\n")


# -------------- main ---------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="v1", help="checkpoint subdir under checkpoints/")
    p.add_argument("--checkpoint", default=None, help="explicit .pth path (overrides --run-name)")
    p.add_argument("--max-per-task", type=int, default=-1, help="cap per task (-1 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"[init] device={args.device}")
    tk = TokWrapper(TOKENIZER_PATH)
    print(f"[init] tokenizer vocab={tk.tk.get_vocab_size()}, eot_id={tk.eot_id}")

    config = LLMConfig()
    model = LLM(config)
    ckpt = args.checkpoint or f"checkpoints/{args.run_name}/final.pth"
    print(f"[init] loading checkpoint {ckpt}")
    state = torch.load(ckpt, map_location=args.device)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()
    max_seq_len = config.seq_len
    print(f"[init] max_seq_len={max_seq_len}, vocab={config.vocab_size}")

    out = evaluate_core(model, tk, args.device, max_seq_len, args.max_per_task)

    print("\n" + "=" * 60)
    print(f"CORE = {out['core_metric']:.4f}")
    print("=" * 60)
    for label in out["results"]:
        print(
            f"  {label:<35} acc={out['results'][label]:.4f} "
            f"centered={out['centered_results'][label]:+.4f}"
        )

    out_dir = f"checkpoints/{args.run_name}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/core_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    write_results_csv(out, f"{out_dir}/core_eval.csv")
    print(f"\n[done] wrote {out_dir}/core_eval.json + .csv")


if __name__ == "__main__":
    main()
