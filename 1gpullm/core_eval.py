import csv
import io
import json
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
EVAL_BUNDLE_DIR = Path("eval_bundle")
TOKENIZER_PATH = Path("tokenizer/tokenizer.json")
ANSWER = "\nAnswer: "
CR, LU, RC, SP, WK = (
    "commonsense_reasoning/",
    "language_understanding/",
    "reading_comprehension/",
    "symbolic_problem_solving/",
    "world_knowledge/",
)
LM, MC, SCHEMA = "language_modeling", "multiple_choice", "schema"


@dataclass(frozen=True)
class CoreTask:
    """Description of one CORE task and how it should be scored."""

    label: str
    dataset: str
    kind: str
    shots: int = 0
    delimiter: str = " "


TASKS = (
    CoreTask("hellaswag_zeroshot", LU + "hellaswag.jsonl", MC),
    CoreTask("jeopardy", WK + "jeopardy_all.jsonl", LM, 10, ANSWER),
    CoreTask("bigbench_qa_wikidata", WK + "bigbench_qa_wikidata.jsonl", LM, 10),
    CoreTask("arc_easy", WK + "arc_easy.jsonl", MC, 10, ANSWER),
    CoreTask("arc_challenge", WK + "arc_challenge.jsonl", MC, 10, ANSWER),
    CoreTask("copa", CR + "copa.jsonl", MC),
    CoreTask("commonsense_qa", CR + "commonsense_qa.jsonl", MC, 10),
    CoreTask("piqa", CR + "piqa.jsonl", MC, 10, ANSWER),
    CoreTask("openbook_qa", CR + "openbook_qa.jsonl", MC),
    CoreTask("lambada_openai", LU + "lambada_openai.jsonl", LM),
    CoreTask("hellaswag", LU + "hellaswag.jsonl", MC, 10),
    CoreTask("winograd", LU + "winograd_wsc.jsonl", SCHEMA),
    CoreTask("winogrande", LU + "winogrande.jsonl", SCHEMA),
    CoreTask("bigbench_dyck_languages", SP + "bigbench_dyck_languages.jsonl", LM, 10),
    CoreTask("agi_eval_lsat_ar", SP + "agi_eval_lsat_ar.jsonl", MC, 3),
    CoreTask("bigbench_cs_algorithms", SP + "bigbench_cs_algorithms.jsonl", LM, 10),
    CoreTask("bigbench_operators", SP + "bigbench_operators.jsonl", LM, 10),
    CoreTask("bigbench_repeat_copy_logic", SP + "bigbench_repeat_copy_logic.jsonl", LM, 10),
    CoreTask("squad", RC + "squad.jsonl", LM, 10),
    CoreTask("coqa", RC + "coqa.jsonl", LM),
    CoreTask("boolq", RC + "boolq.jsonl", MC, 10, ANSWER),
    CoreTask(
        "bigbench_language_identification",
        LU + "bigbench_language_identification.jsonl",
        MC,
        10,
    ),
)


class TokenizerAdapter:
    """Tiny adapter around the repo tokenizer used by CORE rendering."""

    def __init__(self, path: Path):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.eot_id = self.tokenizer.token_to_id("<|endoftext|>")
        if self.eot_id is None:
            raise RuntimeError("tokenizer is missing <|endoftext|>")

    def encode_many(
        self,
        texts: list[str],
        prepend_eot: bool = True,
    ) -> list[list[int]]:
        rows = []
        for enc in self.tokenizer.encode_batch(texts, add_special_tokens=False):
            ids = list(enc.ids)
            rows.append(([self.eot_id] if prepend_eot else []) + ids)
        return rows


@torch.inference_mode()
def model_losses_and_preds(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-position next-token losses and greedy predictions."""

    logits = model(input_ids)
    targets = torch.roll(input_ids, shifts=-1, dims=1)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(input_ids)
    losses[:, -1] = float("nan")
    return losses, logits.argmax(dim=-1)


def evaluate_core(
    model: torch.nn.Module,
    tokenizer: TokenizerAdapter,
    device: str,
    max_seq_len: int,
    max_per_task: int = -1,
) -> dict:
    """Run the CORE task set and return raw plus random-baseline centered scores."""

    ensure_bundle()
    baselines = random_baselines()
    results, centered = {}, {}
    print(f"[core] {len(TASKS)} tasks, max_per_task={max_per_task}")

    for task in TASKS:
        data = shuffled(load_jsonl(EVAL_BUNDLE_DIR / "eval_data" / task.dataset))
        if max_per_task > 0:
            data = data[:max_per_task]
        started = time.time()
        acc = evaluate_task(model, tokenizer, data, task, device, max_seq_len)
        results[task.label] = acc
        centered[task.label] = center(acc, baselines[task.label] / 100.0)
        print(
            f"[core] {task.label:<35} acc={acc:.4f} "
            f"centered={centered[task.label]:+.4f} ({time.time() - started:.1f}s)"
        )

    score = sum(centered.values()) / len(centered)
    return {"results": results, "centered_results": centered, "core_metric": score}


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: TokenizerAdapter,
    data: list[dict],
    task: CoreTask,
    device: str,
    max_seq_len: int,
) -> float:
    correct = 0
    started = time.time()
    for idx, item in enumerate(data):
        fewshot = sample_fewshot(data, idx, task.shots)
        correct += evaluate_item(model, tokenizer, item, fewshot, task, device, max_seq_len)
        if (idx + 1) % 200 == 0:
            rate = (idx + 1) / max(time.time() - started, 1e-9)
            print(
                f"    {idx + 1}/{len(data)} acc={correct / (idx + 1):.4f} "
                f"{rate:.1f}/s",
                end="\r",
            )
    return correct / len(data) if data else 0.0


def evaluate_item(
    model: torch.nn.Module,
    tokenizer: TokenizerAdapter,
    item: dict,
    fewshot: list[dict],
    task: CoreTask,
    device: str,
    max_seq_len: int,
) -> bool:
    """Score one item according to its task type."""

    if task.kind == "language_modeling":
        return evaluate_lm(model, tokenizer, item, fewshot, task, device, max_seq_len)
    if task.kind == "multiple_choice":
        prompts = render_multiple_choice(item, fewshot, task.delimiter)
        rows = tokenizer.encode_many(prompts)
        starts = [common_prefix(rows)] * len(rows)
        ends = [len(row) for row in rows]
    elif task.kind == "schema":
        prompts = render_schema(item, fewshot, task.delimiter)
        rows = tokenizer.encode_many(prompts)
        suffix = common_suffix(rows)
        ends = [len(row) for row in rows]
        starts = [end - suffix for end in ends]
    else:
        raise ValueError(f"unknown task kind: {task.kind}")

    rows, starts, ends = crop_spans(rows, starts, ends, max_seq_len)
    if all(start is None for start in starts):
        return False
    losses, _ = model_losses_and_preds(model, pad(rows, tokenizer.eot_id).to(device))
    # Multiple choice is normalized by mean continuation loss so longer choices
    # are not punished merely for having more tokens.
    choice_losses = [
        float("inf") if start is None else losses[i, start - 1 : end - 1].mean().item()
        for i, (start, end) in enumerate(zip(starts, ends))
    ]
    return choice_losses.index(min(choice_losses)) == item["gold"]


def evaluate_lm(
    model: torch.nn.Module,
    tokenizer: TokenizerAdapter,
    item: dict,
    fewshot: list[dict],
    task: CoreTask,
    device: str,
    max_seq_len: int,
) -> bool:
    """Language-modeling CORE items require exact greedy continuation tokens."""

    without, with_answer = render_language_modeling(item, fewshot, task.delimiter)
    prompt, full = tokenizer.encode_many([without, with_answer])
    start = common_prefix([prompt, full])
    if start == 0 or start >= len(full):
        return False
    if len(full) > max_seq_len:
        crop = len(full) - max_seq_len
        full = full[crop:]
        start -= crop
        if start < 1:
            return False
    _, preds = model_losses_and_preds(model, torch.tensor([full], device=device))
    return preds[0, start - 1 : len(full) - 1].tolist() == full[start:]


def render_multiple_choice(item: dict, fewshot: list[dict], delimiter: str) -> list[str]:
    prefix = "".join(f"{ex['query']}{delimiter}{ex['choices'][ex['gold']]}\n\n" for ex in fewshot)
    return [f"{prefix}{item['query']}{delimiter}{choice}" for choice in item["choices"]]


def render_schema(item: dict, fewshot: list[dict], delimiter: str) -> list[str]:
    prefix = "".join(
        f"{ex['context_options'][ex['gold']]}{delimiter}{ex['continuation']}\n\n"
        for ex in fewshot
    )
    return [f"{prefix}{ctx}{delimiter}{item['continuation']}" for ctx in item["context_options"]]


def render_language_modeling(
    item: dict,
    fewshot: list[dict],
    delimiter: str,
) -> tuple[str, str]:
    prefix = "".join(
        f"{ex['context'].strip()}{delimiter}{ex['continuation']}\n\n"
        for ex in fewshot
    )
    base = f"{prefix}{item['context'].strip()}{delimiter}"
    return base.strip(), f"{base}{item['continuation']}"


def sample_fewshot(data: list[dict], idx: int, n: int) -> list[dict]:
    if n <= 0:
        return []
    candidates = [i for i in range(len(data)) if i != idx]
    if len(candidates) < n:
        return []
    rng = random.Random(1234 + idx)
    return [data[i] for i in rng.sample(candidates, n)]


def crop_spans(
    rows: list[list[int]],
    starts: list[int],
    ends: list[int],
    max_len: int,
) -> tuple[list[list[int]], list[int | None], list[int | None]]:
    """Left-crop token rows and their scored spans to fit the context window."""

    new_rows, new_starts, new_ends = [], [], []
    for row, start, end in zip(rows, starts, ends):
        if len(row) > max_len:
            crop = len(row) - max_len
            row = row[crop:]
            start -= crop
            end -= crop
        if start < 1 or end <= start:
            start = end = None
        new_rows.append(row)
        new_starts.append(start)
        new_ends.append(end)
    return new_rows, new_starts, new_ends


def pad(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    out = torch.full((len(rows), max(map(len, rows))), pad_id, dtype=torch.long)
    for i, row in enumerate(rows):
        out[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def common_prefix(rows: list[list[int]]) -> int:
    for i, tokens in enumerate(zip(*rows)):
        if len(set(tokens)) != 1:
            return i
    return min(map(len, rows))


def common_suffix(rows: list[list[int]]) -> int:
    reversed_rows = [list(reversed(row)) for row in rows]
    return common_prefix(reversed_rows)


def center(acc: float, random_baseline: float) -> float:
    """Center task accuracy against its random baseline."""

    return (acc - random_baseline) / (1.0 - random_baseline) if random_baseline < 1.0 else 0.0


def shuffled(data: list[dict]) -> list[dict]:
    data = list(data)
    random.Random(1337).shuffle(data)
    return data


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def random_baselines() -> dict[str, float]:
    path = EVAL_BUNDLE_DIR / "eval_meta_data.csv"
    with open(path) as f:
        return {row["Eval Task"]: float(row["Random baseline"]) for row in csv.DictReader(f)}


def ensure_bundle() -> None:
    if EVAL_BUNDLE_DIR.exists():
        return
    print(f"[bundle] downloading {EVAL_BUNDLE_URL}")
    with urllib.request.urlopen(EVAL_BUNDLE_URL) as response:
        data = response.read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(".")


def write_outputs(run_name: str, output: dict) -> None:
    out_dir = Path("checkpoints") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "core_eval.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(out_dir / "core_eval.csv", "w") as f:
        f.write("Task,Accuracy,Centered\n")
        for label, acc in output["results"].items():
            f.write(f"{label},{acc:.6f},{output['centered_results'][label]:.6f}\n")
        f.write(f"CORE,,{output['core_metric']:.6f}\n")
