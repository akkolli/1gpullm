# One GPU LM

Small GPT-style language model training on one RTX 5090.

## Workflow

```bash
make sync
make data
make train
make sft
make chat PROMPT="Write a short recipe for tea."
make eval
make generate PROMPT="Once upon a time"
```

The same commands are also exposed through `uv`:

```bash
uv run lm-data
uv run lm-train
uv run lm-sft --base-run-name v1.9 --run-name chat-sft
uv run lm-chat --run-name chat-sft "Write a short recipe for tea."
uv run lm-eval --run-name v1.9
uv run lm-generate --run-name v1.9 "Once upon a time"
```

## Chat SFT

`lm-sft` loads a pretrained checkpoint, formats instruction/chat rows, and
trains only on assistant tokens. User, system, role-marker, and padding tokens
are masked with `ignore_index`.

Training logs include cumulative processed tokens, processed tokens/sec,
estimated dense training FLOPs/sec, and MFU. By default MFU uses the hard-coded
GeForce RTX 5090 dense Tensor Core peaks in
`GPU_PEAK_TFLOPS_BY_PRECISION` from `one_gpu_lm/training.py`: BF16 is
209.5 TFLOPS, and FP8 is 838.0 TFLOPS. Change that mapping for another GPU,
or pass `--mfu-peak-tflops` to override it for one run.

The default dataset is `HuggingFaceH4/ultrachat_200k` with `train_sft` and
`test_sft` splits:

```bash
uv run lm-sft \
  --base-run-name v1.9 \
  --run-name chat-sft \
  --epochs 1 \
  --train-steps 1000 \
  --batch-size 8 \
  --peak-lr 2e-5
```

With `make`, pass `MFU_PEAK_TFLOPS=...` only when overriding the default.

For Alpaca-style data, point `--dataset` at a dataset with
`instruction`/`input`/`output` columns. For chat data, `messages` or
`conversations` rows are accepted.

After SFT:

```bash
uv run lm-chat --run-name chat-sft "Explain Newton's second law in one sentence."
```

For a small end-to-end smoke run:

```bash
make smoke
```

## Checks

```bash
make test
make profile
```

## Layout

`one_gpu_lm/` contains reusable model, data, training, checkpoint, generation,
and CORE evaluation code. `scripts/` contains thin command-line entrypoints.
`tests/` locks down custom loss math, checkpoint round-trips, data batching, and
CORE scoring helpers.

## Artifacts

Generated datasets, tokenizer files, checkpoints, and CORE bundles are ignored.
Training writes `final.pth`, `train_state.pt`, and `metrics.json` under
`checkpoints/<run-name>/`.
