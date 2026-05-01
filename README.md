# One GPU LM

Small GPT-style language model training on one RTX 5090.

## Workflow

```bash
python -m scripts.prepare_data
python -m scripts.train
python -m scripts.eval_core --run-name v1.9
python -m scripts.generate --run-name v1.9 "Once upon a time"
```

## Checks

```bash
python -m unittest discover -s tests
python -m scripts.profile_train --batch-size 32 --seq-len 1024
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
