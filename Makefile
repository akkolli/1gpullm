.PHONY: sync test data train sft eval generate chat profile smoke smoke-data smoke-train

RUN_NAME ?= v1.9
CHAT_RUN_NAME ?= chat-sft
SFT_DATASET ?= HuggingFaceH4/ultrachat_200k
MFU_PEAK_TFLOPS ?= 0
PROMPT ?= Once upon a time

sync:
	uv sync

test:
	uv run python -m unittest discover -s tests

data:
	uv run lm-data

train:
	uv run lm-train --run-name $(RUN_NAME) --mfu-peak-tflops $(MFU_PEAK_TFLOPS)

sft:
	uv run lm-sft --base-run-name $(RUN_NAME) --run-name $(CHAT_RUN_NAME) --dataset $(SFT_DATASET) --mfu-peak-tflops $(MFU_PEAK_TFLOPS)

eval:
	uv run lm-eval --run-name $(RUN_NAME)

generate:
	uv run lm-generate --run-name $(RUN_NAME) "$(PROMPT)"

chat:
	uv run lm-chat --run-name $(CHAT_RUN_NAME) "$(PROMPT)"

profile:
	uv run lm-profile --batch-size 32 --seq-len 1024

smoke-data:
	uv run lm-data --max-tokenizer-docs 1000 --max-train-tokens 1000000 --workers 2

smoke-train:
	uv run lm-train --run-name smoke --epochs 1 --train-steps 1 --val-steps 1 --batch-size 1 --precision bf16 --no-compile

smoke: smoke-data smoke-train
