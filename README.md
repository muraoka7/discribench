# DiscriBench

DiscriBench is a multiple-choice VQA dataset containing 1200 samples to evaluate, analyze, and diagnose the discriminative capability of Vision-Language Models (VLMs).

See [README.md](data/README.md) for dataset details.

## Setup

```shell
uv sync && source .venv/bin/activate
# or
# pip install -r requirements.txt
```

- Python 3.11 is recommended. This project currently requires Python >= 3.11.11.
- `uv` is the recommended setup method because the repository is managed with `pyproject.toml` and `uv.lock`.

## How to evaluate VLMs on DiscriBench

1. Generate model prediction

Predictions are saved to `outputs/outputs.jsonl` by default. You can change this with `--out_file`.

openai models:

```shell
export OPENAI_API_KEY=...  # set your openai key
python src/predict_openai.py \
  --model_name gpt-4o-2024-08-06 \
  --lang en \
  --data_type discribench
```

Main options:
- `--model_name`: OpenAI model name
- `--lang`: `en` or `ja`
- `--data_type`: `discribench`, `main`, `lan_easy`, `lan_medium`, `vis_easy`, `vis_medium`, `answer_embed`
- `--out_file`: output JSONL path

anthropic models:

```shell
export ANTHROPIC_API_KEY=...  # set your anthropic key
python src/predict_claude.py \
  --model_name claude-3-5-sonnet-20241022 \
  --lang ja \
  --data_type discribench
```

Main options:
- `--model_name`: Anthropic model name
- `--lang`: `en` or `ja`
- `--data_type`: `discribench`, `main`, `lan_easy`, `lan_medium`, `vis_easy`, `vis_medium`, `answer_embed`
- `--out_file`: output JSONL path

other open VLMs:

```shell
python src/predict_vlm.py \
  --model_name idefics3 \
  --lang en \
  --data_type discribench
```

Open VLM requirements:
- Open VLM inference uses `vllm`.
- This path is intended for CUDA GPU environments, not CPU-only environments.
- Some models require `--model_ver`.

Examples:

```shell
# idefics3: --model_ver is not required
python src/predict_vlm.py --model_name idefics3 --lang en

# qwen2-vl: --model_ver is required
python src/predict_vlm.py --model_name qwen2-vl --model_ver 7B --lang en
```

Supported `--model_name` / `--model_ver` combinations:
- `idefics3`: no `--model_ver`
- `internvl`: `2.5-2B`, `2.5-8B`, `2.5-26B`, `2.5-38B`, `2.5-78B`
- `llava-ov`: `7B-chat`, `72B-chat`
- `phi3v`: `3.5`
- `pixtral`: `12B`
- `qwen2-vl`: `2B`, `7B`, `72B`

Main options:
- `--model_name`: open VLM family
- `--model_ver`: model size/version for models that require it
- `--lang`: `en` or `ja`
- `--data_type`: `discribench`, `main`, `lan_easy`, `lan_medium`, `vis_easy`, `vis_medium`, `answer_embed`
- `--out_file`: output JSONL path

2. Compute accuracy

```shell
python src/score.py outputs/outputs.jsonl
```

## Licenses

This repository applies multiple licenses to different materials. See README.md for each directory.
The summary of license information is provided below:

- [`src/`](src/README.md): Apache 2.0
- [`data/`](data/README.md):
    - For the Exam portion derived from the English listening subject in the Common Test for University Admissions, **all rights reserved** by its original creator, the National Center for University Entrance Examinations (https://www.dnc.ac.jp/).
    - Images in the COCO portion have their own licenses.
    - For the rest part that is created in this work, Apache-2.0 applies.

## Citation

```bibtex
@inproceedings{muraoka:LREC2026:discribench,
    title = "Evaluating Discriminability of Vision-Language Models",
    author = "Muraoka, Masayasu and Okazaki, Naoaki",
    booktitle = "The Fifteenth Language Resources and Evaluation Conference",
    month = May,
    year = "2026",
    address = "Palma, Mallorca, Spain",
    url = "To appear"
}
```
