# DiscriBench

DiscriBench is a multiple-choice VQA dataset containing 1200 samples to evaluate, analyze, and diagnose the discriminative capability of Vision-Language Models (VLMs).

See [README.md](data/README.md) for dataset details.

## Setup

```shell
uv sync
# or
# pip install -r requirements.txt
```

## How to evaluate VLMs on DiscriBench

1. Generate model prediction

openai models:

```shell
export OPENAI_API_KEY=...  # set your openai key
python src/predict_openai.py
```

anthropic models:

```shell
export ANTHROPIC_API_KEY=...  # set your anthropic key
python src/predict_claude.py
```

other open VLMs:

```shell
python src/predict_vlm.py
```

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
