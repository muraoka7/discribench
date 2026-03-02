#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Make model prediction of open VLMs.
"""

__author__ = "Masayasu MURAOKA"

import argparse
from dataclasses import asdict
import json
import os
from typing import List

from vllm import SamplingParams, RequestOutput

from open_vlm_utils import (
    DiscriBenchSample,
    ModelRequestData,
    MODEL_MAP,
)
from utils import (
    DATA_TYPES,
    SUPPORTED_LANGS,
    load_data,
)


def main() -> None:
    args = get_args()

    data = load_data(args.data_type, args.lang)
    req_data = MODEL_MAP[args.model_name](data, args.model_ver, args.lang)

    results = predict(req_data)

    if args.out_file is not None:
        out_dir = os.path.dirname(os.path.abspath(args.out_file))
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as fo:
        for r in results:
            print(json.dumps(r, ensure_ascii=False), file=fo)
    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        default="discribench",
        choices=DATA_TYPES,
        help="data type to make model prediction."
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=SUPPORTED_LANGS,
        help="language of eval data."
    )
    parser.add_argument(
        "--model_name",
        default="idefics3",
        choices=MODEL_MAP.keys(),
        help="model name."
    )
    parser.add_argument(
        "--model_ver",
        default=None,
        help="Model size and version."
    )
    parser.add_argument(
        "--out_file",
        default="outputs/outputs.jsonl",
        help="output file path to save model prediction."
    )
    args = parser.parse_args()
    return args


def predict(req_data: ModelRequestData) -> list[DiscriBenchSample]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        stop_token_ids=req_data.stop_token_ids,
    )

    outputs: List[RequestOutput]
    outputs = [
        req_data.llm.generate(
            p,
            sampling_params=sampling_params,
            use_tqdm=False
        )[0]
        for p in req_data.prompts
    ]
    assert len(outputs) == len(req_data.raw_inputs)

    results = []
    for output, sample, prompt in zip(outputs, req_data.raw_inputs, req_data.prompts):
        sample["input"] = prompt["prompt"]
        sample["output"] = output.outputs[0].text
        sample["response"] = asdict(output.outputs[0])
        results.append(sample)

    return results


if __name__ == "__main__":
    main()
