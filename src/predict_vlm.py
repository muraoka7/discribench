#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Make model prediction of open VLMs.
"""

__author__ = "Masayasu MURAOKA"

import argparse
from dataclasses import asdict, is_dataclass
import json
import os
from typing import Any, List

from vllm import SamplingParams, RequestOutput

from open_vlm_utils import (
    DiscriBenchSample,
    ModelRequestData,
    MODEL_MAP,
    MODEL_VERSION_MAP,
)
from utils import (
    DATA_TYPES,
    SUPPORTED_LANGS,
    load_data,
)


def main() -> None:
    args = get_args()
    validate_args(args)

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


def validate_args(args: argparse.Namespace) -> None:
    allowed_versions = MODEL_VERSION_MAP[args.model_name]
    if allowed_versions is None:
        return

    if args.model_ver is None:
        raise ValueError(
            f"--model_ver is required for {args.model_name}. "
            f"Choose one of: {', '.join(allowed_versions)}"
        )

    if args.model_ver not in allowed_versions:
        raise ValueError(
            f"Invalid --model_ver for {args.model_name}: {args.model_ver}. "
            f"Choose one of: {', '.join(allowed_versions)}"
        )


def serialize_response(output: Any) -> dict[str, Any]:
    if is_dataclass(output):
        return asdict(output)
    if hasattr(output, "to_dict") and callable(output.to_dict):
        return output.to_dict()
    if hasattr(output, "__dict__"):
        return dict(output.__dict__)
    raise TypeError(f"Unsupported response type for serialization: {type(output).__name__}")


def predict(req_data: ModelRequestData) -> list[DiscriBenchSample]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=req_data.stop_token_ids,
    )

    outputs: List[RequestOutput]
    outputs = req_data.llm.generate(
        req_data.prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    assert len(outputs) == len(req_data.raw_inputs)

    results = []
    for output, sample, prompt in zip(outputs, req_data.raw_inputs, req_data.prompts):
        if not output.outputs:
            raise RuntimeError(f"Model generation returned no outputs for prompt: {prompt['prompt'][:200]}")
        sample["input"] = prompt["prompt"]
        sample["output"] = output.outputs[0].text
        sample["response"] = serialize_response(output.outputs[0])
        results.append(sample)

    return results


if __name__ == "__main__":
    main()
