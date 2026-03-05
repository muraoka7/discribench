#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Make model prediction of openai's model.
"""

__author__ = "Masayasu MURAOKA"

import argparse
import base64
import json
import os
import typing as T

from openai import OpenAI

from utils import (
    DATA_TYPES,
    PROMPT_TEMPLATE,
    SUPPORTED_LANGS,
    get_response_with_backoff,
    load_data,
)


def main() -> None:
    args = get_args()

    data = load_data(args.data_type, args.lang)
    client = OpenAI()
    api_func = client.chat.completions.create

    results = predict(data, args.lang, api_func, args.model_name)

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
        default="gpt-4o-2024-08-06",
        help="model name."
    )
    parser.add_argument(
        "--out_file",
        default="outputs/outputs.jsonl",
        help="output file path to save model prediction."
    )
    args = parser.parse_args()
    return args


def make_input(datum: dict, lang: str) -> tuple[list[dict], str]:
    image_media_type = "image/jpeg"
    images = [
        base64.b64encode(
            open(image_file, "rb").read()
        ).decode("utf-8") for image_file in datum["image_files"]
    ]

    if len(images) > 1:
        image_contents = [
            {
                "type": "text",
                "text": "Image 1: "
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{images[0]}", "detail": "high"},
            },
            {
                "type": "text",
                "text": "\nImage 2: "
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{images[1]}", "detail": "high"},
            },
            {
                "type": "text",
                "text": "\nImage 3: "
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{images[2]}", "detail": "high"},
            },
            {
                "type": "text",
                "text": "\nImage 4: "
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{images[3]}", "detail": "high"},
            },
        ]
    else:
        image_contents = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{images[0]}", "detail": "high"},
            },
        ]

    prompt = PROMPT_TEMPLATE[lang].format(datum['context'], datum['question'])
    messages = [
        {
            "role": "user",
            "content": image_contents + [
                {
                    "type": "text",
                    "text": "\n" + prompt,
                }
            ]
        }
    ]
    return messages, prompt


def predict(samples: list[dict], lang: str, api_func: T.Callable, model_name: str) -> list[dict]:
    results = []
    for sample in samples:
        vlm_input, prompt = make_input(sample, lang)
        out_text, response = get_response(vlm_input, api_func, model_name)
        sample["input"] = prompt
        sample["output"] = out_text
        sample["response"] = response
        results.append(sample)
    return results


def get_response(vlm_input: list[dict], api_func: T.Callable, model_name: str) -> tuple[str, dict|None]:
    raw_response = get_response_with_backoff(
        api_func,
        model=model_name,
        max_completion_tokens=512,
        messages=vlm_input,
    )
    out_text = ""
    if raw_response is not None:
        out_text = raw_response.choices[0].message.content
        raw_response = raw_response.to_dict()
    return out_text, raw_response


if __name__ == '__main__':
    main()
