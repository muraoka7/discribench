#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Make model prediction of anthropic's model.
"""

__author__ = "Masayasu MURAOKA"

import argparse
import base64
import json
import os
import typing as T

import anthropic

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
    api_func = anthropic.Anthropic().messages.create

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
        default="claude-3-5-sonnet-20241022",
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
    images = []
    for image_file in datum["image_files"]:
        with open(image_file, "rb") as image_stream:
            images.append(base64.b64encode(image_stream.read()).decode("utf-8"))

    image_contents = []
    if len(images) == 1:
        image_contents.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": images[0],
                },
            }
        )
    else:
        for idx, image in enumerate(images, start=1):
            image_contents.extend(
                [
                    {
                        "type": "text",
                        "text": f"{'' if idx == 1 else chr(10)}Image {idx}: "
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image,
                        },
                    },
                ]
            )

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
        max_tokens=512,
        messages=vlm_input,
    )
    out_text = ""
    if raw_response is not None:
        out_text = raw_response.content[0].text
        raw_response = raw_response.to_dict()
    return out_text, raw_response


if __name__ == '__main__':
    main()
