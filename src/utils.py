#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Utilities.
"""

__author__ = "Masayasu MURAOKA"

import json
import os
import typing as T

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


DATA_TYPES = {
    "discribench",
    "main",
    "lan_easy",
    "lan_medium",
    "vis_easy",
    "vis_medium",
    "answer_embed",
}
PROMPT_TEMPLATE = {
    "en": """{}
Question: {} Please choose the number of the image that matches the answer to the question, given the situation and conversation. The output format is 'Answer: <image_number>
Reason: <reason>', in which the reason can be optionally added to explain your answer.""",
    "ja": """{}
質問: {} 与えられた状況文と会話文に基づいて、質問の答えとなる画像の番号を選んでください。出力形式は「答え：<画像番号>
理由：<理由>」とし、任意で回答理由も出力することができます。"""
}
SUPPORTED_LANGS = {"en", "ja"}


def load_data(data_type: str, lang: str) -> list[dict]:
    n = 200
    if data_type == "discribench":
        n = 200
        filepath = f"data/{data_type}_{lang}_{n}.jsonl"
    else:
        n = 100
        filepath = f"data/ablation/{data_type}_{lang}_{n}.jsonl"

    data = []
    with open(filepath, "r") as fi:
        for l in fi:
            d = json.loads(l)
            d["image_files"] = [os.path.join("data", i) for i in d["image_files"]]
            data.append(d)

    return data


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return status_code in {408, 409, 429} or status_code >= 500

    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


@retry(
    retry=retry_if_exception(_is_retryable_error),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(10),
)
def get_response_with_backoff(api_func: T.Callable, **kwargs: T.Any) -> T.Any:
    return api_func(**kwargs)
