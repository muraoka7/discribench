#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Compute model accuracy.
"""

__author__ = "Masayasu MURAOKA"

import argparse
import json
import re
import unicodedata


def main() -> None:
    args = get_args()

    regex = r"(Answer: |答え：|回答：)? ?(Image[- ]?|画像[- ]?)?[1-4１２３４]"
    scores = []
    with open(args.input) as fi:
        for line in fi:
            datum = json.loads(line)
            output = datum["output"]
            output = unicodedata.normalize('NFKC', output)
            res = re.search(regex, output)
            if res is None:
                pred = -1
            else:
                pred = int(res.group()[-1])
            gold = datum["answer"]
            scores.append(1 if pred == gold else 0)

    acc = sum(scores) / len(scores)
    print(f"Accuracy: {acc:.2%} ({sum(scores)} / {len(scores)})")

    return


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="jsonl file of model prediction.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
