# -*- coding:utf-8 -*-
"""
Utilities for open VLMs.
"""

__author__ = "Masayasu MURAOKA"

import typing as T

from PIL import Image
import torch
from transformers import AutoTokenizer
from vllm import LLM, TextPrompt

from utils import PROMPT_TEMPLATE


DiscriBenchSample = dict[str, T.Any]
MODEL_VERSION_MAP = {
    "idefics3": None,
    "internvl": ("2.5-2B", "2.5-8B", "2.5-26B", "2.5-38B", "2.5-78B"),
    "llava-ov": ("7B-chat", "72B-chat"),
    "phi3v": ("3.5",),
    "pixtral": ("12B",),
    "qwen2-vl": ("2B", "7B", "72B"),
}


class ModelRequestData(T.NamedTuple):
    llm: LLM
    stop_token_ids: list[int] | None
    prompts: list[TextPrompt]
    raw_inputs: list[DiscriBenchSample]


def load_image(img_path: str) -> Image.Image:
    with Image.open(img_path) as img:
        return img.convert("RGB")


def ensure_cuda_available(model_name: str) -> int:
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError(f"{model_name} requires a CUDA GPU, but no CUDA devices were detected.")
    return device_count


def load_idefics3(samples: list[DiscriBenchSample], model_ver: str | None, lang: str) -> ModelRequestData:
    model_name = "HuggingFaceM4/Idefics3-8B-Llama3"

    llm = LLM(
        model=model_name,
        max_model_len=97504,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 4},
        mm_processor_kwargs={
            "size": {
                "longest_edge": 1820
            },
        },
    )

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: <image>"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "<image>"

        prompt = f"""<|begin_of_text|>User:{img_str}
{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}<end_of_utterance>
Assistant:"""

        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(f) for f in sample["image_files"]]
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=None,
        prompts=prompts,
        raw_inputs=samples,
    )


def load_internvl(samples: list[DiscriBenchSample], model_ver: str | None, lang: str) -> ModelRequestData:
    bos = "<s>"
    if model_ver == "2.5-2B":
        model_name = "OpenGVLab/InternVL2_5-2B"
    elif model_ver == "2.5-8B":
        model_name = "OpenGVLab/InternVL2_5-8B"
    elif model_ver == "2.5-26B":
        model_name = "OpenGVLab/InternVL2_5-26B"
    elif model_ver == "2.5-38B":
        model_name = "OpenGVLab/InternVL2_5-38B"
        bos = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
"""
    elif model_ver == "2.5-78B":
        model_name = "OpenGVLab/InternVL2_5-78B"
        bos = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
"""
    else:
        raise NotImplementedError

    device_count = ensure_cuda_available(model_name)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 4},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=device_count,
        gpu_memory_utilization=0.95,
    )

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: <image>"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "<image>"

        prompt = f"""{bos}<|im_start|>user
{img_str}
{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}<|im_end|>
<|im_start|>assistant
"""

        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(f) for f in sample["image_files"]]
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=stop_token_ids,
        prompts=prompts,
        raw_inputs=samples,
    )


def load_llava_ov(samples: list[DiscriBenchSample], model_ver: str | None, lang: str) -> ModelRequestData:
    if model_ver == "7B-chat":
        model_name = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
    elif model_ver == "72B-chat":
        model_name = "llava-hf/llava-onevision-qwen2-72b-ov-chat-hf"
    else:
        raise NotImplementedError

    device_count = ensure_cuda_available(model_name)
    llm = LLM(
        model=model_name,
        max_model_len=16384,
        max_num_seqs=16,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 4},
        tensor_parallel_size=device_count,
        gpu_memory_utilization=0.95,
    )

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: <image>"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "<image>"

        prompt = f"""<|im_start|>user {img_str}
{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}<|im_end|><|im_start|>assistant
"""

        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(f) for f in sample["image_files"]]
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=None,
        prompts=prompts,
        raw_inputs=samples,
    )


def load_phi3v(samples: list[DiscriBenchSample], model_ver: str | None, lang: str) -> ModelRequestData:
    if model_ver == "3.5":
        model_name = "microsoft/Phi-3.5-vision-instruct"
    else:
        raise NotImplementedError

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=71232,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 4},
        mm_processor_kwargs={"num_crops": 16},
    )

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: <|image_{i}|>"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "<|image_1|>"

        prompt = f"""<|user|>
{img_str}
{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}<|end|>
<|assistant|>
"""

        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(f) for f in sample["image_files"]]
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=None,
        prompts=prompts,
        raw_inputs=samples,
    )


def load_pixtral(samples: list[DiscriBenchSample], model_ver: str | None, lang: str) -> ModelRequestData:
    if model_ver == "12B":
        model_name = "mistral-community/pixtral-12b"
        llm = LLM(
            model=model_name,
            max_model_len=32768,
            max_num_seqs=16,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 4},
        )
    else:
        raise NotImplementedError

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: [IMG]"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "[IMG]"

        prompt = f"""<s>[INST]{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}
{img_str}[/INST]"""

        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": [load_image(f) for f in sample["image_files"]]
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=None,
        prompts=prompts,
        raw_inputs=samples,
    )


def load_qwen2_vl(samples: list[DiscriBenchSample], model_ver: str, lang: str) -> ModelRequestData:
    if model_ver == "2B":
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
    elif model_ver == "7B":
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
    elif model_ver == "72B":
        model_name = "Qwen/Qwen2-VL-72B-Instruct"
    else:
        raise NotImplementedError

    device_count = ensure_cuda_available(model_name)
    llm = LLM(
        model=model_name,
        max_model_len=32768,
        max_num_seqs=16,
        limit_mm_per_prompt={"image": 4},
        tensor_parallel_size=device_count,
        gpu_memory_utilization=0.99,
    )

    prompts = []
    for sample in samples:
        if len(sample["image_files"]) > 1:
            img_str = "\n".join(
                f"Image {i}: <|vision_start|><|image_pad|><|vision_end|>"
                for i, _ in enumerate(sample["image_files"], start=1)
            )
        else:
            img_str = "<|vision_start|><|image_pad|><|vision_end|>"

        prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{img_str}
{PROMPT_TEMPLATE[lang].format(sample['context'], sample['question'])}<|im_end|>
<|im_start|>assistant
"""

        image_data = [load_image(f) for f in sample["image_files"]]
        prompts.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            }
        })

    return ModelRequestData(
        llm=llm,
        stop_token_ids=None,
        prompts=prompts,
        raw_inputs=samples,
    )


MODEL_MAP = {
    "idefics3": load_idefics3,
    "internvl": load_internvl,
    "llava-ov": load_llava_ov,
    "phi3v": load_phi3v,
    "pixtral": load_pixtral,
    "qwen2-vl": load_qwen2_vl,
}
