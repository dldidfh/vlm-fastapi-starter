from typing import List

from core.model import OvisModelServer, QwenModelServer
from core.prompts import (
    category_prompt,
    summarize_prompt,
    motion_recognition_prompt,
    object_detection_prompt,
)
from utils.postprocess import parse_category


def run_summary(ovis_model: OvisModelServer, qwen_model: QwenModelServer, images: List) -> str:
    category_raw = qwen_model.generate(images, category_prompt())
    category = parse_category(category_raw)
    summary = ovis_model.generate(images, summarize_prompt(category))
    return summary


def run_motion(ovis_model: OvisModelServer, images: List) -> str:
    return ovis_model.generate(images, motion_recognition_prompt())


def run_object(ovis_model: OvisModelServer, image) -> str:
    return ovis_model.generate([image], object_detection_prompt())
