import os
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration


_MODELS: "tuple[OvisModelServer, QwenModelServer] | None" = None


def set_models(ovis_model: "OvisModelServer", qwen_model: "QwenModelServer") -> None:
    global _MODELS
    _MODELS = (ovis_model, qwen_model)


def get_models() -> Tuple["OvisModelServer", "QwenModelServer"]:
    if _MODELS is None:
        raise RuntimeError("Models are not loaded. Server startup must load the models first.")
    return _MODELS


def _parse_torch_dtype(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str, torch.bfloat16)


def _parse_qwen_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    return _parse_torch_dtype(dtype_str)


def _allow_ovis_aimv2_override() -> None:
    if getattr(AutoConfig.register, "_ovis_patched", False):
        return

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "aimv2" not in CONFIG_MAPPING:
        return

    original_register = AutoConfig.register

    def _register(model_type, config, exist_ok=False):
        if model_type == "aimv2":
            exist_ok = True
        return original_register(model_type, config, exist_ok=exist_ok)

    _register._ovis_patched = True
    AutoConfig.register = _register


class OvisModelServer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: torch.dtype,
        multimodal_max_length: int,
        max_partition: int,
        max_new_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.multimodal_max_length = multimodal_max_length
        self.max_partition = max_partition
        self.max_new_tokens = max_new_tokens

        _allow_ovis_aimv2_override()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            multimodal_max_length=multimodal_max_length,
            trust_remote_code=True,
            device_map=device,
            attn_implementation="sdpa"
        )
        self.model.eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    @classmethod
    def from_env(cls) -> "OvisModelServer":
        model_id = os.getenv("OVIS_MODEL_ID", "AIDC-AI/Ovis2-8B")
        device = os.getenv("OVIS_DEVICE", "auto")
        dtype_str = os.getenv("OVIS_DTYPE", "bfloat16")
        multimodal_max_length = int(os.getenv("OVIS_MULTIMODAL_MAX_LENGTH", "32768"))
        max_partition = int(os.getenv("OVIS_MAX_PARTITION", "9"))
        max_new_tokens = int(os.getenv("OVIS_MAX_NEW_TOKENS", "256"))
        dtype = _parse_torch_dtype(dtype_str)
        return cls(
            model_id=model_id,
            device=device,
            dtype=dtype,
            multimodal_max_length=multimodal_max_length,
            max_partition=max_partition,
            max_new_tokens=max_new_tokens,
        )

    def _build_query(self, prompt: str, num_images: int) -> str:
        if num_images <= 1:
            return f"<image>\n{prompt}"
        prefix = "\n".join([f"Image {i + 1}: <image>" for i in range(num_images)])
        return f"{prefix}\n{prompt}"

    @torch.inference_mode()
    def generate(self, images: List[Image.Image], prompt: str) -> str:
        query = self._build_query(prompt, len(images))
        _, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=self.max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)

        input_ids = input_ids.to(device=self.model.device)
        attention_mask = attention_mask.to(device=self.model.device)
        pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=self.max_new_tokens,
        )

        input_len = input_ids.shape[-1]
        generated = output_ids[0][input_len:]
        return self.text_tokenizer.decode(generated, skip_special_tokens=True).strip()


class QwenModelServer:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype,
        max_new_tokens: int,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @classmethod
    def from_env(cls) -> "QwenModelServer":
        model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
        device = os.getenv("QWEN_DEVICE", "auto")
        dtype_str = os.getenv("QWEN_DTYPE", "auto")
        max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "128"))
        dtype = _parse_qwen_dtype(dtype_str)
        return cls(
            model_id=model_id,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )

    @torch.inference_mode()
    def generate(self, images: List[Image.Image], prompt: str) -> str:
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""
