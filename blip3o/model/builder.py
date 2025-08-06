import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from blip3o.model import *
from blip3o.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from blip3o.train.train import smart_tokenizer_and_embedding_resize

# Changed for mac mps
SUPPORTED_DEVICES = ["cuda", "mps", "cpu"]

def load_pretrained_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs
):
    """
    Loads a pretrained model, adapting to the specified device (cuda, mps, cpu).
    """
    if device not in SUPPORTED_DEVICES:
        warnings.warn(f"Device '{device}' not in supported list {SUPPORTED_DEVICES}. Defaulting to 'cpu'.")
        device = "cpu"

    # kwargs = {"device_map": device_map, **kwargs} # 原始寫法，但 device_map 與 .to() 有衝突風險

    if device == "cuda":
        if use_flash_attn:
             kwargs['torch_dtype'] = torch.bfloat16
        else:
             kwargs['torch_dtype'] = torch.float16

    elif device == "mps":
        kwargs['torch_dtype'] = torch.float16
        kwargs.pop('device_map', None)
        device_map = None
    else: # cpu
        kwargs['torch_dtype'] = torch.float32
        kwargs.pop('device_map', None)
        device_map = None

    if load_8bit:
        if device == "cuda":
            kwargs['load_in_8bit'] = True
            kwargs.pop('torch_dtype', None)
        else:
            warnings.warn(f"8-bit loading is not supported on device '{device}'. Loading in {kwargs.get('torch_dtype', 'default')} precision.")
            load_8bit = False
    elif load_4bit:
        if device == "cuda":
            kwargs['load_in_4bit'] = True
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            kwargs.pop('torch_dtype', None)
        else:
            warnings.warn(f"4-bit loading is not supported on device '{device}'. Loading in {kwargs.get('torch_dtype', 'default')} precision.")
            load_4bit = False

    if use_flash_attn:
        if device == "cuda":
            kwargs['attn_implementation'] = 'flash_attention_2'
        else:
            warnings.warn(f"Flash Attention 2 might not be available or optimized for device '{device}'. Proceeding without it.")
            kwargs.pop('attn_implementation', None)


    # --- 載入模型和 Tokenizer ---
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    print(f"Loading model from {model_path} with kwargs: {kwargs}...")
    model = blip3oQwenForInferenceLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **{k: v for k, v in kwargs.items() if k not in ['device_map'] or (k == 'device_map' and device == 'cuda')} # 條件性傳遞 device_map
    )
    print("Model loaded into RAM.")
    if device != "cuda" or device_map is None:
        print(f"Moving model to device: {device}...")
        model = model.to(device) # move to mps or cpu
        print(f"Model moved to device: {next(model.parameters()).device}")

    image_processor = None

    # 處理特殊 token
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer)) # 調整嵌入層大小


    # --- 確定上下文長度 ---
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len








def load_pretrained_model_lmms_eval(model_path, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'


    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = blip3oQwenForInferenceLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16)

    image_processor = None
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len

