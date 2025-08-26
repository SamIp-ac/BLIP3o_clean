# fastapi_app_v2.py
import os
import sys
import gc
import base64
import io
import torch
import traceback
from PIL import Image
from pathlib import Path
import argparse
import asyncio
import requests

from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from pydantic import BaseModel, Field
from typing import List, Union, Literal
from fastapi.concurrency import run_in_threadpool

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None
    print("⚠️ Warning: pdf2image is not installed. PDF processing will be disabled.")

try:
    from blip3o.model.builder import load_pretrained_model
    from blip3o.utils import disable_torch_init
    from transformers import AutoProcessor
except ImportError as e:
    print(f"❌ Fatal error: Unable to import custom module: {e}")
    sys.exit(1)

MODEL = None
PROCESSOR = None
DEVICE = 'mps'


async def load_image_from_source(source: str) -> Image.Image:
    """在後台線程中異步地從 URL, Base64 或本地路徑加載圖片。"""
    def _load():
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        elif source.startswith("/") and os.path.exists(source):
            return Image.open(source)
        else:
            data = source
            if "base64," in source:
                data = source.split("base64,")[1]
            img_bytes = base64.b64decode(data)
            return Image.open(io.BytesIO(img_bytes))
    try:
        return await run_in_threadpool(_load)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from source: {str(e)[:100]}")

async def preprocess_image_in_background(
    image: Image.Image, 
    max_size: int = 1024,
    resample_method = Image.Resampling.LANCZOS
) -> Image.Image:
    """在後台線程中異步地對單張圖片進行預處理。"""
    def _preprocess():
        width, height = image.size
        if width > max_size or height > max_size:
            image.thumbnail((max_size, max_size), resample_method)
        return image.convert("RGB")
    return await run_in_threadpool(_preprocess)


def load_global_model(model_path: str):
    global MODEL, PROCESSOR
    absolute_model_path = Path(model_path).expanduser().resolve()
    if not absolute_model_path.exists():
        print(f"❌ Fatal error: The specified model path does not exist: {absolute_model_path}")
        sys.exit(1)
    print(f"⏳ Start loading the model from a local path: {absolute_model_path}")
    try:
        disable_torch_init()
        _, model_instance, _ = load_pretrained_model(str(absolute_model_path), device=DEVICE)
        
        print("⏳ Compiling the model with torch.compile... (this will add a one-time delay to the first request)")
        model_instance = torch.compile(model_instance, backend="aot_eager")
        print("✅ Model compiled successfully.")

        model_instance = model_instance.to(DEVICE)
        processor_instance = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)
        MODEL = model_instance
        PROCESSOR = processor_instance
        print(f"✅ Model and processor moved to: {DEVICE}")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

class InputText(BaseModel):
    type: Literal["text"]
    text: str

class ImageUrl(BaseModel):
    url: str

class InputImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

class Message(BaseModel):
    role: str
    content: Union[str, List[Union[InputText, InputImage]]]

class RequestPayload(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=2048, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = Field(default=False)


app = FastAPI(title="BLIP3o High-Performance API")

def run_inference(prompt: str, images: List[Image.Image]) -> str:
    """
    一個獨立的、高性能的推理函數。
    【版本3：已簡化並移除了不必要的開銷】
    """
    try:
        text_prompt_for_qwen = f"<|im_start|>user\nPicture 1:<img>{prompt}<|im_end|>\n<|im_start|>assistant\n"

        inputs = PROCESSOR(
            text=[text_prompt_for_qwen],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)

        generated_ids = MODEL.generate(**inputs, max_new_tokens=2048)

        input_token_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_token_len:]
        
        output_text = PROCESSOR.batch_decode(
            generated_ids_trimmed.to('cpu'),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    except Exception as e:
        print(f"❌ Error occurred during inference: {e}")
        traceback.print_exc()
        raise e

@app.post("/v1/chat/completions")
async def create_chat_completion(payload: RequestPayload):
    if not MODEL or not PROCESSOR:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        user_message = next((msg for msg in reversed(payload.messages) if msg.role == 'user'), None)
        if not user_message or not isinstance(user_message.content, list):
            raise HTTPException(status_code=400, detail="Invalid user message format.")
        
        prompt = next((item.text for item in user_message.content if isinstance(item, InputText)), "")
        image_sources = [item.image_url.url for item in user_message.content if isinstance(item, InputImage)]
        if not image_sources:
            raise HTTPException(status_code=400, detail="No image provided.")

        print(f"正在並行加載 {len(image_sources)} 張圖片...")
        load_tasks = [load_image_from_source(src) for src in image_sources]
        pil_images = await asyncio.gather(*load_tasks)
        print("所有圖片加載完成。")

        print(f"正在並行預處理 {len(pil_images)} 張圖片...")
        preprocess_tasks = [preprocess_image_in_background(img) for img in pil_images]
        final_images = await asyncio.gather(*preprocess_tasks)
        print("所有圖片預處理完成。")

        print("正在將推理任務提交到後台線程...")
        result_text = await run_in_threadpool(
            run_inference,
            prompt=prompt,
            images=final_images
        )
        print("推理任務完成。")
        
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result_text.strip()}}]
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in create_chat_completion: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "BLIP3o API is running." if MODEL else "BLIP3o API is starting, model not ready."}


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="BLIP3o FastAPI Server")
    parser.add_argument("model_path", type=str, help="Path to the local BLIP3o model directory.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=9998, help="Port to run the server on.")
    args = parser.parse_args()
    print("🚀 Initialing BLIP3o FastAPI Serve...")
    load_global_model(args.model_path)
    uvicorn.run(app, host=args.host, port=args.port)