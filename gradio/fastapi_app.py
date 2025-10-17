# this file relative path: gradio/fastapi_app.py, root project filder name: BLIP3O_MPS2
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

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, RootModel 
from typing import List, Union, Literal
from fastapi.concurrency import run_in_threadpool

try:
    from blip3o.model.builder import load_pretrained_model
    from blip3o.utils import disable_torch_init
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
except ImportError as e:
    print(f"❌ Fatal error: Unable to import custom module: {e}")
    sys.exit(1)

MODEL = None
PROCESSOR = None
DEVICE = os.getenv("DEVICE", "mps")

def preprocess_image(
    image: Image.Image, 
    max_size: int = 1024,
    resample_method = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    檢查圖片分辨率，如果任何一邊超過 max_size，就按比例縮小圖片。
    """
    width, height = image.size
    
    if width > max_size or height > max_size:
        print(f"圖片原始分辨率 ({width}x{height}) 過高，正在縮小...")
        image.thumbnail((max_size, max_size), resample_method)
        new_width, new_height = image.size
        print(f"圖片已縮小至 ({new_width}x{new_height})。")
        
    return image

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

class InputText(BaseModel):
    type: Literal["text"]
    text: str

class ImageUrl(BaseModel):
    url: str

class InputImage(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

# content 字段直接定义为 Union 类型的列表
class UserMessageContent(BaseModel):
    content: List[Union[InputText, InputImage]]

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

def run_inference(prompt: str, images: Image.Image) -> str:
    """这是一个独立的、高性能的推理函数。"""
    try:
        message_content = []
        for img in images:
            message_content.append({"type": "image", "image": img})
        message_content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": message_content}]

        text_prompt_for_qwen = PROCESSOR.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = PROCESSOR(
            text=[text_prompt_for_qwen],
            images=image_inputs,
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
        print(f"❌Error occur during inference: {e}")
        traceback.print_exc()
        raise e
    finally:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

@app.post("/v1/chat/completions")
async def create_chat_completion(payload: RequestPayload):
    if not MODEL or not PROCESSOR:
        raise HTTPException(status_code=503, detail="Model load failed...")

    try:
        # 找到最后一个 role 为 'user' 的消息
        user_message = next((msg for msg in reversed(payload.messages) if msg.role == 'user'), None)
        
        if not user_message or not isinstance(user_message.content, list):
            raise HTTPException(status_code=400, detail="Request must contain a valid user message with list content.")

        if isinstance(user_message.content, str):
            # 如果是字串，代表這個 user message 裡沒有圖片
            prompt = user_message.content
            image_url_strings = []
        elif isinstance(user_message.content, list):
            # 如果是列表，就按原來的邏輯處理
            prompt = next((item.text for item in user_message.content if isinstance(item, InputText)), "")
            image_url_strings = [item.image_url.url for item in user_message.content if isinstance(item, InputImage)]
        else:
            raise HTTPException(status_code=400, detail="User message content has an invalid format.")

        if not image_url_strings:
            raise HTTPException(status_code=400, detail="User message must contain at least one image_url.")

        pil_images = []
        
        # 遍历所有图片来源，并逐一加载
        for source in image_url_strings:
            pil_image = None
            if source.startswith("http://") or source.startswith("https://"):
                try:
                    response = requests.get(source, stream=True, timeout=10)
                    response.raise_for_status()
                    img_bytes = response.content
                    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Fail to download image from URL {source}: {e}")
            
            # 2. 检查是否是服务器上的本地文件路径
            elif source.startswith("/") and os.path.exists(source):
                try:
                    pil_image = Image.open(source).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Fail to load image from local path {source}: {e}")
            
            # 3. 如果都不是，则假定它是 Base64 数据
            else:
                try:
                    if "base64," in source:
                        source = source.split("base64,")[1]
                    img_bytes = base64.b64decode(source)
                    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Fail to decode Base64 image data: {e}")
                    
            if pil_image:
                # 对每一张加载的图片进行预处理
                processed_image = preprocess_image(pil_image)
                pil_images.append(processed_image)

        if not pil_images:
            raise HTTPException(status_code=500, detail="No images were successfully loaded.")

        result_text = await run_in_threadpool(
        run_inference,
        prompt=prompt,
        images=pil_images
    )
        
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result_text.strip()}}]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        print(f"Error in create_chat_completion: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "BLIP3o API is running. System prompt input will be ignored in this api, and model name you can use 'blip'." if MODEL else "BLIP3o API is starting, model not ready."}


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