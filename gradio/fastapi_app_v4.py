# fastapi_app_v3.py - Optimized BLIP3o FastAPI Server for Long Document OCR
# This version addresses memory leaks, high CPU usage, and performance issues
# for long document OCR tasks with proper resource management and chunking

import os
import sys
import gc
import base64
import io
import psutil
import asyncio
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Literal, Optional, Dict, Any
from pathlib import Path
import argparse
import traceback
import warnings
from contextlib import asynccontextmanager

import torch
import requests
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, File, UploadFile, Response, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Global variables for model management
MODEL = None
PROCESSOR = None
TOKENIZER = None
MODEL_LOCK = threading.Lock()

# Configuration constants
DEVICE = os.getenv("DEVICE", "mps")
MODEL_PATH = None
MAX_IMAGE_SIZE = 1024
MAX_CHUNK_SIZE = 3  # Maximum number of images to process at once
MEMORY_THRESHOLD = 0.85  # Clean up if memory usage exceeds 85%
MAX_TOKENS = 4096  # Increased default for longer OCR content
MAX_NEW_TOKENS = 2048  # Increased default for longer responses

# Thread pool for CPU-intensive tasks - optimized for high CPU utilization
# Will be initialized in main function

class SystemStats:
    """Monitor system resources"""
    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        return self.process.memory_percent()

    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage if available"""
        if torch.backends.mps.is_available() and DEVICE == "mps":
            try:
                # MPS doesn't have direct memory monitoring like CUDA
                return self.get_memory_usage()  # Use system memory as proxy
            except:
                return 0.0
        return 0.0

    def should_cleanup(self) -> bool:
        """Check if cleanup is needed based on memory usage"""
        return self.get_memory_usage() > MEMORY_THRESHOLD

def cleanup_memory():
    """Aggressive memory cleanup"""
    try:
        # Force garbage collection
        gc.collect()

        # Clear MPS cache if available
        if torch.backends.mps.is_available() and DEVICE == "mps":
            torch.mps.empty_cache()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

def preprocess_image_optimized(
    image: Image.Image,
    max_size: int = MAX_IMAGE_SIZE,
    quality: int = 85
) -> Image.Image:
    """
    Optimized image preprocessing with better memory management
    """
    try:
        # Convert to RGB early to avoid format issues
        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size

        # Only resize if necessary
        if width > max_size or height > max_size:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)

            # Use high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Optimize for memory usage
        image = ImageOps.exif_transpose(image)

        return image

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return original image as fallback
        return image.convert('RGB') if image.mode != 'RGB' else image

async def load_image_from_source_optimized(source: str) -> Image.Image:
    """Optimized async image loading with better error handling"""
    def _load():
        try:
            if source.startswith("http://") or source.startswith("https://"):
                response = requests.get(source, stream=True, timeout=30)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))

            elif source.startswith("/") and os.path.exists(source):
                return Image.open(source)

            else:
                # Assume base64
                data = source
                if "base64," in source:
                    data = source.split("base64,")[1]
                img_bytes = base64.b64decode(data)
                return Image.open(io.BytesIO(img_bytes))

        except Exception as e:
            logger.error(f"Failed to load image from {source[:50]}...: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)[:100]}")

    try:
        # Use the global CPU executor for parallel processing
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(CPU_EXECUTOR, _load)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image loading failed: {str(e)[:100]}")

async def preprocess_image_async(image: Image.Image) -> Image.Image:
    """Async image preprocessing"""
    def _preprocess():
        return preprocess_image_optimized(image)

    try:
        # Use the global CPU executor for parallel processing
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(CPU_EXECUTOR, _preprocess)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return image

def load_global_model_optimized(model_path: str):
    """Load model with better memory management"""
    global MODEL, PROCESSOR, TOKENIZER

    absolute_model_path = Path(model_path).expanduser().resolve()
    if not absolute_model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {absolute_model_path}")

    logger.info(f"Loading model from: {absolute_model_path}")

    try:
        from blip3o.model.builder import load_pretrained_model
        from blip3o.utils import disable_torch_init
        from transformers import AutoProcessor

        disable_torch_init()

        # Load with optimized settings
        tokenizer, model_instance, context_len = load_pretrained_model(
            str(absolute_model_path),
            device=DEVICE,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
        )

        # Compile model for better performance
        logger.info("Compiling model...")
        model_instance = torch.compile(model_instance, backend="aot_eager")
        logger.info("Model compiled successfully")

        model_instance = model_instance.to(DEVICE)
        processor_instance = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True
        )

        with MODEL_LOCK:
            MODEL = model_instance
            PROCESSOR = processor_instance
            TOKENIZER = tokenizer

        logger.info(f"Model loaded successfully on {DEVICE}")

        # Initial cleanup
        cleanup_memory()

    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        traceback.print_exc()
        raise

def chunk_document(images: List[Image.Image], max_images_per_chunk: int = MAX_CHUNK_SIZE) -> List[List[Image.Image]]:
    """Split document into manageable chunks"""
    if len(images) <= max_images_per_chunk:
        return [images]

    chunks = []
    for i in range(0, len(images), max_images_per_chunk):
        chunk = images[i:i + max_images_per_chunk]
        chunks.append(chunk)

    logger.info(f"Document split into {len(chunks)} chunks")
    return chunks

def run_inference_optimized(prompt: str, images: List[Image.Image], max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Optimized inference with better memory management and thread safety"""
    with MODEL_LOCK:  # Protect against concurrent access to global MODEL and PROCESSOR
        try:
            # Prepare message content - CORRECTED: Include actual image objects
            message_content = []
            for img in images:
                message_content.append({"type": "image", "image": img})
            message_content.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": message_content}]

            # Apply chat template
            text_prompt = PROCESSOR.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process vision info
            from qwen_vl_utils import process_vision_info
            image_inputs, _ = process_vision_info(messages)

            # Prepare inputs with memory optimization
            inputs = PROCESSOR(
                text=[text_prompt],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = inputs.to(DEVICE)

            # Generate with optimized settings
            with torch.no_grad():
                generated_ids = MODEL.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # Use dynamic max_new_tokens from request
                    do_sample=False,  # Deterministic for OCR tasks
                    pad_token_id=PROCESSOR.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Extract response
            input_token_len = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_token_len:]

            output_text = PROCESSOR.batch_decode(
                generated_ids_trimmed.to('cpu'),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return output_text.strip()

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            traceback.print_exc()
            raise
        finally:
            # Clean up tensors
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'generated_ids_trimmed' in locals():
                del generated_ids_trimmed

# Pydantic models
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

class OCRRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: int = Field(default=MAX_TOKENS, ge=1)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)  # Lower for OCR
    stream: bool = Field(default=False)
    chunk_size: int = Field(default=MAX_CHUNK_SIZE, ge=1, le=10)

class OCRResponse(BaseModel):
    content: str
    chunk_index: int
    total_chunks: int
    memory_usage: float

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting BLIP3o FastAPI Server...")
    yield
    # Shutdown
    logger.info("Shutting down BLIP3o FastAPI Server...")
    cleanup_memory()
    global CPU_EXECUTOR
    if CPU_EXECUTOR:
        CPU_EXECUTOR.shutdown(wait=True)
        logger.info("CPU executor shutdown completed")

app = FastAPI(
    title="BLIP3o OCR API v4",
    description="Optimized API for long document OCR with memory management",
    lifespan=lifespan
)

async def process_ocr_chunk(
    prompt: str,
    images: List[Image.Image],
    chunk_index: int,
    total_chunks: int,
    stats: SystemStats,
    max_new_tokens: int = MAX_NEW_TOKENS
) -> OCRResponse:
    """Process a single chunk of the OCR task"""
    try:
        logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks}")

        # Check memory before processing
        if stats.should_cleanup():
            cleanup_memory()

        # Run inference in thread pool
        result_text = await run_in_threadpool(
            run_inference_optimized,
            prompt=prompt,
            images=images,
            max_new_tokens=max_new_tokens
        )

        # Clean up images after processing
        for img in images:
            del img
        del images

        return OCRResponse(
            content=result_text,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            memory_usage=stats.get_memory_usage()
        )

    except Exception as e:
        logger.error(f"Chunk {chunk_index} processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunk processing failed: {str(e)[:100]}")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: OCRRequest, background_tasks: BackgroundTasks):
    """Optimized OCR endpoint with chunking and memory management"""
    global MODEL, PROCESSOR

    # Thread-safe check for model availability
    with MODEL_LOCK:
        model_ready = MODEL is not None and PROCESSOR is not None

    if not model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        # Extract user message
        user_message = next((msg for msg in reversed(request.messages) if msg.role == 'user'), None)
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Parse content - handle both old and new request formats
        if isinstance(user_message.content, str):
            # Handle string format (legacy)
            prompt = user_message.content
            image_sources = []
        elif isinstance(user_message.content, list):
            # Handle new structured format
            prompt = next((item.text for item in user_message.content if isinstance(item, InputText)), "")
            image_sources = []
            for item in user_message.content:
                if isinstance(item, InputImage):
                    image_sources.append(item.image_url.url)
                # Handle legacy format where image URLs might be direct strings
                elif isinstance(item, dict) and item.get("type") == "image_url":
                    image_sources.append(item.get("url", ""))
        else:
            raise HTTPException(status_code=400, detail="User message content has an invalid format.")

        if not image_sources:
            raise HTTPException(status_code=400, detail="No images provided for OCR")

        # Load and preprocess images with progress tracking and error handling
        logger.info(f"Loading {len(image_sources)} images...")
        load_tasks = [load_image_from_source_optimized(src) for src in image_sources]
        try:
            pil_images = await asyncio.gather(*load_tasks)
        except Exception as e:
            logger.error(f"Some images failed to load: {e}")
            # Continue with successfully loaded images
            pil_images = [img for img in await asyncio.gather(*load_tasks, return_exceptions=True) if not isinstance(img, Exception)]

        if not pil_images:
            raise HTTPException(status_code=400, detail="No images could be loaded successfully")

        logger.info(f"Successfully loaded {len(pil_images)} images, preprocessing...")
        preprocess_tasks = [preprocess_image_async(img) for img in pil_images]
        try:
            processed_images = await asyncio.gather(*preprocess_tasks)
        except Exception as e:
            logger.error(f"Some images failed to preprocess: {e}")
            # Continue with successfully processed images
            processed_images = [img for img in await asyncio.gather(*preprocess_tasks, return_exceptions=True) if not isinstance(img, Exception)]

        if not processed_images:
            raise HTTPException(status_code=500, detail="No images could be processed successfully")

        # Clean up original images
        for img in pil_images:
            del img
        del pil_images

        # Chunk the document
        image_chunks = chunk_document(processed_images, request.chunk_size)

        # Initialize system stats
        stats = SystemStats()

        # Process chunks
        if request.stream:
            # Stream response for large documents
            async def generate_response():
                all_content = []

                for i, chunk in enumerate(image_chunks):
                    # Add context about chunk position
                    chunk_prompt = f"{prompt}\n\n[Processing page {i+1} of {len(image_chunks)}]"

                    response = await process_ocr_chunk(
                        chunk_prompt,
                        chunk,
                        i,
                        len(image_chunks),
                        stats,
                        max_new_tokens=request.max_tokens
                    )

                    all_content.append(response.content)

                    # Yield intermediate result
                    yield f"data: {response.model_dump_json()}\n\n"

                    # Cleanup after each chunk
                    cleanup_memory()

                    # Small delay to prevent overwhelming
                    await asyncio.sleep(0.1)

                # Final merged result
                final_content = "\n".join(all_content)
                final_response = OCRResponse(
                    content=final_content,
                    chunk_index=len(image_chunks),
                    total_chunks=len(image_chunks),
                    memory_usage=stats.get_memory_usage()
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                content=generate_response(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # Process all chunks and return combined result
            all_results = []

            for i, chunk in enumerate(image_chunks):
                chunk_prompt = f"{prompt}\n\n[Processing page {i+1} of {len(image_chunks)}]"

                response = await process_ocr_chunk(
                    chunk_prompt,
                    chunk,
                    i,
                    len(image_chunks),
                    stats,
                    max_new_tokens=request.max_tokens
                )

                all_results.append(response.content)

                # Cleanup after each chunk
                cleanup_memory()

            # Combine results
            final_content = "\n".join(all_results)

            return {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_content.strip()
                    }
                }],
                "usage": {
                    "total_chunks": len(image_chunks),
                    "memory_usage": stats.get_memory_usage()
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)[:100]}")
    finally:
        # Final cleanup
        background_tasks.add_task(cleanup_memory)

@app.get("/health")
async def health_check():
    """Health check endpoint with system stats"""
    stats = SystemStats()

    # Thread-safe model status check
    with MODEL_LOCK:
        model_loaded = MODEL is not None
        processor_loaded = PROCESSOR is not None

    return {
        "status": "healthy" if model_loaded and processor_loaded else "loading",
        "device": DEVICE,
        "memory_usage": stats.get_memory_usage(),
        "gpu_memory_usage": stats.get_gpu_memory_usage(),
        "model_loaded": model_loaded,
        "processor_loaded": processor_loaded
    }

@app.post("/cleanup")
async def manual_cleanup():
    """Manual memory cleanup endpoint"""
    cleanup_memory()
    stats = SystemStats()

    return {
        "message": "Memory cleanup completed",
        "memory_usage": stats.get_memory_usage()
    }

@app.get("/")
def read_root():
    # Thread-safe model status check
    with MODEL_LOCK:
        model_ready = MODEL is not None

    return {
        "status": "BLIP3o OCR API v4 is running" if model_ready else "BLIP3o OCR API v4 is starting",
        "version": "3.0",
        "features": [
            "Memory management",
            "Document chunking",
            "Async image processing",
            "Streaming responses",
            "Resource monitoring"
        ],
        "Remarks": "Set llm url to 9998 port will auto use blip model, you can use 'blip' as model name, this model do not read system prompt, just use \"\" as system prompt"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP3o OCR FastAPI Server v3")
    parser.add_argument("model_path", type=str, help="Path to the local BLIP3o model directory.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=9998, help="Port to run the server on.")
    parser.add_argument("--max-image-size", type=int, default=MAX_IMAGE_SIZE, help="Maximum image size for preprocessing.")
    parser.add_argument("--chunk-size", type=int, default=MAX_CHUNK_SIZE, help="Maximum images per chunk.")
    parser.add_argument("--cpu-workers", type=int, default=None, help="Number of CPU worker threads for image processing (default: auto-detect based on CPU cores).")

    args = parser.parse_args()

    # Initialize CPU executor with optimal worker count
    if args.cpu_workers is None:
        # Auto-detect optimal worker count based on CPU cores
        cpu_count = os.cpu_count() or 4  # Fallback to 4 if detection fails
        # Use 75% of CPU cores, with a minimum of 2 and maximum of 8
        optimal_workers = max(2, min(cpu_count * 3 // 4, 8))
        args.cpu_workers = optimal_workers
        logger.info(f"Auto-detected {cpu_count} CPU cores, using {optimal_workers} worker threads")
    else:
        logger.info(f"Using manually configured {args.cpu_workers} CPU worker threads")

    # Initialize the CPU executor
    global CPU_EXECUTOR
    CPU_EXECUTOR = ThreadPoolExecutor(max_workers=args.cpu_workers, thread_name_prefix="cpu_worker")
    logger.info(f"CPU executor initialized with {args.cpu_workers} worker threads")

    # Update global config
    MODEL_PATH = args.model_path
    MAX_IMAGE_SIZE = args.max_image_size
    MAX_CHUNK_SIZE = args.chunk_size

    logger.info("🚀 Initializing BLIP3o OCR FastAPI Server v3...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Max image size: {MAX_IMAGE_SIZE}")
    logger.info(f"Chunk size: {MAX_CHUNK_SIZE}")
    logger.info(f"CPU workers: {args.cpu_workers}")

    # Load model
    load_global_model_optimized(args.model_path)

    # Start server
    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
