"""
fastapi_app_v12.py

v12 is based on fastapi_app_v10, with optional concurrent LLM OCR workers
(configurable via --llm-workers / OCR_LLM_WORKERS). All other behavior matches v10.
"""

import os
import gc
import time
import argparse
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Runtime defaults requested by user (set only when not already provided).
# IMPORTANT: do this before importing fastapi_app_v6 so DEVICE is picked up.
DEFAULT_RUNTIME_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8",
    "INFERENCE_WARN_SECONDS": "120",
    "DISABLE_TORCH_COMPILE": "1",
    "GENERATION_MAX_TIME": "120",
    "DEVICE": "cuda",
}
for _k, _v in DEFAULT_RUNTIME_ENV.items():
    os.environ.setdefault(_k, _v)

import fastapi_app_v6 as base


logger = logging.getLogger(__name__)

# OCR tuning defaults (can be overridden via env vars)
OCR_TARGET_MIN_SIDE = int(os.getenv("OCR_TARGET_MIN_SIDE", "1400"))
OCR_MAX_UPSCALE = float(os.getenv("OCR_MAX_UPSCALE", "2.0"))
OCR_AUTOCONTRAST_CUTOFF = float(os.getenv("OCR_AUTOCONTRAST_CUTOFF", "0.5"))
OCR_SHARPNESS_FACTOR = float(os.getenv("OCR_SHARPNESS_FACTOR", "1.35"))
OCR_CONTRAST_FACTOR = float(os.getenv("OCR_CONTRAST_FACTOR", "1.08"))
OCR_UNSHARP_RADIUS = float(os.getenv("OCR_UNSHARP_RADIUS", "1.2"))
OCR_UNSHARP_PERCENT = int(os.getenv("OCR_UNSHARP_PERCENT", "140"))
OCR_UNSHARP_THRESHOLD = int(os.getenv("OCR_UNSHARP_THRESHOLD", "2"))

# LLM parallel slots (default 1 = same as v10). Set via --llm-workers or OCR_LLM_WORKERS.
OCR_LLM_WORKERS = max(1, int(os.getenv("OCR_LLM_WORKERS", "1")))
_INFERENCE_SLOT_LOCK = threading.Semaphore(OCR_LLM_WORKERS)
_BASE_RUN_INFERENCE_OPTIMIZED = base.run_inference_optimized


def preprocess_image(
    image: Image.Image,
    max_size: int = None,
    resample_method=Image.Resampling.LANCZOS
) -> Image.Image:
    """
    OCR-oriented preprocessing:
    1) Normalize orientation and colorspace
    2) Upscale small images to preserve tiny glyphs
    3) Mild contrast + sharpness enhancement
    4) Controlled downscale only when image exceeds max_size
    """
    if max_size is None:
        max_size = base.MAX_IMAGE_SIZE

    try:
        # Honor EXIF orientation and use RGB consistently
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        width, height = image.size
        min_side = min(width, height)

        # Upscale small text regions before OCR (helps short IDs/serials)
        if min_side < OCR_TARGET_MIN_SIDE:
            scale = min(OCR_TARGET_MIN_SIDE / max(min_side, 1), OCR_MAX_UPSCALE)
            if scale > 1.01:
                new_w = int(width * scale)
                new_h = int(height * scale)
                logger.info(
                    f"OCR upscale from ({width}x{height}) to ({new_w}x{new_h}), scale={scale:.2f}"
                )
                image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Gentle enhancement to recover character edges without over-amplifying noise
        image = ImageOps.autocontrast(image, cutoff=OCR_AUTOCONTRAST_CUTOFF)
        image = ImageEnhance.Contrast(image).enhance(OCR_CONTRAST_FACTOR)
        image = ImageEnhance.Sharpness(image).enhance(OCR_SHARPNESS_FACTOR)
        image = image.filter(
            ImageFilter.UnsharpMask(
                radius=OCR_UNSHARP_RADIUS,
                percent=OCR_UNSHARP_PERCENT,
                threshold=OCR_UNSHARP_THRESHOLD,
            )
        )

        # Keep memory bounded: only downscale if still too large
        width, height = image.size
        if width > max_size or height > max_size:
            logger.info(f"Image resolution ({width}x{height}) too high, resizing to max_size={max_size}...")
            image.thumbnail((max_size, max_size), resample_method)
            logger.info(f"Image resized to {image.size[0]}x{image.size[1]}.")

        return image
    except Exception as e:
        # Fall back to original image if enhancement fails
        logger.warning(f"OCR preprocess fallback due to error: {e}")
        return image


async def preprocess_image_async(image: Image.Image) -> Image.Image:
    """Async OCR preprocessing using v6 CPU executor."""
    def _preprocess():
        return preprocess_image(image)

    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(base.CPU_EXECUTOR, _preprocess)
    except Exception as e:
        logger.error(f"Image preprocessing failed in v10: {e}")
        return image


def run_inference_optimized(prompt: str, images, max_new_tokens: int = None):
    """Allow up to OCR_LLM_WORKERS concurrent inferences on the shared model."""
    if max_new_tokens is None:
        max_new_tokens = base.MAX_NEW_TOKENS

    old_lock = base.MODEL_LOCK
    base.MODEL_LOCK = _INFERENCE_SLOT_LOCK
    try:
        return _BASE_RUN_INFERENCE_OPTIMIZED(
            prompt,
            images,
            max_new_tokens=max_new_tokens,
        )
    finally:
        base.MODEL_LOCK = old_lock


@asynccontextmanager
async def lifespan_v12(_app):
    logger.info("Starting BLIP3o FastAPI Server...")
    base.INFERENCE_SEMAPHORE = asyncio.Semaphore(OCR_LLM_WORKERS)
    logger.info(f"Initialized inference semaphore (concurrency limit: {OCR_LLM_WORKERS})")
    yield
    logger.info("Shutting down BLIP3o FastAPI Server...")
    base.cleanup_memory()
    if base.CPU_EXECUTOR:
        base.CPU_EXECUTOR.shutdown(wait=True)
        logger.info("CPU executor shutdown completed")


# Set after model load; read by /health without acquiring MODEL_LOCK during OCR.
_SERVICE_READY = False


def _inference_busy() -> bool:
    """True when an OCR request holds the inference semaphore."""
    sem = base.INFERENCE_SEMAPHORE
    if sem is None:
        return False
    return getattr(sem, "_value", 1) == 0


def _wrap_load_global_model():
    original = base.load_global_model_optimized

    def wrapped(model_path: str):
        global _SERVICE_READY
        _SERVICE_READY = False
        try:
            original(model_path)
            _SERVICE_READY = base.MODEL is not None and base.PROCESSOR is not None
        except Exception:
            _SERVICE_READY = False
            raise

    return wrapped


async def health_check_nonblocking():
    """
    Non-blocking health endpoint.
    v6 /health used MODEL_LOCK, which is held for entire OCR inference and blocked probes.
    """
    stats = base.SystemStats()
    model_loaded = _SERVICE_READY or (base.MODEL is not None and base.PROCESSOR is not None)
    processor_loaded = model_loaded
    busy = _inference_busy()

    if not model_loaded:
        status = "loading"
    elif busy:
        status = "busy"
    else:
        status = "healthy"

    return {
        "status": status,
        "device": base.DEVICE,
        "memory_usage": stats.get_memory_usage(),
        "gpu_memory_usage": stats.get_gpu_memory_usage(),
        "model_loaded": model_loaded,
        "processor_loaded": processor_loaded,
        "inference_busy": busy,
    }


def _replace_health_route():
    base.app.router.routes = [
        route
        for route in base.app.router.routes
        if not (getattr(route, "path", None) == "/health")
    ]
    base.app.add_api_route("/health", health_check_nonblocking, methods=["GET"])


# Monkey patch v6 to reuse all API/business logic with improved preprocessing
base.preprocess_image = preprocess_image
base.preprocess_image_async = preprocess_image_async
base.run_inference_optimized = run_inference_optimized
base.load_global_model_optimized = _wrap_load_global_model()
base.app.router.lifespan_context = lifespan_v12
_replace_health_route()
base.app.title = "BLIP3o OCR API v12.0"
base.app.description = "v10-based API with configurable concurrent LLM OCR workers"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BLIP3o OCR FastAPI Server v12")
    parser.add_argument("model_path", type=str, help="Path to the local BLIP3o model directory.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--port", type=int, default=9998, help="Port to run the server on.")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=int(os.getenv("MAX_IMAGE_SIZE", "1536")),
        help="Maximum image size for preprocessing."
    )
    parser.add_argument("--chunk-size", type=int, default=base.MAX_CHUNK_SIZE, help="Maximum images per chunk.")
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Number of CPU worker threads for image processing (default: auto-detect based on CPU cores)."
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=int(os.getenv("OCR_LLM_WORKERS", "1")),
        help="Max concurrent LLM OCR tasks (default: 1, same as v10). Use 2+ for parallel.",
    )
    args = parser.parse_args()

    llm_workers = max(1, args.llm_workers)
    globals()["OCR_LLM_WORKERS"] = llm_workers
    globals()["_INFERENCE_SLOT_LOCK"] = threading.Semaphore(llm_workers)

    # Initialize CPU executor similarly to v8
    if args.cpu_workers is None:
        cpu_count = os.cpu_count() or 4
        optimal_workers = max(2, min(cpu_count * 3 // 4, 8))
        args.cpu_workers = optimal_workers
        logger.info(f"Auto-detected {cpu_count} CPU cores, using {optimal_workers} worker threads")
    else:
        logger.info(f"Using manually configured {args.cpu_workers} CPU worker threads")

    base.CPU_EXECUTOR = ThreadPoolExecutor(max_workers=args.cpu_workers, thread_name_prefix="cpu_worker")
    logger.info(f"CPU executor initialized with {args.cpu_workers} worker threads")

    # Update v6 globals to make patched preprocess use latest runtime settings
    base.MODEL_PATH = args.model_path
    base.MAX_IMAGE_SIZE = args.max_image_size
    base.MAX_CHUNK_SIZE = args.chunk_size

    logger.info("Initializing BLIP3o OCR FastAPI Server v12...")
    logger.info(f"Device: {base.DEVICE}")
    logger.info(f"Max image size: {base.MAX_IMAGE_SIZE}")
    logger.info(f"Chunk size: {base.MAX_CHUNK_SIZE}")
    logger.info(f"CPU workers: {args.cpu_workers}")
    logger.info(f"LLM workers (concurrent OCR): {llm_workers}")
    logger.info(
        "Runtime defaults (effective): PYTORCH_CUDA_ALLOC_CONF=%s, INFERENCE_WARN_SECONDS=%s, "
        "DISABLE_TORCH_COMPILE=%s, GENERATION_MAX_TIME=%s, DEVICE=%s",
        os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        os.getenv("INFERENCE_WARN_SECONDS"),
        os.getenv("DISABLE_TORCH_COMPILE"),
        os.getenv("GENERATION_MAX_TIME"),
        os.getenv("DEVICE"),
    )
    logger.info(
        "OCR tuning: min_side=%s, max_upscale=%s, sharpness=%s, contrast=%s",
        OCR_TARGET_MIN_SIDE,
        OCR_MAX_UPSCALE,
        OCR_SHARPNESS_FACTOR,
        OCR_CONTRAST_FACTOR,
    )

    base.load_global_model_optimized(args.model_path)

    import uvicorn
    uvicorn.run(
        base.app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=True
    )
