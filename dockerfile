FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    git wget curl ca-certificates gnupg \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip

# Set working directory
WORKDIR /workspace
COPY . /workspace

# ---------------------------------------------------
# Step 1: Install PyTorch 2.8.0 (CUDA 12.9 build)
# ---------------------------------------------------
RUN pip install --no-cache-dir torch==2.8.0+cu129 torchvision==0.23.0+cu129 torchaudio==2.8.0+cu129 \
    --index-url https://download.pytorch.org/whl/cu129

# ---------------------------------------------------
# Step 2: Install dependencies (excluding torch/flash-attn/xformers)
# ---------------------------------------------------
# 這裡建議你準備一個 requirements.base.txt（不包含 torch / flash-attn / xformers）
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------
# Step 3: Install xformers (需要 torch 已安裝)
# ---------------------------------------------------
RUN pip install --no-cache-dir "xformers>=0.0.30"

# ---------------------------------------------------
# Step 4: Install prebuilt flash-attn wheel
# ---------------------------------------------------
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu129torch2.8-cp312-cp312-linux_x86_64.whl

# Optional flags to control flash-attn/xformers
ENV FLASH_ATTENTION_DISABLE=0
ENV XFORMERS_DISABLE_FLASH_ATTN=0
ENV XFORMERS_DISABLE_TRITON=0

# Set PYTHONPATH
ENV PYTHONPATH=/workspace

EXPOSE 9998

CMD ["python3", "gradio/fastapi_app_v4.py", "/models/blip3o"]
