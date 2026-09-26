FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOCK_MODE=true

WORKDIR /app

# 1. Install system tools, ffmpeg (for audio decoding), and Python 3.12 from PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    cmake \
    ffmpeg \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*


# 2. Configure build flags for Turing architecture (GTX 1660 Ti = compute 75)
ENV CMAKE_ARGS="-DGGML_CUDA=on" \
    CUDA_DOCKER_ARCH=75

COPY requirements.txt .

# 3. Install Python dependencies
RUN pip install --no-cache-dir --no-binary llama-cpp-python -r requirements.txt

# 4. Link NVIDIA pip-installed cuBLAS/cuDNN libraries so faster-whisper can find them
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Copy application source code
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]