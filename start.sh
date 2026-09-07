#!/usr/bin/env bash
set -e
export DEBIAN_FRONTEND=noninteractive

# Configuration
MODEL_DIR="/workspace/models"
MODEL_FILE="Qwen3.8-27B-UD-IQ4_XS.gguf"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
MODEL_URL="https://huggingface.co/Unsloth/Qwen3.8-27B-GGUF/resolve/main/${MODEL_FILE}"
LOG_FILE="/workspace/llama-server.log"

echo "=== 1. Installing System Dependencies & CUDA Repos ==="
apt-get update && apt-get install -y wget ca-certificates lsb-release aria2

# Add Nvidia repository
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
if [ ! -f "$KEYRING_DEB" ]; then
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -sr | tr -d '.')/x86_64/${KEYRING_DEB}"
    dpkg -i "$KEYRING_DEB"
    apt-get update
fi

# Install CUDA 12 Runtime, cuBLAS, and NCCL libraries
apt-get install -y cuda-cudart-12-8 libcublas-12-8 libnccl2

# Update dynamic library search path
echo "/usr/local/cuda-12.8/lib64" > /etc/ld.so.conf.d/cuda.conf
ldconfig

echo "=== 2. Fetching Latest llama.cpp Binary ==="
cd /tmp
LATEST=$(curl -s https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases/latest | grep tag_name | cut -d'"' -f4)
echo "Downloading release: ${LATEST}"

wget -q --show-progress "https://github.com/ai-dock/llama.cpp-cuda/releases/download/${LATEST}/llama.cpp-${LATEST}-cuda-12.8-amd64.tar.gz"
tar -xzf "llama.cpp-${LATEST}-cuda-12.8-amd64.tar.gz"

# Copy all executables and shared libraries
cp -a cuda-12.8/*.so* /usr/local/lib/ 2>/dev/null || true
cp -a cuda-12.8/llama-* /usr/local/bin/
chmod +x /usr/local/bin/llama-*

# Refresh linker cache
ldconfig

echo "Checking binary version:"
llama-server --version

echo "=== 3. Downloading Model with aria2c ==="
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading ${MODEL_FILE} using 16 parallel threads..."
    aria2c -x 16 -s 16 -k 1M \
      -d "$MODEL_DIR" \
      -o "$MODEL_FILE" \
      "$MODEL_URL"
else
    echo "Model already exists at ${MODEL_PATH}. Skipping download."
fi

echo "=== 4. Starting llama-server ==="
# Kill any existing llama-server process if running
pkill -f llama-server || true

nohup llama-server \
  -m "$MODEL_PATH" \
  --jinja \
  -ngl 99 \
  -fa on \
  -c 64000 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-effort medium \
  --reasoning-preserve \
  -s 1 \
  --host 0.0.0.0 \
  --port 11434 \
  > "$LOG_FILE" 2>&1 &

echo "=== Server Launched in Background ==="
echo "Logs are being written to $LOG_FILE"
echo "To watch log output, run: tail -f $LOG_FILE"
