# ─────────────────────────────────────────────────────────────
# Stage 1: Base image
# We use python:3.11-slim — a minimal Linux + Python image (~120MB)
# NOT python:3.11 (full) which is ~900MB — keep images lean in prod!
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# ─────────────────────────────────────────────────────────────
# Install system dependencies
# llama-cpp-python needs a C compiler to build from source
# ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*   # clean up to reduce image size

# ─────────────────────────────────────────────────────────────
# Install Python dependencies
# llama-cpp-python = runs GGUF quantized models efficiently on CPU
# huggingface_hub  = downloads models from HuggingFace
# fastapi + uvicorn = the web server that serves the REST API
# ─────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    llama-cpp-python==0.3.2 \
    huggingface_hub==0.25.0 \
    fastapi==0.115.0 \
    uvicorn==0.32.0 \
    pydantic==2.9.0

# ─────────────────────────────────────────────────────────────
# Copy our custom API server script into the image
# ─────────────────────────────────────────────────────────────
COPY server.py .

# ─────────────────────────────────────────────────────────────
# Expose port 8000 — this is the port our API listens on
# NOTE: This is just documentation. kubectl port-forward
#       is what actually makes it reachable from your laptop.
# ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────
# Startup command — runs when the container starts in K8s
# ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

