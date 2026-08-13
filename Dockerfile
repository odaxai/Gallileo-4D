# Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction
# 3rd Place at PhysAI Dynamic 4D Reconstruction Challenge (ECCV 2026)
# Winning submission: opt4_r728_116.csv — 0.58356 private / 0.55513 public
#
# DEFAULT BEHAVIOUR (docker run, no arguments):
#   Reproduces the EXACT winning submission bit-for-bit from the shipped
#   components and verifies it cryptographically (MD5 + SHA256).
#
#   Expected MD5:    24d729dedd16de4f65e2d67301455c48
#   Expected SHA256: 9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="OdaxAI Research <research@odaxai.com>"
LABEL description="Gallileo-4D: exact reproduction of the 0.58356 APD winning submission"
LABEL version="2.0"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy \
    pandas \
    pillow \
    opencv-python-headless \
    transformers \
    huggingface_hub \
    pytest

# Copy application code (includes reference/components/*.csv.gz)
COPY . .

# Create directories
RUN mkdir -p /app/data /app/results /app/checkpoints /app/external/4RC

# Set environment variables
ENV PYTHONPATH="/app:/app/external/4RC"
ENV CHECKPOINT_DIR="/app/checkpoints"
ENV DATA_DIR="/app/data"
ENV RESULTS_DIR="/app/results"

# NOTE: 4RC must be mounted at runtime only if you want to re-run GPU
# inference from scratch. It is NOT needed for the exact reproduction.
# Mount with: -v /path/to/4RC:/app/external/4RC

# Sanity checks at build time
RUN python -c "import gallileo4d; print('Gallileo-4D package OK')" && \
    python -c "import numpy, pandas, torch; print(f'PyTorch {torch.__version__}, NumPy {numpy.__version__}, Pandas {pandas.__version__}')" && \
    ls -la reference/components/

# Default command: reproduce the winning submission bit-for-bit and
# verify MD5 + SHA256. Output written to /app/results/submission.csv
# (mount /app/results to keep it).
CMD ["bash", "scripts/reproduce_winning.sh", "reference/components", "/app/results/submission.csv"]
