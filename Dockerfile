# =========================
# GoriProject Dockerfile
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 必須パッケージ
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    wget curl vim \
    && rm -rf /var/lib/apt/lists/*

# pipアップグレード
RUN pip install --upgrade pip

# requirements.txtをコピーして依存関係インストール
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 作業ディレクトリ
WORKDIR /workspace

# 環境変数
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PL_DISABLE_FORK=1

# TF32最適化（RTX 30xx/40xx/A100対応）
ENV TORCH_ALLOW_TF32_OVERRIDE=1

# エントリポイント
CMD ["bash"]