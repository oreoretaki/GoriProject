#!/bin/bash
# =========================
# vast.ai デプロイメントスクリプト
# =========================

# Docker Hub 情報（既にプッシュ済み）
DOCKER_USERNAME="oreoretaki"
IMAGE_NAME="goriproject"
TAG="latest"

echo "🚀 GoriProject vast.ai デプロイメント準備完了"
echo "✅ Docker Hub イメージ: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo "✅ GitHub リポジトリ: https://github.com/oreoretaki/GoriProject"
echo ""
echo "📝 vast.ai での使用方法:"
echo "1. vast.ai で新しいインスタンスを作成"
echo "2. Docker Image に以下を指定:"
echo "   ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo "3. GPU は RTX 4090 または A100 推奨"
echo "4. ディスクは最低 50GB 推奨"
echo ""
echo "🎯 vast.ai セットアップコマンド:"
echo "   # リポジトリクローン"
echo "   git clone https://github.com/oreoretaki/GoriProject.git /workspace/GoriProject"
echo "   cd /workspace/GoriProject"
echo ""
echo "   # T5-Large学習開始"
echo "   cd stage1"
echo "   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1"
echo ""
echo "💡 推奨スペック:"
echo "   • GPU: RTX 4090 または A100 (24GB VRAM)"
echo "   • RAM: 32GB+"
echo "   • Storage: 100GB+"