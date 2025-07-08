#!/bin/bash
# =========================
# vast.ai デプロイメントスクリプト
# =========================

# Docker Hub のユーザー名を設定
DOCKER_USERNAME="${DOCKER_USERNAME:-your_dockerhub_username}"
IMAGE_NAME="goriproject"
TAG="latest"

echo "🚀 GoriProject vast.ai デプロイメント開始"

# 1. Docker イメージのビルド
echo "📦 Docker イメージをビルド中..."
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -ne 0 ]; then
    echo "❌ Docker ビルドに失敗しました"
    exit 1
fi

# 2. Docker Hub へのタグ付け
echo "🏷️  Docker Hub 用にタグ付け中..."
docker tag ${IMAGE_NAME}:${TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

# 3. Docker Hub へのプッシュ
echo "☁️  Docker Hub へプッシュ中..."
echo "注意: docker login が必要です"
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

if [ $? -ne 0 ]; then
    echo "❌ Docker Hub へのプッシュに失敗しました"
    echo "💡 ヒント: docker login を実行してください"
    exit 1
fi

echo "✅ デプロイメント準備完了！"
echo ""
echo "📝 vast.ai での使用方法:"
echo "1. vast.ai で新しいインスタンスを作成"
echo "2. Docker Image に以下を指定:"
echo "   ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo "3. GPU は RTX 4090 または A100 推奨"
echo "4. ディスクは最低 50GB 推奨"
echo ""
echo "🎯 インスタンス内での実行例:"
echo "   cd /workspace/stage1"
echo "   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1"