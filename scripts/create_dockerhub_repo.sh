#!/bin/bash
# =========================
# Docker Hub リポジトリ作成スクリプト
# =========================

# 設定
export DOCKERHUB_USER="oreoretaki"
export REPO_NAME="goriproject"

echo "🐳 Docker Hub リポジトリ作成開始..."

# Docker Hub Personal Access Token の入力を求める
echo "📝 Docker Hub Personal Access Token を入力してください:"
read -s DOCKERHUB_TOKEN

if [ -z "$DOCKERHUB_TOKEN" ]; then
    echo "❌ トークンが入力されていません"
    exit 1
fi

echo ""
echo "🔄 リポジトリ作成中..."

# Docker Hub API でリポジトリ作成
response=$(curl -s -X POST https://hub.docker.com/v2/repositories/ \
  -H "Content-Type: application/json" \
  -u "${DOCKERHUB_USER}:${DOCKERHUB_TOKEN}" \
  -d '{
        "namespace": "'"${DOCKERHUB_USER}"'",
        "name":      "'"${REPO_NAME}"'",
        "description": "GoriProject Stage1 T5-Large training environment with CUDA 11.8 and PyTorch",
        "is_private": false
      }')

# レスポンス確認
if echo "$response" | grep -q '"name"'; then
    echo "✅ Docker Hub リポジトリ作成成功！"
    echo "📦 リポジトリ: https://hub.docker.com/r/${DOCKERHUB_USER}/${REPO_NAME}"
    echo ""
    echo "🚀 次のステップ:"
    echo "   docker push ${DOCKERHUB_USER}/${REPO_NAME}:latest"
else
    echo "❌ リポジトリ作成に失敗しました"
    echo "エラー詳細: $response"
    echo ""
    echo "💡 可能な原因:"
    echo "   • リポジトリが既に存在する"
    echo "   • Personal Access Token が無効"
    echo "   • 権限が不足している"
fi