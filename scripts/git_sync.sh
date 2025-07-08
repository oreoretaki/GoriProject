#!/bin/bash
# =========================
# GitHub双方向同期スクリプト
# プル→プッシュの完全同期
# =========================

# 設定（環境変数から読み込み）
GITHUB_USERNAME="oreoretaki"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REPO_NAME="GoriProject"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN環境変数が設定されていません"
    echo "実行例: GITHUB_TOKEN=your_token ./scripts/git_sync.sh"
    exit 1
fi

echo "🔄 GitHub双方向同期開始..."

# Step 1: プル
echo "📥 Step 1: リモートから最新を取得..."
./scripts/git_pull.sh

if [ $? -ne 0 ]; then
    echo "❌ プルに失敗しました。同期を中断します。"
    exit 1
fi

echo ""
echo "📤 Step 2: ローカル変更をプッシュ..."

# Step 2: プッシュ
./scripts/git_push.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 双方向同期完了！"
    echo "📦 リポジトリURL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
else
    echo "❌ プッシュに失敗しました"
    exit 1
fi