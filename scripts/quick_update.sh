#!/bin/bash
# =========================
# クイックアップデートスクリプト
# 実験結果を即座にGitHubに同期
# =========================

# 設定（環境変数から読み込み）
GITHUB_USERNAME="oreoretaki"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REPO_NAME="GoriProject"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN環境変数が設定されていません"
    echo "実行例: GITHUB_TOKEN=your_token ./scripts/quick_update.sh"
    exit 1
fi

echo "⚡ クイックアップデート開始..."

# カスタムコミットメッセージがあるかチェック
if [ "$1" ]; then
    COMMIT_MESSAGE="$1"
else
    # デフォルトメッセージ（最新versionを自動検出）
    LATEST_VERSION=$(find stage1/logs/stage1 -name "version_*" -type d | sort -V | tail -1 | grep -o 'version_[0-9]*')
    COMMIT_MESSAGE="Update: Latest experiments and ${LATEST_VERSION} results"
fi

# リモートURLをToken付きで設定
REMOTE_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
git remote set-url origin "$REMOTE_URL" 2>/dev/null

# ワンライナー実行
git add . && git commit -m "$COMMIT_MESSAGE" && git push

if [ $? -eq 0 ]; then
    echo "✅ アップデート完了！"
    echo "📦 https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
else
    echo "⚠️  変更がないか、エラーが発生しました"
fi