#!/bin/bash
# =========================
# GitHub自動プッシュスクリプト
# =========================

# 設定（環境変数から読み込み）
GITHUB_USERNAME="oreoretaki"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REPO_NAME="GoriProject"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN環境変数が設定されていません"
    echo "実行例: GITHUB_TOKEN=your_token ./scripts/git_push.sh"
    exit 1
fi

echo "🚀 GitHub自動プッシュ開始..."

# リモートURLをToken付きHTTPSに設定
REMOTE_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# 既存のリモートを削除してトークン付きで再設定
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

# ステージング、コミット、プッシュ
echo "📝 変更をステージング中..."
git add .

# コミットメッセージの作成
COMMIT_MESSAGE="Auto update: $(date '+%Y-%m-%d %H:%M:%S')

- Latest training logs and experiments
- Updated configurations and documentation
- TF32 optimization and Docker improvements"

echo "💾 コミット中..."
if git diff --staged --quiet; then
    echo "⚠️  変更がありません。プッシュをスキップします。"
else
    git commit -m "$COMMIT_MESSAGE"
    
    echo "☁️  GitHubにプッシュ中..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ プッシュ成功！"
        echo "📦 リポジトリURL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    else
        echo "❌ プッシュに失敗しました"
        exit 1
    fi
fi

echo "🎉 完了！"