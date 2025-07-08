#!/bin/bash
# =========================
# GitHub自動プルスクリプト
# =========================

# 設定（環境変数から読み込み）
GITHUB_USERNAME="oreoretaki"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
REPO_NAME="GoriProject"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN環境変数が設定されていません"
    echo "実行例: GITHUB_TOKEN=your_token ./scripts/git_pull.sh"
    exit 1
fi

echo "📥 GitHub自動プル開始..."

# リモートURLをToken付きHTTPSに設定
REMOTE_URL="https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
git remote set-url origin "$REMOTE_URL" 2>/dev/null

# 現在のブランチを確認
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 現在のブランチ: $CURRENT_BRANCH"

# ローカル変更があるかチェック
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "⚠️  ローカルに未コミットの変更があります："
    git status --porcelain
    echo ""
    echo "選択肢："
    echo "1. stash - 変更を一時保存してプル"
    echo "2. commit - 変更をコミットしてからプル"
    echo "3. force - 変更を破棄してプル"
    echo "4. cancel - プルをキャンセル"
    read -p "選択してください (1/2/3/4): " choice
    
    case $choice in
        1)
            echo "📦 変更をstash中..."
            git stash push -m "Auto stash before pull $(date '+%Y-%m-%d %H:%M:%S')"
            STASHED=true
            ;;
        2)
            echo "💾 変更をコミット中..."
            git add .
            git commit -m "Auto commit before pull: $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
        3)
            echo "🗑️  変更を破棄中..."
            git reset --hard HEAD
            git clean -fd
            ;;
        4)
            echo "❌ プルをキャンセルしました"
            exit 0
            ;;
        *)
            echo "❌ 無効な選択です"
            exit 1
            ;;
    esac
fi

# リモートの最新情報を取得
echo "🔄 リモート情報を取得中..."
git fetch origin

# プルの実行
echo "⬇️  プル中..."
if git pull origin "$CURRENT_BRANCH"; then
    echo "✅ プル成功！"
    
    # stashした場合は復元するか確認
    if [ "$STASHED" = true ]; then
        echo ""
        read -p "stashした変更を復元しますか？ (y/n): " restore_stash
        if [ "$restore_stash" = "y" ] || [ "$restore_stash" = "Y" ]; then
            echo "📤 stashを復元中..."
            if git stash pop; then
                echo "✅ stash復元成功"
            else
                echo "⚠️  stash復元でコンフリクトが発生しました"
                echo "手動で解決してください: git status"
            fi
        else
            echo "💾 stashは保持されています: git stash list"
        fi
    fi
    
    # 最新の状態を表示
    echo ""
    echo "📊 最新の状態:"
    git log --oneline -5
    
else
    echo "❌ プルに失敗しました"
    exit 1
fi

echo "🎉 完了！"