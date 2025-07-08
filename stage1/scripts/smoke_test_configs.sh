#!/bin/bash
# 4つのT5設定ファイルのスモークテスト
# 各設定で1バッチだけ実行して起動～終了まで確認

set -e

echo "🔍 T5設定ファイル スモークテスト開始"
echo "   目的: 全設定が正常に起動・終了することを確認"
echo ""

# 設定ファイル一覧
CONFIGS=(
    "t5_baseline_10ep"
    "t5_freeze_10ep" 
    "t5_nofreeze_10ep"
    "t5_frozen_all_10ep"
)

DATA_DIR="../data/derived"
SMOKE_TEST_LOG="smoke_test_results.log"

# ログファイル初期化
echo "Smoke Test Results - $(date)" > "${SMOKE_TEST_LOG}"
echo "=================================" >> "${SMOKE_TEST_LOG}"

SUCCESS_COUNT=0
TOTAL_COUNT=${#CONFIGS[@]}

for config in "${CONFIGS[@]}"; do
    echo "🧪 テスト中: ${config}.yaml"
    
    START_TIME=$(date +%s)
    
    # fast_dev_run で1バッチのみ実行
    if python3 scripts/train_stage1.py \
        --config "configs/${config}.yaml" \
        --data_dir "${DATA_DIR}" \
        --fast_dev_run \
        --devices 1 \
        >> "${SMOKE_TEST_LOG}" 2>&1; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "   ✅ 成功 (${DURATION}秒)"
        echo "✅ SUCCESS: ${config}.yaml (${DURATION}s)" >> "${SMOKE_TEST_LOG}"
        ((SUCCESS_COUNT++))
        
    else
        END_TIME=$(date +%s) 
        DURATION=$((END_TIME - START_TIME))
        
        echo "   ❌ 失敗 (${DURATION}秒)"
        echo "❌ FAILED: ${config}.yaml (${DURATION}s)" >> "${SMOKE_TEST_LOG}"
        echo "エラー詳細は ${SMOKE_TEST_LOG} を確認してください"
    fi
    
    echo ""
done

# 結果サマリー
echo "📊 スモークテスト結果"
echo "   成功: ${SUCCESS_COUNT}/${TOTAL_COUNT}"

if [ ${SUCCESS_COUNT} -eq ${TOTAL_COUNT} ]; then
    echo "   🎉 全設定が正常に動作します！"
    echo ""
    echo "🚀 本実験実行準備完了:"
    echo "   bash scripts/run_t5_experiments.sh"
else
    echo "   ⚠️  一部設定に問題があります"
    echo "   詳細: ${SMOKE_TEST_LOG}"
fi

echo ""
echo "📄 詳細ログ: ${SMOKE_TEST_LOG}"