#!/bin/bash
# T5転移学習 vs ランダム初期化 比較実験自動実行スクリプト

# Note: set -e removed to continue experiments even if one fails

# 実験設定
DATA_DIR="../data/derived"
RESULTS_DIR="./t5_experiment_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 結果ディレクトリ作成
mkdir -p "${RESULTS_DIR}/${TIMESTAMP}"

echo "🚀 T5転移学習比較実験開始"
echo "   タイムスタンプ: ${TIMESTAMP}"
echo "   結果保存先: ${RESULTS_DIR}/${TIMESTAMP}"
echo "   データディレクトリ: ${DATA_DIR}"
echo ""

# T5モデルの事前ダウンロード確認
echo "🤗 T5モデルの確認中..."
if [ -d "/home/taki/.cache/huggingface/hub/models--t5-base" ]; then
    echo "✅ T5モデルが利用可能です"
else
    echo "❌ T5モデルが見つかりません"
    echo "   解決方法: python3 scripts/download_t5.py --model t5-base"
    exit 1
fi


echo ""

# 実験の配列定義（10エポック版）
declare -a EXPERIMENTS=(
    "t5_freeze_10ep:T5凍結2エポック→解凍(10ep)"
    "t5_nofreeze_10ep:T5凍結なし(10ep)"
    "t5_frozen_all_10ep:T5完全凍結(10ep)"
    "t5_baseline_10ep:ベースライン(10ep)"
)

# 各実験を順次実行
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name description <<< "$experiment"
    
    echo "📊 実験開始: ${description}"
    echo "   設定ファイル: configs/${config_name}.yaml"
    
    # 実験固有のログディレクトリ
    EXPERIMENT_LOG_DIR="${RESULTS_DIR}/${TIMESTAMP}/${config_name}"
    mkdir -p "${EXPERIMENT_LOG_DIR}"
    
    # 開始時刻記録
    START_TIME=$(date +%s)
    
    # 訓練実行（10エポック、Early Stopping付き）
    if python3 scripts/train_stage1.py \
        --config "configs/${config_name}.yaml" \
        --data_dir "${DATA_DIR}" \
        > "${EXPERIMENT_LOG_DIR}/training.log" 2>&1; then
        
        EXPERIMENT_STATUS="SUCCESS"
    else
        EXPERIMENT_STATUS="FAILED" 
        echo "   ⚠️ 実験でエラーが発生しましたが、続行します"
    fi
    
    # 終了時刻記録
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ "$EXPERIMENT_STATUS" = "SUCCESS" ]; then
        echo "   ✅ 実験完了: ${description}"
    else
        echo "   ❌ 実験失敗: ${description}"
    fi
    echo "   実行時間: ${DURATION}秒"
    echo "   ログ: ${EXPERIMENT_LOG_DIR}/training.log"
    
    # 実行結果サマリー抽出
    echo "実験: ${description}" > "${EXPERIMENT_LOG_DIR}/summary.txt"
    echo "設定: ${config_name}.yaml" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    echo "ステータス: ${EXPERIMENT_STATUS}" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    echo "実行時間: ${DURATION}秒" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    echo "" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    
    if [ "$EXPERIMENT_STATUS" = "SUCCESS" ]; then
        # 最終メトリクス抽出（最後のエポックの値）
        echo "最終メトリクス:" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
        tail -n 50 "${EXPERIMENT_LOG_DIR}/training.log" | grep -E "(train_loss|val_loss|val_correlation)" | tail -n 5 >> "${EXPERIMENT_LOG_DIR}/summary.txt" || echo "メトリクス抽出失敗" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    else
        echo "実験が失敗したため、メトリクスは利用できません" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
        echo "エラー詳細は training.log を確認してください" >> "${EXPERIMENT_LOG_DIR}/summary.txt"
    fi
    
    echo ""
done

echo "🎯 全実験完了"
echo ""

# 比較レポート生成
REPORT_FILE="${RESULTS_DIR}/${TIMESTAMP}/comparison_report.md"
echo "📈 比較レポート生成中: ${REPORT_FILE}"

cat > "${REPORT_FILE}" << EOF
# T5転移学習 vs ランダム初期化 比較実験レポート

**実験日時**: $(date)  
**タイムスタンプ**: ${TIMESTAMP}

## 実験設定

| 実験名 | 設定ファイル | 説明 |
|--------|-------------|------|
| T5-Freeze | t5_freeze_10ep.yaml | T5を2エポック凍結後解凍 |
| T5-NoFreeze | t5_nofreeze_10ep.yaml | T5を最初から学習 |
| T5-FrozenAll | t5_frozen_all_10ep.yaml | T5を完全凍結（表現のみ利用） |
| Baseline | t5_baseline_10ep.yaml | ランダム初期化（従来手法） |

## 実験結果

EOF

# 各実験の結果をレポートに追加
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r config_name description <<< "$experiment"
    
    echo "### ${description}" >> "${REPORT_FILE}"
    echo "" >> "${REPORT_FILE}"
    echo "\`\`\`" >> "${REPORT_FILE}"
    cat "${RESULTS_DIR}/${TIMESTAMP}/${config_name}/summary.txt" >> "${REPORT_FILE}"
    echo "\`\`\`" >> "${REPORT_FILE}"
    echo "" >> "${REPORT_FILE}"
done

cat >> "${REPORT_FILE}" << EOF
## 分析

### 完全凍結T5の効果（T5-FrozenAll）
**期待傾向**:
- **val_loss**: ランダム初期化より速く下がるが、途中から横ばい
- **val_correlation**: 0 → +0.02 程度で頭打ち
- **grad_norm**: 低い（0.5 前後）- T5部分が学習されないため

### 収束速度比較
- エポック1, 3, 5での各メトリクスの値を比較
- T5転移学習による早期収束の効果を確認
- 完全凍結での初期ブースト効果を測定

### 最終性能比較
- エポック10での最終val_correlationを比較
- T5の事前学習済み知識の純粋な効果（完全凍結）
- 微調整による追加効果（Freeze vs NoFreeze vs FrozenAll）

### 計算効率
- 実行時間の比較（完全凍結は最も高速）
- gradient norm パターンの違い
- Early Stopping による効率化効果

## 結論

[手動で分析結果を記入してください]
- T5事前学習表現の純粋効果: FrozenAll vs Baseline
- 微調整の追加価値: Freeze/NoFreeze vs FrozenAll
- 最適なT5適応戦略の特定

## TensorBoardログ

各実験のTensorBoardログは以下にあります：
- \`logs/stage1/\` ディレクトリ内の各バージョン

比較表示コマンド：
\`\`\`bash
tensorboard --logdir logs/stage1/ --port 6006
\`\`\`

## 再現方法

同じ実験を再実行する場合：
\`\`\`bash
# 個別実行例
python3 scripts/train_stage1.py --config configs/t5_freeze.yaml --data_dir ../data/derived --max_epochs 10

# 自動実験再実行
bash scripts/run_t5_experiments.sh
\`\`\`
EOF

echo "✅ 比較レポート生成完了: ${REPORT_FILE}"
echo ""

# TensorBoard起動提案
echo "📊 TensorBoard起動提案:"
echo "   tensorboard --logdir logs/stage1/ --port 6006"
echo ""

echo "🎉 T5転移学習比較実験完了"
echo "   結果ディレクトリ: ${RESULTS_DIR}/${TIMESTAMP}/"
echo "   比較レポート: ${REPORT_FILE}"