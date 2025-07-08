# T5転移学習実装計画 - Stage-1拡張

## 🎯 目的

論文 "Random Initialization Can't Catch Up" の知見をStage-1アーキテクチャに組み込み、T5-smallの事前学習済み重みによる転移学習でマスク付き再構成タスクの収束を加速する。

## 📚 背景知見

- **論文主張**: 汎用言語モデル（T5等）で初期化した時系列モデルは、ランダム初期化より一貫して優れる
- **Stage-1適応性**: 自己教師ありマスク付き再構成タスクは言語モデルのMLMタスクと類似性が高い
- **期待効果**: 転移ギャップの享受、収束加速、学習安定性向上

## 🏗️ 設計方針

### Stage-1設計思想の踏襲
1. **設定ファイル駆動**: 全ての設定はYAMLで管理
2. **モジュラー設計**: 既存コードへの最小限の影響
3. **テスト可能性**: 小さなコンポーネントに分割
4. **段階的実装**: 漸進的な機能追加と検証

### 非破壊的統合
- 既存の`SharedEncoder`をオプションで置換
- 従来のランダム初期化も選択可能
- 設定フラグ一つで切り替え可能

## 🛠️ 実装計画

### Phase 1: 基盤実装 (2-3時間)

#### 1.1 依存関係追加
```bash
# requirements.txt更新
transformers>=4.42.0
sentencepiece>=0.1.99
```

#### 1.2 T5アダプター実装 (`src/lm_adapter.py`)
```python
class T5TimeSeriesAdapter(nn.Module):
    """T5エンコーダーを時系列データに適応させるアダプター"""
    
    def __init__(self, config):
        # Patchify: 時系列を擬似トークン化
        # T5エンコーダー: 事前学習済み重みロード
        # 次元整合: d_modelを512に統一
        
    def forward(self, x):
        # [B, TF, T, C] -> Patch化 -> T5エンコーダー -> 元形状復元
```

#### 1.3 モデル統合 (`src/model.py`修正)
```python
# SharedEncoderの条件分岐
if config.get('transfer_learning', {}).get('use_pretrained_lm', False):
    self.shared_encoder = T5TimeSeriesAdapter(config)
else:
    self.shared_encoder = VanillaSharedEncoder(config)
```

### Phase 2: 学習戦略実装 (1-2時間)

#### 2.1 段階的解凍機能 (`scripts/train_stage1.py`修正)
```python
class GradualUnfreezingCallback(pl.Callback):
    """エポック進行に応じてT5層を段階的に解凍"""
    
    def on_train_epoch_start(self, trainer, pl_module):
        # config.freeze_lm_epochsに基づいて解凍制御
```

#### 2.2 差分学習率適用
```python
# T5部分: base_lr × 0.1
# 新規部分: base_lr × 1.0
```

### Phase 3: 設定・実験管理 (1時間)

#### 3.1 設定ファイル拡張
```yaml
# configs/t5_transfer.yaml (新規)
transfer_learning:
  use_pretrained_lm: true
  lm_name_or_path: "google/t5-small"
  freeze_lm_epochs: 3
  patch_len: 16
  lm_learning_rate_factor: 0.1

# 3つの実験設定
# - baseline.yaml (use_pretrained_lm: false)
# - t5_freeze.yaml (freeze_epochs: 3)  
# - t5_nofreeze.yaml (freeze_epochs: 0)
```

#### 3.2 自動実験スクリプト
```bash
# scripts/run_t5_experiments.sh
# 3つの設定で自動実行・比較
```

### Phase 4: 評価・検証 (1時間)

#### 4.1 メトリクス比較
- エポック1, 3, 5での val_loss, val_correlation
- 学習時間・メモリ使用量の測定
- TensorBoard可視化強化

#### 4.2 結果レポート
```markdown
# experiment_t5_vs_random.md
## 実験結果
### 収束速度比較
### 最終性能比較
### 計算コスト比較
```

## 📊 実装詳細

### ファイル構成（追加・変更）
```
stage1/
├── src/
│   ├── lm_adapter.py           # 新規: T5アダプター
│   └── model.py               # 修正: T5統合
├── scripts/
│   ├── train_stage1.py        # 修正: 段階解凍対応
│   └── run_t5_experiments.sh  # 新規: 自動実験
├── configs/
│   ├── t5_baseline.yaml       # 新規: ベースライン
│   ├── t5_freeze.yaml         # 新規: 3エポック凍結
│   └── t5_nofreeze.yaml       # 新規: 凍結なし
└── docs/
    └── experiment_t5_vs_random.md # 新規: 実験レポート
```

### T5統合の技術的詳細

#### 1. 時系列の擬似トークン化
```python
# [B, TF, T=64, C=6] -> Patch(16) -> [B, TF, 4, 6*16]
# -> Linear Projection -> [B, TF, 4, 512] (T5のd_model)
```

#### 2. T5エンコーダー活用
```python
# T5-smallのencoder.block.0～3をロード
# 残りはランダム初期化で継ぎ足し
```

#### 3. 段階的解凍戦略
```python
# Epoch 0-2: T5層凍結 (requires_grad=False)
# Epoch 3+: T5層解凍 (requires_grad=True, lr×0.1)
```

## 🎯 期待される成果

### 定量的指標
- **収束加速**: エポック3時点での val_correlation が 20-30% 向上
- **最終性能**: エポック15での val_correlation が 10-15% 向上
- **学習安定性**: 損失の分散が 20-30% 減少

### 定性的効果
- **早期収束判定**: より少ないエポックで性能評価可能
- **ハイパーパラメータ感度低下**: 学習率等の調整が容易
- **転移可能性向上**: 他の時系列タスクへの応用可能性

## ⚠️ リスク管理

### 技術的リスク
1. **メモリ不足**: T5-small + バッチサイズでVRAM超過
   - 対策: batch_size自動調整、gradient_checkpointing
2. **次元不整合**: 既存アーキテクチャとの統合問題
   - 対策: アダプター層で次元変換を柔軟に対応
3. **学習不安定**: T5との学習率不整合
   - 対策: 差分学習率と段階解凍で制御

### 実験的リスク
1. **転移効果なし**: T5が時系列データに効果なしの可能性
   - 対策: 最低限のベースライン比較は実施、結果も価値ある知見
2. **計算コスト増大**: T5による学習時間延長
   - 対策: 目標+15%以内、超過時は軽量版検討

## 📋 実装チェックリスト

### Phase 1: 基盤実装
- [ ] `requirements.txt` 更新
- [ ] `src/lm_adapter.py` 実装
- [ ] `src/model.py` T5統合
- [ ] 基本動作テスト (fast_dev_run)

### Phase 2: 学習戦略
- [ ] 段階解凍コールバック実装
- [ ] 差分学習率適用
- [ ] Lightning統合テスト

### Phase 3: 実験設定
- [ ] 3つの設定ファイル作成
- [ ] 自動実験スクリプト作成
- [ ] TensorBoardログ強化

### Phase 4: 評価・ドキュメント
- [ ] 3設定での比較実験実行
- [ ] 結果レポート作成
- [ ] README.md 更新

## 🚀 実装開始

このドキュメントに基づいて、Phase 1から順次実装を開始します。各Phaseの完了後、必ず動作確認とテストを実施し、問題があれば設計を見直します。

**Stage-1の設計思想（設定駆動・モジュラー・テスト可能）を維持しながら、T5転移学習の恩恵を最大限活用する実装を目指します。**