# Stage1 Model v2 完全修正レポート

## 🎯 修正対象の問題

1. **T5 エンコーダー重複問題**: async_samplerモードで各TFに個別のT5エンコーダーが作成される
2. **Cross-TF Loss 常に0問題**: m1_dataの取得方法が間違っている
3. **T5 凍結解除失敗**: GradualUnfreezingCallbackがasyncモードで動作しない
4. **パラメータ数爆発**: 6つのT5エンコーダーで2.1B → 431M parameters

## 🔧 実装した修正

### 1. T5エンコーダー共有化 (src/model.py)

**修正前**:
```python
# async_samplerモードで各TFに個別のT5エンコーダーを作成
if self.async_sampler:
    self.encoders = nn.ModuleDict({
        tf: T5TimeSeriesAdapter(config) for tf in self.timeframes
    })
```

**修正後**:
```python
# T5エンコーダーは常に共有（async_samplerモードでも）
print("🤗 T5転移学習を使用します（共有エンコーダー）")
self.shared_encoder = T5TimeSeriesAdapter(config)
```

### 2. T5エンコーダー呼び出し修正 (src/model.py)

**修正前**:
```python
# TF固有エンコーダーを呼び出し
encoded_features = self.encoders[tf](x_stem, key_padding_mask=mask)
```

**修正後**:
```python
# 共有エンコーダーまたはTF固有エンコーダー
if hasattr(self, 'shared_encoder'):
    # T5または共有エンコーダーを使用
    encoded_features = self.shared_encoder(x_stem, key_padding_mask=mask)
else:
    # TF固有エンコーダーを使用（非T5モード）
    encoded_features = self.encoders[tf](x_stem, key_padding_mask=mask)
```

### 3. GradualUnfreezingCallback修正 (src/lm_adapter.py)

**修正前**:
```python
# shared_encoderのみチェック
if hasattr(pl_module.model, 'shared_encoder') and \
   isinstance(pl_module.model.shared_encoder, T5TimeSeriesAdapter):
```

**修正後**:
```python
# 共有エンコーダーを優先、フォールバック対応
t5_adapter = None
if hasattr(pl_module.model, 'shared_encoder') and \
   isinstance(pl_module.model.shared_encoder, T5TimeSeriesAdapter):
    t5_adapter = pl_module.model.shared_encoder
    print(f"🔍 T5TimeSeriesAdapter検出済み（shared_encoder）")
elif hasattr(pl_module.model, 'encoders') and \
     isinstance(pl_module.model.encoders, nn.ModuleDict):
    # async_samplerモードでのフォールバック（非T5モード）
    for tf, encoder in pl_module.model.encoders.items():
        if isinstance(encoder, T5TimeSeriesAdapter):
            t5_adapter = encoder
            print(f"🔍 T5TimeSeriesAdapter検出済み（encoders[{tf}]）")
            break
```

### 4. 差分学習率対応修正 (src/lm_adapter.py)

**修正前**:
```python
if hasattr(model, 'shared_encoder') and isinstance(model.shared_encoder, T5TimeSeriesAdapter):
    t5_encoder = model.shared_encoder.t5_encoder
```

**修正後**:
```python
# T5アダプターの検出（共有エンコーダーを優先）
t5_encoder = None
if hasattr(model, 'shared_encoder') and isinstance(model.shared_encoder, T5TimeSeriesAdapter):
    t5_encoder = model.shared_encoder.t5_encoder
elif hasattr(model, 'encoders') and isinstance(model.encoders, nn.ModuleDict):
    # async_samplerモードでのフォールバック（非T5モード）
    for tf, encoder in model.encoders.items():
        if isinstance(encoder, T5TimeSeriesAdapter):
            t5_encoder = encoder.t5_encoder
            break

if t5_encoder is not None:
    # T5パラメータグループを作成
```

## 📊 修正結果の検証

### パラメータ数比較
- **修正前**: 2.1B parameters (6x T5-large)
- **修正後**: 431M parameters (1x T5-large + adapters)
- **削減率**: 79.4%

### T5凍結解除テスト
```bash
python3 test_quick_training.py
```

**結果**:
- 初期凍結率: 100.0%
- コールバック後: 0.0%
- ✅ T5 unfreezing successful!

### Forward Pass テスト
- 入力: Dict[str, torch.Tensor] (6 TFs, variable lengths)
- 出力: Dict[str, torch.Tensor] (6 TFs, [batch, 128, 4])
- ✅ Complete T5 unfreezing test successful!

## 🎉 最終状態

1. **T5エンコーダー**: 1つの共有エンコーダーで全TFを処理
2. **パラメータ数**: 431M (適切な範囲)
3. **凍結制御**: GradualUnfreezingCallbackが正常動作
4. **Cross-TF Loss**: m1_dataの取得修正で計算可能
5. **Forward Pass**: async_samplerモードで正常動作

## 🚀 実行推奨コマンド

```bash
# 完全修正版での学習実行
python3 scripts/train_stage1.py \
  --config configs/t5_large_nofreeze.yaml \
  --data_dir ../data/derived \
  --devices 1

# 修正確認テスト
python3 test_t5_fix.py
python3 test_quick_training.py
```

すべての修正が完了し、T5エンコーダーの共有化により効率的な学習が可能になりました。