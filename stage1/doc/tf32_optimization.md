# TF32最適化設定

## 概要
TensorFloat-32 (TF32)は、RTX 30xx/40xx/A100等の新しいGPUで利用可能な数値形式で、FP32の精度を保ちながら約2倍の高速化を実現します。

## 設定内容

### 1. コード内設定 (`src/optimization.py`)
```python
# TF32を有効化
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Tensor Core最適化
torch.set_float32_matmul_precision('high')

# CuDNN最適化
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### 2. 環境変数設定 (Dockerfile)
```bash
ENV TORCH_ALLOW_TF32_OVERRIDE=1
```

## 対応GPU

### ✅ TF32対応
- **RTX 30xx系**: RTX 3080, 3090, 3090 Ti
- **RTX 40xx系**: RTX 4080, 4090
- **Data Center**: A100, H100

### ❌ TF32非対応
- RTX 20xx系以前
- GTX系
- Tesla V100以前

## パフォーマンス向上

### 期待される効果
- **行列演算**: 約1.5-2倍高速化
- **Transformer**: 約1.3-1.8倍高速化
- **メモリ使用量**: 変化なし（精度は維持）

### T5-Large での予想効果
```
従来（FP32のみ）: 
- 1エポック: ~45分
- 5エポック: ~3.75時間

TF32有効化後:
- 1エポック: ~30分 (33%短縮)
- 5エポック: ~2.5時間 (33%短縮)
```

## 確認方法

### 1. 実行時ログで確認
```
🚀 PyTorch最適化設定開始...
✅ TF32有効化: matmul & cuDNN
✅ Tensor Core最適化: float32 matmul precision = 'high'
✅ CuDNN benchmark有効化
✅ CuDNN非決定論的アルゴリズム許可
✅ CUDAキャッシュクリア
✅ 全最適化設定完了!

📊 現在の最適化設定:
  TF32 (matmul): True
  TF32 (cuDNN):  True
  Float32 precision: high
  CUDA device: NVIDIA GeForce RTX 4090
  CUDA capability: (8, 9)
```

### 2. 手動確認
```python
import torch
print(f"TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
print(f"Float32 precision: {torch.get_float32_matmul_precision()}")
```

## 注意事項

### 1. 数値精度
- TF32は約6桁の有効数字（FP32は約7桁）
- 機械学習では通常問題なし
- 科学計算では要注意

### 2. 非対応GPU
- 古いGPUでは設定が無視される
- エラーにはならない（自動フォールバック）

### 3. デバッグ時
- 再現性が必要な場合は無効化
- `torch.backends.cudnn.deterministic = True`

## vast.ai での活用

### 推奨インスタンス
1. **RTX 4090**: 最もコスパが良い
2. **A100**: 大量メモリが必要な場合
3. **RTX 4080**: 予算重視

### 期待される時間短縮
```
従来の5エポック訓練（10%データ）:
- RTX 4090: ~4時間 → ~2.7時間
- A100: ~3時間 → ~2時間
```

## 関連ファイル

- `src/optimization.py`: TF32設定モジュール
- `scripts/train_stage1.py`: 自動最適化適用
- `scripts/evaluate_stage1.py`: 評価時最適化
- `Dockerfile`: 環境変数設定