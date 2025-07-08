# Stage 1 開発者ガイド

## 🛠️ 開発環境セットアップ

### 必要な依存関係

```bash
# 基本的な機械学習スタック
pip install torch>=2.0.0
pip install pytorch-lightning>=2.0.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install pyarrow>=12.0.0
pip install scipy>=1.10.0

# 追加ライブラリ
pip install tensorboard
pip install matplotlib
pip install seaborn
pip install tqdm
pip install omegaconf
pip install torchmetrics
pip install psutil
```

### 開発ツール

```bash
# コード品質
pip install black isort flake8
pip install pytest pytest-cov

# 型チェック  
pip install mypy
```

## 🔧 コードベース理解

### 主要クラス関係図

```
Stage1Dataset
├── MultiTFWindowSampler    # データ切り出し
├── FeatureEngineer         # 特徴量変換
├── TFNormalizer           # 正規化
└── MaskingStrategy        # マスキング

Stage1Model
├── TFSpecificStem × 6     # TF固有処理
├── SharedEncoder          # 共有エンコーダー
├── Bottleneck            # 圧縮
└── TFDecoder × 6         # TF別再構築

Stage1CombinedLoss
├── HuberLoss             # 基本再構築
├── STFTLoss              # スペクトラム
├── CrossTFConsistencyLoss # 整合性
└── AmplitudePhaseCorrelationLoss # 高次統計
```

### 重要な設計パターン

#### 1. 設定駆動開発
```python
# 全ての設定はYAMLで管理
config = yaml.safe_load(open('configs/base.yaml'))

# 実行時設定変更
config['training']['batch_size'] = 32
config['model']['encoder']['n_layers'] = 12
```

#### 2. PyTorch Lightning採用理由
```python
# 定型的な訓練ループを抽象化
class Stage1LightningModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # 損失計算のみに集中
        return self.criterion(self.model(batch['features']), batch['targets'])
    
    def configure_optimizers(self):
        # オプティマイザー設定を分離
        return self._create_optimizer_and_scheduler()
```

#### 3. モジュラー設計
```python
# 各コンポーネントは独立してテスト可能
def test_window_sampler():
    sampler = MultiTFWindowSampler(dummy_data, seq_len=50)
    assert len(sampler) > 0
    
def test_feature_engineer():
    engineer = FeatureEngineer(config)
    features, targets = engineer.process_window(window_data)
    assert features.shape[-1] == 6  # 6特徴量
```

## 🚀 開発ワークフロー

### 1. 新機能開発手順

```bash
# 1. ブランチ作成
git checkout -b feature/new-masking-strategy

# 2. 実装
# src/ 配下でコード変更

# 3. テスト追加/更新
# tests/ 配下でテスト拡張

# 4. ローカルテスト
cd tests && python3 test_stage1_data.py

# 5. 小規模実験で動作確認（fast_dev_run完了済み）
cd stage1 && python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --max_epochs 3 \
  --devices 1

# 6. コード品質チェック
black src/ scripts/
isort src/ scripts/
flake8 src/ scripts/

# 7. コミット・プッシュ
git add . && git commit -m "Add new masking strategy"
git push origin feature/new-masking-strategy
```

### 2. デバッグ手順

#### データパイプラインデバッグ
```python
# 1. 小さなデータで確認
config['data']['seq_len'] = 10  # 短縮
dataset = Stage1Dataset('../data/derived', config, split='train')
sample = dataset[0]
print(f"Features shape: {sample['features'].shape}")

# 2. 可視化でデバッグ
import matplotlib.pyplot as plt
m1_features = sample['features'][0]  # M1
plt.plot(m1_features[:, 3])  # close価格
plt.title("M1 Close Prices")
plt.show()

# 3. マスク確認
masks = sample['masks']
print(f"Mask ratio per TF: {masks.mean(dim=1)}")
```

#### モデルデバッグ
```python
# 1. Forward pass確認
model = Stage1Model(config)
batch_size = 2
dummy_input = torch.randn(batch_size, 6, 50, 6)
outputs = model(dummy_input)
print(f"Output shape: {outputs['reconstructed'].shape}")

# 2. 勾配流れ確認
loss = torch.sum(outputs['reconstructed'])
loss.backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient: {name}")

# 3. パラメータ数確認
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

### 3. 実験管理

#### 設定ファイル管理
```yaml
# configs/experiment_name.yaml
# 基本設定を継承して部分変更
defaults:
  - base

# 実験固有の変更
model:
  encoder:
    n_layers: 16  # 深いモデル実験

training:
  batch_size: 16  # 大きなモデル用
  
experiment_name: "deep_encoder_v1"
```

#### 実験実行
```bash
# コンポーネントテスト（推奨：最初に実行）
python3 test_components.py

# fast_dev_run（動作確認）
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run

# 段階的学習実行（推奨順序）

# 1. 開発確認
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run

# 2. 中規模学習（15エポック）
python3 scripts/train_stage1.py \
  --config configs/medium.yaml \
  --data_dir ../data/derived

# 3. 本番学習（40エポック）
python3 scripts/train_stage1.py \
  --config configs/production.yaml \
  --data_dir ../data/derived

# カスタム設定での実行
python3 scripts/train_stage1.py \
  --config configs/production.yaml \
  --data_dir ../data/derived \
  --max_epochs 20 \
  --batch_size 24

# TensorBoard監視
tensorboard --logdir logs/

# 結果評価
python3 scripts/evaluate_stage1.py \
  --config configs/test.yaml \
  --ckpt checkpoints/stage1_best.ckpt \
  --data_dir ../data/derived
```

## 🧪 テスト戦略

### テスト階層

```
tests/
├── test_stage1_data.py           # データパイプライン統合テスト
├── unit/
│   ├── test_window_sampler.py    # ウィンドウサンプリング単体
│   ├── test_feature_engineering.py # 特徴量計算単体  
│   ├── test_masking.py           # マスキング単体
│   ├── test_normalization.py     # 正規化単体
│   ├── test_model.py             # モデル単体
│   └── test_losses.py            # 損失関数単体
└── integration/
    ├── test_training_loop.py     # 訓練ループ統合
    └── test_evaluation.py        # 評価統合
```

### 新機能テスト例

```python
# tests/unit/test_new_feature.py
import unittest
import torch
from stage1.src.new_feature import NewFeatureClass

class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        self.config = {...}  # テスト用設定
        self.feature = NewFeatureClass(self.config)
    
    def test_basic_functionality(self):
        """基本機能テスト"""
        input_data = torch.randn(4, 6, 50, 6)
        output = self.feature(input_data)
        
        # 形状チェック
        self.assertEqual(output.shape, expected_shape)
        
        # 値域チェック
        self.assertTrue(torch.all(output >= 0))
        
        # NaN/Inf チェック
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_edge_cases(self):
        """エッジケーステスト"""
        # 空入力
        empty_input = torch.zeros(1, 6, 0, 6)
        with self.assertRaises(ValueError):
            self.feature(empty_input)
        
        # 単一サンプル
        single_input = torch.randn(1, 6, 1, 6)
        output = self.feature(single_input)
        self.assertIsNotNone(output)

if __name__ == '__main__':
    unittest.main()
```

## 📊 性能プロファイリング

### ベクトル化WindowSampler性能

Stage 1では**O(N²) → O(N log N)**への大幅な性能改善を達成しました：

```
実データ性能結果（3.1M行）:
- 従来実装: 486秒（フォールバック）
- ベクトル化: 22秒（22倍高速化！）
- メモリ効率: バッチ処理で安定
- 学習ループ: fast_dev_run完全成功
```

**技術要因:**
- NumPy searchsorted による高速時間検索
- タイムゾーン統一処理（tz-aware/tz-naive対応）
- フォールバック機能による安全性
- 動的latent_len計算（seq_len=50 → latent_len=12）
- デバイス整合性保証（GPU/CPU統合）

### 学習ループ最適化

```python
# 完了済み最適化項目
- 動的形状処理: compressed.size(2)で実測値使用
- デバイス統一: torch.tensor(0.0, device=pred.device)
- Tensor Core最適化: torch.set_float32_matmul_precision('high')
- MPI完全回避: 環境変数をimport前に設定
```

### I/O・DataLoader最適化

```yaml
# 高速化設定（configs/test.yaml実装済み）
dataloader:
  num_workers: 8        # I/O並列化（元: 2）
  prefetch_factor: 4    # プリフェッチ追加
  persistent_workers: true
  pin_memory: true
  
training:
  batch_size: 64        # 大幅増加（元: 8）
  precision: "16-mixed" # Tensor Core活用
  accumulate_grad_batches: 2  # 見かけbatch_size=128

data:
  seq_len: 48           # 4の倍数最適化（元: 50）
  
scheduler:
  pct_start: 0.05       # 短縮warm-up（元: 0.3）
  interval: "step"      # step単位調整
```

### ウィンドウキャッシュ最適化

```python
# 自動キャッシュシステム実装済み
- ハッシュベースキャッシュ: MD5でデータ特性を識別
- インデックス保存: np.save/loadで高速化
- 20秒 → 数百ms: 40-100倍の高速化
- キャッシュディレクトリ: ../data/derived/cache/
```

### 期待される総合性能

```
従来版: 2 it/s → 36h/epoch
最適化版: 7-9 it/s → 5-7h/epoch
総合効果: 5-7倍高速化
```

### メモリ使用量監視

```python
import psutil
import torch

def profile_memory_usage():
    """メモリ使用量プロファイリング"""
    
    # システムメモリ
    process = psutil.Process()
    print(f"System memory: {process.memory_info().rss / 1024**2:.1f} MB")
    
    # GPU メモリ
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

# 訓練中の監視
def training_step_with_profiling(self, batch, batch_idx):
    if batch_idx % 100 == 0:
        profile_memory_usage()
    
    return self.criterion(self.model(batch['features']), batch['targets'])
```

### 計算時間プロファイリング

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """計算時間測定"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"{description}: {elapsed:.4f}秒")

# 使用例
with timer("Data loading"):
    batch = next(iter(dataloader))

with timer("Forward pass"):
    outputs = model(batch['features'])

with timer("Loss calculation"):
    losses = criterion(outputs['reconstructed'], batch['targets'])
```

## 🐛 よくある問題と解決法

### 1. CUDA Out of Memory
```python
# 問題: GPU メモリ不足
RuntimeError: CUDA out of memory

# 解決策
# 1. バッチサイズ削減
config['training']['batch_size'] = 8  # デフォルト24から削減

# 2. 勾配累積使用
config['training']['accumulate_grad_batches'] = 4

# 3. 混合精度使用
config['training']['precision'] = 'bf16'

# 4. モデルサイズ削減
config['model']['encoder']['n_layers'] = 4  # デフォルト8から削減
```

### 2. 収束しない
```python
# 問題: 損失が下がらない

# 診断手順
# 1. 学習率確認
config['training']['scheduler']['max_lr'] = 1e-4  # 下げる

# 2. 勾配クリッピング確認
config['training']['gradient_clip'] = 0.5  # 強化

# 3. オーバーフィッティングテスト
python3 scripts/train_stage1.py --config configs/overfit_test.yaml

# 4. データ確認
python3 tests/test_stage1_data.py
```

### 3. MPI環境問題（WSL）
```bash
# 問題: RuntimeError: cannot load MPI library

# 解決策1: 環境変数設定
export PYTORCH_LIGHTNING_DISABLE_MPI=1
export OMPI_DISABLED=1
export I_MPI_DISABLED=1

# 解決策2: 専用スクリプト使用
./scripts/run_stage1.sh

# 解決策3: importより前に設定（推奨）
# scripts/train_stage1.py の冒頭で設定済み
```

### 4. マスキング率異常
```python
# 問題: マスクが効いていない

# デバッグ
sample = dataset[0]
masks = sample['masks']
print(f"Actual mask ratios: {masks.mean(dim=-1)}")
print(f"Expected: {config['masking']['mask_ratio']}")

# マスク可視化
import matplotlib.pyplot as plt
plt.imshow(masks[0].numpy(), aspect='auto')  # TF=0のマスク
plt.colorbar()
plt.title("Masking Pattern")
plt.show()
```

### 5. データ読み込み問題（ウィンドウサンプリング0個）
```python
# 問題: 0個の有効ウィンドウ（M1以外のTFで1970年データ）
0 有効ウィンドウ

# 原因: timestamp列が正しくインデックスとして使用されていない
# M1: 正しい日時インデックス
# 他TF: 1970年からのマイクロ秒インデックス（偽の日時）

# 解決策: data_loader.py の修正済み
# M1以外はtimestamp列をインデックスに使用
if tf == 'm1':
    df.index = pd.to_datetime(df.index)
else:
    if 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'])
        df = df.drop('timestamp', axis=1)
```

### 6. cuFFT混合精度問題
```python
# 問題: 16bit混合精度でFFT実行時エラー
RuntimeError: cuFFT only supports dimensions whose sizes are powers of two 
when computing in half precision, but got a signal size of[48]

# 解決策1: FFT計算を32bitに強制（実装済み）
pred_signal_32 = pred_signal.float()
target_signal_32 = target_signal.float()
pred_fft = torch.fft.fft(pred_signal_32, dim=-1)

# 解決策2: seq_lenを2の累乗に変更
seq_len: 64  # 48 → 64
```

### 7. メソッド名不一致問題
```python
# 問題: AttributeError: 'FeatureEngineer' object has no attribute 'process_multi_tf'

# 原因: 実装と呼び出しのメソッド名不一致
# 修正前: self.feature_engineer.process_multi_tf(window_data)
# 修正後: self.feature_engineer.process_window(window_data)

# 戻り値も修正
features, targets = self.feature_engineer.process_window(window_data)
```

## 📈 性能最適化

### 1. データローダー最適化
```python
# 高速化設定
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,           # CPU並列化
    pin_memory=True,         # GPU転送高速化
    persistent_workers=True, # ワーカー再利用
    prefetch_factor=4        # プリフェッチ
)
```

### 2. モデル最適化
```python
# コンパイル高速化 (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')

# 効率的な attention
from torch.nn.functional import scaled_dot_product_attention
# FlashAttention自動適用
```

### 3. 混合精度最適化
```yaml
# configs/performance.yaml
training:
  precision: 'bf16'           # BFloat16使用
  gradient_clip: 1.0          # 数値安定性
  
model:
  encoder:
    flash_attn: true          # FlashAttention有効化
```

## 🔄 CI/CD 統合

### GitHub Actions 設定例

```yaml
# .github/workflows/stage1_test.yml
name: Stage 1 Tests

on:
  push:
    paths: ['stage1/**', 'tests/**']
  pull_request:
    paths: ['stage1/**', 'tests/**']

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run data pipeline tests
      run: |
        cd tests
        python3 test_stage1_data.py
        
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v
        
    - name: Check code style
      run: |
        black --check stage1/
        isort --check stage1/
        flake8 stage1/
```

## 🚀 本番デプロイ準備

### モデル保存・ロード

```python
# 保存 (訓練後)
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'epoch': epoch,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': best_loss,
}, 'stage1_production.pth')

# ロード (推論用)
checkpoint = torch.load('stage1_production.pth')
config = checkpoint['config']
model = Stage1Model(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 推論パイプライン

```python
class Stage1InferenceEngine:
    """本番推論エンジン"""
    
    def __init__(self, checkpoint_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        self.model = Stage1Model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 正規化統計ロード
        self.normalizer = TFNormalizer(self.config)
        self.normalizer.load_stats()
    
    def extract_features(self, multi_tf_data: Dict[str, pd.DataFrame]) -> torch.Tensor:
        """マルチTFデータから特徴量抽出"""
        
        with torch.no_grad():
            # 前処理
            engineer = FeatureEngineer(self.config)
            features, _ = engineer.process_window(multi_tf_data)
            features = self.normalizer.normalize(features.unsqueeze(0))
            
            # エンコード
            features = features.to(self.device)
            outputs = self.model(features)
            encoded = outputs['encoded']  # [1, n_tf, latent_len, d_model]
            
            return encoded.cpu()

# 使用例
engine = Stage1InferenceEngine('checkpoints/stage1_best.pth')
features = engine.extract_features(current_market_data)
# features を Stage 2 RL エージェントに渡す
```

このガイドにより、開発者はStage 1の実装・デバッグ・最適化・デプロイまで包括的に理解できます。