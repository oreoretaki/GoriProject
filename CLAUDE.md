# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GoriProject is a sophisticated USD/JPY machine learning system that implements a multi-stage architecture for financial time-series analysis and trading. The project uses self-supervised learning with advanced multi-timeframe reconstruction techniques.

**Tech Stack**: PyTorch, PyTorch Lightning, Transformers (T5), Pandas, Docker, CUDA 11.8/12.8

## Project Architecture

This is a USD/JPY machine learning project with a stage-based architecture:

- **data/**: Shared data directory containing source SQLite DB and derived high-quality Parquet files
- **stage0/**: Data preprocessing foundation (✅ COMPLETED) - transforms raw OANDA data into ML-ready format
- **stage1/**: Self-supervised multi-TF reconstruction (✅ COMPLETED) - trains encoder-decoder for masked span reconstruction
- **doc/**: Comprehensive project documentation (6 files covering architecture, API, developer guide)
- **tests/**: Unit tests for data pipeline and components
- **scripts/**: Deployment and Git automation scripts
- **old/**: Deprecated files and experiments (ignore unless researching history)

### Stage 0 - Data Preprocessing Foundation (COMPLETED)

Stage 0 is the critical data preprocessing pipeline that transforms raw OANDA USD/JPY historical data into high-quality, ML-ready datasets. It follows a strict ETL architecture:

**Data Flow**: SQLite DB → ETL Processing → Parquet Files → Validation → CI/CD

**Key Components**:
- `scripts/make_tf_data.py`: Main ETL pipeline that generates all timeframes from M1 source
- `scripts/run_validate.py`: Fast validation script (0.7s) that ensures 100% timeframe consistency
- `reports/hash_manifest.json`: MD5 integrity tracking for all data files
- `.github/workflows/data_hash.yml`: CI/CD pipeline for automated data quality checks

**Critical Data Constraints**:
- Volume column is ABSOLUTELY PROHIBITED (contains tick counts, not actual trading volume)
- Original SQLite timeframe tables are PROHIBITED (only 0.3-0.6% consistency)
- Only use Parquet files from `data/derived/simple_gap_aware_*.parquet` (100% consistency guaranteed)

## Common Commands

### 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/oreoretaki/GoriProject.git
cd GoriProject

# Verify Stage 0 data quality (should complete in ~0.8s)
cd stage0
python3 scripts/run_validate.py

# Run Stage 1 fast test
cd ../stage1
python3 scripts/train_stage1.py --config configs/test.yaml --data_dir ../data/derived --fast_dev_run --devices 1
```

### Stage 0 Commands
```bash
cd stage0

# Data validation (0.7-0.8s execution, 100% consistency expected)
python3 scripts/run_validate.py

# Data integrity check
python3 scripts/verify_hash_manifest.py

# Regenerate all timeframe data from source (if needed)
python3 scripts/make_tf_data.py

# Run tests
pytest                           # Run all tests
pytest tests/test_boundary.py    # Run boundary condition tests
python3 tests/test_boundary.py   # Direct execution
```

### Stage 1 Training Commands
```bash
cd stage1

# Fast test execution (recommended first step) - lightweight config
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run \
  --devices 1

# Short training run (3 epochs)
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --max_epochs 3 \
  --devices 1

# Full training with base config
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --devices 1

# T5 transfer learning (recommended for production)
python3 scripts/train_stage1.py \
  --config configs/t5_large_nofreeze.yaml \
  --data_dir ../data/derived \
  --devices 1 \
  --precision bf16-true \
  --max_epochs 5 \
  --mask_token_lr_scale 0.1

# Multiple seed evaluation
python3 scripts/train_stage1.py \
  --config configs/shared_base.yaml \
  --seeds 42 123 2025
```

### Evaluation Commands
```bash
cd stage1

# Evaluate trained model
python3 scripts/evaluate_stage1.py \
  --config configs/base.yaml \
  --ckpt checkpoints/stage1-epoch-XX.ckpt \
  --data_dir ../data/derived \
  --output evaluation_results.json

# Evaluate with different mask ratios
python3 scripts/evaluate_stage1.py \
  --config configs/base.yaml \
  --ckpt checkpoints/best.ckpt \
  --eval_mask_ratio 0.0  # No masking
```

### Testing Commands
```bash
# Test data pipeline
cd tests
python3 test_stage1_data.py

# Unit tests for losses
python3 unit/test_losses.py

# Stage 0 tests
cd ../stage0/tests
python3 test_boundary.py
python3 test_tf_generation.py
```

### Monitoring Commands
```bash
# Start TensorBoard
tensorboard --logdir stage1/logs/

# Monitor GPU usage during training
cd stage1
python3 scripts/monitor_gpu.py

# Analyze training logs
python3 analyze_logs.py

# Profile model bottlenecks
python3 scripts/profile_bottleneck.py
```

### Deployment & Docker Commands
```bash
# Build Docker image
docker build -t goriproject:latest .

# Run with Docker Compose
docker-compose up -d

# Deploy to vast.ai (includes setup instructions)
./scripts/deploy_vastai.sh

# Use pre-built Docker image
docker run -it --gpus all oreoretaki/goriproject:latest

# Quick Git sync
GITHUB_TOKEN=your_token ./scripts/quick_update.sh "Update message"

# Full Git sync with pull
./scripts/git_sync.sh
```

## Stage 0 Data Architecture

**Processing Pipeline**:
1. Load M1 data from SQLite (Volume column excluded)
2. Apply timezone standardization (UTC)
3. Generate higher timeframes using pandas resample (label='left', closed='left')
4. Apply OHLC logical validation
5. Save as Parquet with Snappy compression
6. Generate integrity reports and manifests

**Available Datasets** (all in `data/derived/`):
- M1: 3.1M records (44.8MB) - Source of truth
- M5: 630K records (10.7MB) 
- M15: 210K records (4.3MB)
- M30: 105K records (2.4MB)
- H1: 53K records (1.3MB)
- H4: 14K records (0.4MB)
- D: 3K records (0.1MB)

**Quality Guarantees**:
- 100% timeframe consistency (M1 aggregation perfectly matches higher TFs)
- OHLC logical constraints enforced
- Gap-aware processing without artificial gap filling
- UTC timezone standardization
- Volume column completely excluded

## Development Requirements

### System Requirements
- **Python**: 3.10+ (3.11 recommended)
- **CUDA**: 11.8 or 12.8 (for GPU training)
- **Memory**: 16GB+ RAM recommended
- **GPU**: RTX 3090/4090, A100, or H100 (24GB+ VRAM for T5-large)
- **Storage**: 50GB+ for data and checkpoints

### Dependencies Installation
```bash
# Install from root requirements.txt (includes CUDA 11.8 PyTorch)
pip install -r requirements.txt

# Or install Stage 1 specific requirements
cd stage1
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.4.1 (CUDA 11.8/12.8)
- PyTorch Lightning 2.4.0
- Transformers 4.53.1 (for T5 models)
- Pandas 2.3.0 + PyArrow for Parquet
- TensorBoard for monitoring

## Critical Rules for Stage 1 Development

When working on Stage 1 (ML preprocessing):

1. **NEVER access original SQLite tables** - they have broken consistency
2. **NEVER use Volume data** - it's tick counts, not trading volume
3. **ALWAYS validate Stage 0 completion** before Stage 1 development: `cd stage0 && python3 scripts/run_validate.py`
4. **ONLY use data from** `data/derived/simple_gap_aware_*.parquet` files
5. **All timestamps are UTC** - no timezone conversion needed

## Stage 0 Success Criteria

Before any Stage 1 work, verify Stage 0 meets these criteria:
- M1→all TF consistency: 100.0%
- Volume exclusion: Complete
- Validation time: <1 minute (typically 0.7s)
- OHLC logical violations: 0
- CI automation: Implemented

Stage 0 is considered production-ready and should not be modified unless data quality issues are discovered.

### Stage 1 - Self-Supervised Multi-TF Reconstruction (COMPLETED)

Stage 1 implements a sophisticated self-supervised learning system that trains a single encoder-decoder model to simultaneously reconstruct masked spans across 6 aligned timeframes (M1, M5, M15, H1, H4, D) while enforcing cross-scale consistency.

**Architecture Components**:
- **Multi-TF Window Sampler**: Synchronized sampling with right-edge alignment (3.3hr windows)
- **Feature Engineering**: [open, high, low, close, Δclose, %body] → 36 channels
- **Masking Strategy**: Random contiguous blocks (5-60 bars), 15% mask ratio, TF-synchronized
- **Model**: TF-specific CNNs + Shared Encoder + Bottleneck + TF-specific Decoders
- **4-Component Loss**: Huber + Multi-resolution STFT + Cross-TF consistency + Amplitude-phase correlation
- **T5 Transfer Learning**: Optional pre-trained T5 encoder initialization with layerwise LR decay

**Recent Updates** (引き継ぎメモより):
1. **TF別WindowSampler**: SingleTFWindowSampler実装、空TF除外、TF固有gap/キャッシュ対応
2. **Learnable Mask Token**: nn.Parameter化、in-place置換実装
3. **マスクdtype統一**: 全てtorch.bool
4. **評価指標リーク対策**: マスク位置のみでcorr計算
5. **Optimizer拡張**: mask_token独立学習率 (mask_token_lr_scale)
6. **Docker/vast.ai対応**: oreoretaki/goriproject:latest、H100対応済み

**Key Files**:
- `stage1/src/data_loader.py`: Main data pipeline orchestration
- `stage1/src/model.py`: Neural network architecture (TF-CNN + Transformer encoder + decoders)
- `stage1/src/losses.py`: Multi-component loss functions
- `stage1/src/lm_adapter.py`: T5 transfer learning adapter
- `stage1/scripts/train_stage1.py`: PyTorch Lightning training with One-Cycle LR
- `stage1/scripts/evaluate_stage1.py`: Comprehensive evaluation with correlation/consistency metrics

**Configuration Files**:
- `configs/shared_base.yaml`: Unified base configuration (seq_len=128, standardized loss weights)
- `configs/test.yaml`: Lightweight test config (505K params, 5 epochs) - recommended for initial tests
- `configs/base.yaml`: Full production training (3.5M params, 40 epochs)
- `configs/t5_freeze.yaml`: T5 with progressive unfreezing (recommended for transfer learning)
- `configs/t5_large_nofreeze.yaml`: T5-large experiments (requires 24GB+ VRAM, H100推奨)

## Critical Rules for Stage 2 Development

When working on Stage 2 (RL Trading Agent):

1. **ALWAYS validate Stage 1 completion** before Stage 2 development
2. **USE learned weights from Stage 1** as initialization for the RL agent encoder
3. **MAINTAIN the same timeframe alignment** principles from Stage 1  
4. **RESPECT the same data constraints** from Stage 0 (no volume, Parquet only, UTC timezone)
5. **LEVERAGE the bottleneck representations** from Stage 1 as state features for RL

Stage 1 trained weights provide rich multi-timeframe representations that capture cross-scale dependencies and temporal patterns essential for trading decisions.

## Stage 1 データリーク対策パラメータ

Stage 1では、時系列検証データリーク問題を解決するため、以下の新パラメータを導入：

### validation.val_gap_days
- **デフォルト値**: `30.0`
- **推奨値**: `30.0` (30日間のギャップ)
- **説明**: 訓練データと検証データの間に設ける時間的ギャップ（日数単位）
- **効果**: 訓練データの最後と検証データの最初の間に30日間のギャップを作り、時系列依存性を完全に断絶

### evaluation.eval_mask_ratio
- **デフォルト値**: `null` (通常のマスク率を使用)
- **推奨値**: 
  - `null`: 通常評価
  - `0.0`: マスクなし評価（全体復元性能測定）
  - `1.0`: 全マスク評価（生成能力測定）
- **説明**: 評価時のマスク率オーバーライド。通常の15%マスクとは異なる条件で評価可能
- **効果**: 訓練時とは独立したマスク条件で、より公平な評価指標を取得

### 使用例
```yaml
# shared_base.yaml での設定
validation:
  val_split: 0.2
  val_gap_days: 30.0             # データリーク防止（30日間の完全分離）

evaluation:
  eval_mask_ratio: null          # 通常評価（null=15%マスク, 0=マスクなし, 1=全マスク）
```

### コマンドライン使用例
```bash
# デフォルト設定（リーク修正済み）
python3 scripts/train_stage1.py --config configs/shared_base.yaml --data_dir ../data/derived --devices 1

# カスタムギャップ設定
python3 scripts/train_stage1.py --config configs/shared_base.yaml --val_gap_days 2.0 --eval_mask_ratio 0.0

# 複数シード評価
python3 scripts/train_stage1.py --config configs/shared_base.yaml --seeds 42 123 2025
```

## 🔧 Environment Compatibility & Troubleshooting

### WSL Environment Setup
The project is optimized for WSL (Windows Subsystem for Linux) environments:
- **MPI Detection**: Automatically handled with `PL_DISABLE_FORK=1` environment variable
- **TensorBoard**: Install with `pip install tensorboard`
- **GPU Access**: Ensure WSL2 with CUDA support is properly configured

### vast.ai Deployment (H100推奨)
```bash
# H100 NVL (CUDA 12.8) インスタンスで動作確認済み
docker run -it --gpus all oreoretaki/goriproject:latest

# 本番5エポック学習
python scripts/train_stage1.py \
    --config configs/t5_large_nofreeze.yaml \
    --data_dir ../data/derived \
    --devices 1 \
    --precision bf16-true \
    --max_epochs 5 \
    --mask_token_lr_scale 0.1
```

### Common Issues & Solutions

#### Triton AUTOTUNE Logs
- **Issue**: 大量のTriton AUTOTUNEログ出力
- **Solution**: 正常動作（カーネルベンチマーク）。初回のみ時間がかかる

#### GPU Memory Issues
```bash
# For T5-large models, use gradient accumulation
python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --accumulate_grad_batches 16
```

#### Data Loading Errors
```bash
# Reduce workers if encountering bus errors
python3 scripts/train_stage1.py --config configs/base.yaml --num_workers 0
```

#### NVML Error / nvidia-smi Instability
- **Solution**: vast.aiホストのdriver更新直後に発生。再起動またはホスト移行で解消

### Performance Optimization
- **TF32 Mode**: Automatically enabled for RTX 30xx/40xx/A100/H100 GPUs
- **Mixed Precision**: Use `precision="bf16-true"` for H100 (最速・安定)
- **Batch Size**: Adjust based on GPU memory (default: 32)

### Recommended Training Progression
1. **Initial Test**: `configs/test.yaml` with `--fast_dev_run` (1 batch確認)
2. **Short Training**: `configs/test.yaml` with `--max_epochs 3`
3. **Production**: `configs/t5_large_nofreeze.yaml` with 5 epochs (val corr ≈ 0.20±0.03目標)
4. **Hyperparameter Tuning**: mask_token_lr_scale感度確認 (0.05, 0.1, 0.2)

### Project Organization
- `old/`: Deprecated files and experiments (ignore unless debugging)
- `doc/`: Comprehensive documentation (Stage1_仕様書.md, ARCHITECTURE.md, etc.)
- `tests/`: Unit tests for all components
- `checkpoints/`: Saved model weights
- `logs/`: TensorBoard logs (view with `tensorboard --logdir logs/`)
- `cache/`: Cached window indices for faster data loading

## CI/CD Integration
The project includes GitHub Actions for automated data integrity checks:
- **Workflow**: `.github/workflows/data_hash.yml` (in stage0/)
- **Triggers**: Push/PR to main branch affecting data files
- **Checks**: Hash verification, data validation, boundary tests
- **Artifacts**: Integrity reports saved for 30 days

## 📌 Project Summary

GoriProject is a production-ready financial ML system with:
- **Data Quality**: 100% timeframe consistency with automated validation
- **Advanced Architecture**: Multi-timeframe self-supervised learning with cross-scale consistency
- **Transfer Learning**: T5 model integration with learnable mask tokens
- **Scalability**: Docker containerization and vast.ai/H100 deployment support
- **Monitoring**: Comprehensive logging, TensorBoard integration, and CI/CD pipelines
- **Documentation**: Extensive technical documentation in both English and Japanese

The project serves as a foundation for USD/JPY trading strategies, with Stage 0 and Stage 1 fully completed and ready for Stage 2 (RL Trading Agent) development.