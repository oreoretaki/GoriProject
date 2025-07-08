# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

This is a USD/JPY machine learning project with a stage-based architecture:

- **data/**: Shared data directory containing source SQLite DB and derived high-quality Parquet files
- **stage0/**: Data preprocessing foundation (✅ COMPLETED) - transforms raw OANDA data into ML-ready format
- **stage1/**: Self-supervised multi-TF reconstruction (✅ COMPLETED) - trains encoder-decoder for masked span reconstruction

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

### Stage 0 Validation
```bash
cd stage0
python3 scripts/run_validate.py
# Expected: 0.7s execution, 100% consistency across all timeframes
```

### Running Tests
```bash
cd stage0
pytest                           # Run all tests
pytest tests/test_boundary.py    # Run boundary condition tests
python3 tests/test_boundary.py   # Direct execution
```

### Data Integrity Verification
```bash
cd stage0
python3 scripts/verify_hash_manifest.py  # Verify data file integrity
```

### Stage 1 Execution
```bash
cd stage1

# Fast test execution (recommended first step)
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run \
  --devices 1

# Development data creation (for large datasets)
python3 scripts/create_dev_data.py
python3 scripts/train_stage1.py \
  --config configs/dev.yaml \
  --data_dir ../data/dev \
  --fast_dev_run \
  --devices 1

# Full training
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --devices 1
```

### ETL Pipeline Execution
```bash
cd stage0
python3 scripts/make_tf_data.py  # Regenerate all timeframe data from source
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

**Dependencies**:
```bash
pip install pandas>=2.0.0 pyarrow>=12.0.0 matplotlib>=3.7.0 psutil>=5.9.0
```

**Python**: 3.11+
**Memory**: 8GB+ recommended for full dataset processing

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

**Key Files**:
- `stage1/src/data_loader.py`: Main data pipeline orchestration
- `stage1/src/model.py`: Neural network architecture (TF-CNN + Transformer encoder + decoders)
- `stage1/src/losses.py`: Multi-component loss functions
- `stage1/scripts/train_stage1.py`: PyTorch Lightning training with One-Cycle LR
- `stage1/scripts/evaluate_stage1.py`: Comprehensive evaluation with correlation/consistency metrics

## Stage 1 Commands

### Training
```bash
cd stage1
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --gpus 1
```

### Evaluation
```bash
python3 scripts/evaluate_stage1.py \
  --config configs/base.yaml \
  --ckpt checkpoints/stage1-epoch-XX.ckpt \
  --data_dir ../data/derived \
  --output evaluation_results.json
```

### Data Pipeline Testing
```bash
cd tests
python3 test_stage1_data.py
```

## Critical Rules for Stage 2 Development

When working on Stage 2 (RL Trading Agent):

1. **ALWAYS validate Stage 1 completion** before Stage 2 development
2. **USE learned weights from Stage 1** as initialization for the RL agent encoder
3. **MAINTAIN the same timeframe alignment** principles from Stage 1  
4. **RESPECT the same data constraints** from Stage 0 (no volume, Parquet only, UTC timezone)
5. **LEVERAGE the bottleneck representations** from Stage 1 as state features for RL

Stage 1 trained weights provide rich multi-timeframe representations that capture cross-scale dependencies and temporal patterns essential for trading decisions.

## 🔧 WSL Environment Compatibility

### Resolved Issues
- **MPI Detection Error**: Resolved with `PL_DISABLE_FORK=1` environment variable in `train_stage1.py`
- **TensorBoard Dependency**: Installation method clarified (`pip install tensorboard`)
- **Large Data Processing**: Handled with progressive lightweight configurations

### Recommended Execution Order
1. **Fast Test**: Use `configs/test.yaml` for initial verification
2. **Development Data**: Create small dataset with `create_dev_data.py`
3. **Progressive Scaling**: Gradually increase data size after success

### Configuration Files
- `base.yaml`: Full production training (3.5M parameters, 40 epochs)
- `test.yaml`: Lightweight test (505K parameters, 5 epochs) - **recommended first step**
- `dev.yaml`: Ultra-lightweight development (large data handling)

### Project Organization
- `old/`: Contains deprecated files and unused virtual environments
- `doc/`: Comprehensive project documentation (6 files, 1,798 lines)
- `tests/`: Unit tests for data pipeline and loss functions