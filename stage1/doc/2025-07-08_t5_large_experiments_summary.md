# T5-Large実験サマリー (2025年7月8日)

## 概要
本日はT5-Largeモデル（738Mパラメータ）の実装と、データ量5倍増加に伴う最適化を実施しました。

## 主要な成果

### 1. T5-Large NaN問題の解決
- **問題**: T5-LargeでAMP（自動混合精度）使用時にNaN発生
- **原因**: AMPのスケーリングによる数値不安定性
- **解決**: precision="32"（FP32固定）により完全解決
- **重要な洞察**: FP16時の巨大な勾配ノルム（数万）は「スケール前」の値で、FP32の0.x台が真の勾配ノルム

### 2. Version 56 実験結果（T5-Large + FP32）

#### 設定
- モデル: T5-Large (738M parameters)
- データ量: 2%（970 train batches）
- 学習率: t5_lr_top=5.0e-6, base_lr=2.0e-5
- gradient_clip: 0.008
- precision: "32"（FP32）
- エポック数: 6（実行）

#### 最終結果（6エポック後）
```
TF別相関値:
M1:  +0.002563 ✅
M5:  -0.007802 ❌
M15: +0.021975 🔥
M30: +0.035845 🔥
H1:  +0.033896 🔥
H4:  -0.002577 ⚠️

平均相関: +0.008066
```

#### 主要な改善
- **H1**: -0.025368 → +0.033896（+0.059265改善）
- **M30**: -0.003734 → +0.035845（+0.039579改善）
- **成功率**: 83%（5/6 TFが改善）

### 3. データ量5倍増加への対応

#### 問題発生（Version 57）
- データ量を5倍（2%→10%）に増加
- 学習率・ウォームアップ未調整で開始
- 結果: 全TFで負の相関、勾配ノルム極小（0.02-0.04）

#### 原因分析
1. 総ステップ数5倍だがウォームアップ期間据え置き → 実質的にウォームアップが長すぎる
2. 相対的に学習率が小さくなった
3. 勾配クリップが厳しすぎる

#### 対策（Version 58用）
```yaml
# 学習率1.5倍スケーリング
t5_lr_top: 5.0e-6 → 7.5e-6
optimizer.lr: 2.0e-5 → 3.0e-5
scheduler.max_lr: 2.5e-5 → 3.75e-5

# その他調整
gradient_clip: 0.008 → 0.012
warmup_epochs: 3 → 1（全体の約5%）
```

## 設定ファイルの最終状態

### shared_base.yaml
```yaml
training:
  gradient_clip: 0.012        # 5倍データ対応で緩和
  precision: "32"             # NaN対策：AMP無効化、FP32固定
  
  optimizer:
    lr: 3.0e-5                # 5倍データ対応で1.5倍
    
  layerwise_lr_decay: 0.90
  t5_lr_top: 7.5e-6           # 5倍データ対応で1.5倍
  
  scheduler:
    max_lr: 3.75e-5           # 5倍データ対応で1.5倍
```

### t5_large_nofreeze.yaml
```yaml
training:
  epochs: 7
  warmup_epochs: 1            # 5倍データ用に短縮
  
development:
  limit_train_batches: 4850   # 約10%（5倍増加）
  limit_val_batches: 970      # 約10%
```

## Docker環境の準備

### 作成ファイル
1. **Dockerfile**: CUDA 11.8対応、Python 3.11
2. **requirements.txt**: PyTorch 2.4.1 (CUDA 11.8版)
3. **docker-compose.yaml**: ローカルテスト用
4. **deploy_vastai.sh**: vast.aiデプロイスクリプト
5. **.dockerignore**: ビルド最適化
6. **DOCKER_README.md**: セットアップガイド

### 課題
- WSL2でdocker-compose未インストール
- Docker Desktop for Windowsのインストールが必要

## 次回の作業項目

1. **Version 58の結果確認**
   - 新設定（学習率1.5倍、warmup短縮）の効果測定
   - 3エポック目以降の相関値改善確認

2. **Docker環境セットアップ**
   - Docker Desktop for Windowsインストール
   - vast.aiへのデプロイ実行

3. **更なる最適化検討**
   - マスク率調整（0.15→0.10）
   - span_max調整（10→6）
   - M5の改善策検討

## 重要な学習事項

1. **データ量増加時は必ず学習率・ウォームアップを再スケール**
2. **FP32使用時の勾配ノルム0.x台は正常（AMPスケーリングなし）**
3. **T5-Largeは中長期足（M15/M30/H1）で特に優秀な性能**
4. **短期足（M1/M5）の学習は依然として課題**

## 実行中のプロセス
- Version 57: 古い設定で実行中（停止推奨）
- Version 58: 新設定で実行中（継続監視）