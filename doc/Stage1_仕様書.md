# Stage 1 — 自己教師ありマルチTF再構築仕様書

## スコープ  
USD/JPYのOHLCデータについて、6つの整列した時間足（M1, M5, M15, H1, H4, D）のマスクされたスパンを同時に再構築し、クロススケール一貫性を強制する単一のエンコーダー・デコーダーを訓練する。学習された重みは、後続のRL取引エージェントのシードとして使用される。

## 1. データセット                    

| TF  | ファイル | レコード数 | 頻度 | 期間 |
|-----|----------|------------|------|------|
| M1  | simple_gap_aware_m1.parquet  | 3,104,383 | 1分 | 2017-01-03 08:00 → 2025-06-17 23:24 UTC |
| M5  | simple_gap_aware_m5.parquet  | 630,474   | 5分 | derived |
| M15 | simple_gap_aware_m15.parquet | 210,477   | 15分 | derived |
| M30 | simple_gap_aware_m30.parquet | 105,252   | 30分 | derived |
| H1  | simple_gap_aware_h1.parquet  | 52,632    | 1時間 | derived |
| H4  | simple_gap_aware_h4.parquet  | 13,601    | 4時間 | derived |
| D   | simple_gap_aware_d.parquet   | 2,640     | 1日 | derived |

すべてのファイルはタイムゾーン対応のUTCインデックスを共有し、100%のTF整合性を持つ。

## 2. 入力パイプライン

**ウィンドウサンプラー**: seq_len = 200 @ M1（≈ 3.3時間）;  
同じカレンダーウィンドウがすべてのTFからスライスされ、シーケンスが右端で整列する。

**特徴量スタック**: [open, high, low, close, Δclose, %body] → 6特徴量 × 6TF → 36チャンネル

**マスキング戦略**:  
- ランダムな連続ブロック（span = 5-60 M1バー）
- TFごとに15%のトークンをマスク、TF間で同期
  
**正規化**: TFごとにz-score; 統計はstats.jsonに保存

## 3. モデルアーキテクチャ

```
┌──────────────┐   ┌──────────────────────────┐
│  TF固有の    │   │  共有クロススケール      │   ┌───────────┐
│  畳み込み    │→  │  Mamba-Flash Encoder     │→ │ Bottleneck │→ TF別デコーダー
└──────────────┘   └──────────────────────────┘   └───────────┘
```

**TF固有ステム**: 1次元depth-wise CNN（kernel = 3）で各TFをd_model = 128に投影

**クロススケールミキサー**: 階層的Mamba-TSブロック + 2層ごとにゲート付きクロススケール注意（FlashAttn-2）

**Bottleneck**: ストライド畳み込みでlatent_len = seq_len/4、グローバルコンテキスト用

**デコーダー**: TFごとに軽量な転置畳み込み + 線形ヘッド

**位置エンコーディング**: TF内用学習済みrotary; TF間用相対時間フレームID埋め込み

## 4. 損失関数

| シンボル | 説明 | 重み |
|----------|------|------|
| L_recon_tf | TFごとのOHLCに対するHuber損失 | 0.6 |
| L_spec_tf | マルチ解像度STFT損失 | 0.2 |
| L_cross | デコードされたTFとaggregation(M1)間のMSE | 0.15 |
| L_amp_phase | 振幅・位相相関 | 0.05 |

総損失: L = Σ_tf w1·L_recon_tf + w2·L_spec_tf + L_cross + L_amp_phase

## 5. 訓練スケジュール

| パラメータ | 値 |
|------------|-----|
| バッチ | 24ウィンドウ（≈ 24·200 = 4,800 M1バー） |
| エポック | 40（早期停止: Δcorr < 0.001 / 5エポック） |
| オプティマイザー | AdamW β=(0.9,0.98), weight_decay 1e-2 |
| 学習率 | One-Cycle: 1.5e-4 → 5e-4 → 1e-5 |
| 勾配スケール | bf16 + DeepSpeed ZeRO-2 |
| データ拡張 | DDIMノイズ, time-warp (p = 0.2), regime-mix (p = 0.1) |

## 6. メトリクス・検証

**Correlation@TF**: 再構築と正解との間のピアソン相関

**整合性比率**: TFバーが集約と正確に一致する割合

**スペクトラムΔ**: 対数パワースペクトラム（0-π）の平均絶対誤差

**CIパイプライン**: run_validate.pyを拡張してすべてのメトリクスを夜間計算

## 7. ディレクトリ・ファイル構成

```
GoriProject/
├─ data/derived/            # ← Stage 0出力
│   └─ simple_gap_aware_{m1,m5,…}.parquet
├─ stage1/
│   ├─ configs/
│   │   └─ base.yaml        # ハイパーパラメータ（OmegaConf）
│   ├─ scripts/
│   │   ├─ train_stage1.py
│   │   └─ evaluate_stage1.py
│   └─ checkpoints/
└─ tests/
    └─ test_stage1_data.py  # ウィンドウサンプラー、整列
```

## 8. 再現手順（最小限）

```bash
# 1️⃣ 環境作成（CUDA 12）
conda env create -f env_stage1.yml
conda activate gori_stage1

# 2️⃣ 訓練
python stage1/scripts/train_stage1.py \
  --config stage1/configs/base.yaml \
  --data_dir data/derived

# 3️⃣ 夜間CI
pytest -q tests/test_stage1_data.py
python stage1/scripts/evaluate_stage1.py --ckpt best.pth
```

## 注記・未解決の質問

**タイムゾーン健全性** — すべてのタイムスタンプはUTC; デコーダー出力は相対オフセットを使用するため、絶対tz不一致は漏洩しない。

**長いギャップ** — 補間の代わりにマスクトークンとして保持 → モデルが「不確実性ポケット」を学習。

**未来情報漏洩** — Mambaブロックの因果マスキングで保護。

詳細や修正が必要なセクションがあればお知らせください。🎯