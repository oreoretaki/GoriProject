# GoriProject - USD/JPY機械学習プロジェクト

## 📁 プロジェクト構成

```
gori_project/
├── data/                    # 共有データディレクトリ
│   ├── oanda_historical.db  # 元SQLiteDB（参考用）
│   └── derived/             # 高品質Parquetデータ
├── doc/                     # 📚 包括的プロジェクトドキュメント
│   ├── Stage1_仕様書.md      # 要件・技術仕様
│   ├── ARCHITECTURE.md      # 設計思想・アーキテクチャ
│   ├── DEVELOPER_GUIDE.md   # 開発・実装ガイド
│   ├── API_REFERENCE.md     # 全クラス・関数仕様
│   ├── CODE_REVIEW_REPORT.md # 品質チェック結果
│   └── INDEX.md             # ドキュメント索引
├── old/                     # 🗑️ 不要ファイル保管庫
├── stage0/ ✅ 完了          # データ前処理基盤
├── stage1/ ✅ 完了          # 自己教師ありマルチTF再構築
├── tests/                   # 🧪 テストファイル
└── README.md               # このファイル
```

## 🎯 各ステージ状況

### Stage 0 - データ前処理基盤 ✅ 完了
- **成果**: M1→全TF 100%整合性データ生成
- **品質**: Volume除外、OHLC検証、Gap対応済み
- **検証**: 0.8秒で自動検証可能
- **CI/CD**: GitHub Actions完備

**詳細**: [stage0/README.md](stage0/README.md)

### Stage 1 - 自己教師ありマルチTF再構築 ✅ 実装完了
- **成果**: 6TF同時再構築モデル実装
- **技術**: TF固有CNN + 共有エンコーダー + 4種損失関数
- **品質**: マスキング戦略、クロススケール一貫性、包括的テスト

**詳細**: [stage1/README.md](stage1/README.md)  
**ドキュメント**: [doc/INDEX.md](doc/INDEX.md)

## 🚀 クイックスタート

### Stage 0 検証（完了確認）
```bash
cd stage0
python3 scripts/run_validate.py
# 期待結果: 0.8秒で100%整合性確認
```

### Stage 1 実行
```bash
cd stage1

# 高速テスト実行（推奨）
python3 scripts/train_stage1.py \
  --config configs/test.yaml \
  --data_dir ../data/derived \
  --fast_dev_run \
  --devices 1

# 本格訓練実行
python3 scripts/train_stage1.py \
  --config configs/base.yaml \
  --data_dir ../data/derived \
  --devices 1
```

## 📊 データ概要

**期間**: 2017年1月3日 ～ 2025年6月17日（約8.5年）  
**通貨ペア**: USD/JPY  
**データソース**: OANDA  
**品質**: Stage 0で100%整合性保証済み  

## 🔐 重要制約

- ❌ **Volume使用絶対禁止**（ティック数≠取引量）
- ❌ **元SQLiteテーブル使用禁止**（0.3%整合性）
- ✅ **Parquetデータのみ使用**（100%整合性）