# 継続作業メモ (2025年7月8日終了時点)

## 現在の状況

### 実行中の実験
- **Version 58**: 新設定（学習率1.5倍、warmup短縮、データ5倍）で実行中
- **監視項目**: 
  - 3エポック目以降の相関値改善
  - 勾配ノルムの活性化（0.02台から0.2-0.5台への回復）
  - 全TFでの負相関からの脱却

### 最新の設定変更
```bash
# 確認コマンド
cd /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1
python3 -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('logs/stage1/version_58', size_guidance={'scalars': 10000})
ea.Reload()
# 最新の結果を確認
"
```

## 直近の最重要タスク

### 1. Version 58の結果分析
- [ ] 3エポック目の相関値確認（負値から正値への転換期待）
- [ ] T5勾配ノルムの活性化確認（目標: 0.2-0.5台）
- [ ] 全TF別相関の改善度測定

### 2. Docker環境完成
- [ ] Docker Desktop for Windowsインストール
- [ ] WSL2統合設定
- [ ] ローカルビルド&テスト実行
- [ ] Docker Hubアカウント設定
- [ ] vast.aiデプロイ

### 3. 設定の最終調整（必要に応じて）
- [ ] マスク率調整（mask_ratio: 0.15→0.10）
- [ ] スパン調整（span_max: 10→6）
- [ ] M5特化対策の検討

## ファイル構造メモ

### 重要な設定ファイル
```
stage1/configs/
├── shared_base.yaml          # 基本設定（学習率1.5倍済み）
├── t5_large_nofreeze.yaml    # T5-Large設定（データ5倍、エポック7）
└── t5_nofreeze_10ep.yaml     # T5-Base参考用

stage1/logs/stage1/
├── version_56/               # T5-Large成功版（2%データ）
├── version_57/               # 5倍データ+旧設定（失敗）
└── version_58/               # 5倍データ+新設定（実行中）
```

### Docker関連ファイル
```
GoriProject/
├── Dockerfile                # CUDA 11.8, Python 3.11
├── docker-compose.yaml       # ローカル実行用
├── requirements.txt          # PyTorch 2.4.1+cu118
├── .dockerignore             # ビルド最適化
├── scripts/deploy_vastai.sh  # デプロイスクリプト
└── DOCKER_README.md          # セットアップガイド
```

## 成功指標

### Version 58で期待する結果
```
目標（3-4エポック後）:
M1:  ±0.000 以上
M5:  -0.005 以上（改善）
M15: +0.015 以上
M30: +0.025 以上
H1:  +0.015 以上
H4:  ±0.000 以上

平均相関: +0.008 以上
勾配ノルム: 0.2-0.5台
```

## 問題発生時の対処

### Version 58も失敗した場合
1. データ量を10%→5%に減少
2. マスク戦略の調整
3. より大きな学習率増加（2倍）
4. gradient_clipをさらに緩和

### Docker問題の対処
1. `docker compose` コマンド使用
2. WSL内直接インストール
3. Windows側でビルド→転送

## 引き継ぎ事項

1. **Version 58は継続監視** - 停止しないこと
2. **Docker環境は優先度高** - vast.ai移行のため
3. **successful設定はversion_56** - 参考基準として保持
4. **全ログファイルを保持** - 実験の連続性のため

---

**次回開始時の最初のコマンド:**
```bash
cd /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1
# Version 58の最新状況確認
```