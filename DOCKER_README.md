# GoriProject Docker セットアップガイド

## 🚀 クイックスタート

### 1. ローカルでのビルドとテスト

```bash
# Docker イメージのビルド
docker-compose build

# コンテナの起動
docker-compose run --rm goriproject bash

# コンテナ内での実行
cd stage1
python3 scripts/train_stage1.py \
  --config configs/t5_large_nofreeze.yaml \
  --data_dir ../data/derived \
  --devices 1
```

### 2. vast.ai へのデプロイ

```bash
# Docker Hub のユーザー名を設定
export DOCKER_USERNAME="your_dockerhub_username"

# デプロイスクリプトの実行
./scripts/deploy_vastai.sh
```

## 📋 必要な準備

1. **Docker Hub アカウント**
   - https://hub.docker.com でアカウント作成
   - `docker login` でログイン

2. **vast.ai アカウント**
   - https://vast.ai でアカウント作成
   - クレジット購入

## 🎯 vast.ai 推奨スペック

- **GPU**: RTX 4090 または A100
- **RAM**: 32GB 以上
- **ディスク**: 50GB 以上
- **CUDA**: 11.8

## 📦 ファイル構成

```
GoriProject/
├── Dockerfile          # Docker イメージ定義
├── docker-compose.yaml # ローカル実行用
├── requirements.txt    # Python パッケージ
├── .dockerignore      # Docker ビルド除外ファイル
└── scripts/
    └── deploy_vastai.sh # vast.ai デプロイスクリプト
```

## 🔧 カスタマイズ

### データの扱い

大容量データの場合、2つの選択肢があります：

1. **Docker イメージに含める**（デフォルト）
   - 簡単だが、イメージサイズが大きくなる

2. **vast.ai の永続ストレージを使用**
   - `.dockerignore` に `data/` を追加
   - vast.ai でストレージをマウント

### GPU メモリ不足時の対処

```yaml
# configs/t5_large_nofreeze.yaml で調整
training:
  batch_size: 16  # 32 → 16 に削減
  accumulate_grad_batches: 16  # 8 → 16 に増加
```

## 💡 トラブルシューティング

### CUDA バージョンの不一致
```dockerfile
# Dockerfile の FROM を変更
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04  # 必要に応じて
```

### メモリ不足
- batch_size を削減
- num_workers を削減
- 不要なプロセスを停止

### 権限エラー
```bash
# vast.ai 内で実行
chmod -R 777 /workspace/logs
chmod -R 777 /workspace/checkpoints
```