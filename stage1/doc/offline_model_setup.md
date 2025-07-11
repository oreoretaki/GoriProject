# オフライン環境でのT5モデルセットアップ

## 手動ダウンロード手順

### 1. オンライン環境でのダウンロード

```bash
# 通常のオンライン環境
python3 scripts/download_t5.py --model t5-small

# プロキシ環境
HTTPS_PROXY=http://proxy.example.com:8080 python3 scripts/download_t5.py --model t5-small

# 認証が必要な場合
huggingface-cli login
python3 scripts/download_t5.py --model t5-small --token $HF_TOKEN
```

### 2. 手動ダウンロード（Webブラウザ経由）

1. https://huggingface.co/t5-small にアクセス
2. "Download model" をクリック → `t5-small.tar.gz` をダウンロード
3. ローカルに解凍:

```bash
# ダウンロードファイルを解凍
tar -xzf t5-small.tar.gz

# HuggingFaceキャッシュディレクトリに配置
mkdir -p ~/.cache/huggingface/hub/models--t5-small
cp -r t5-small/* ~/.cache/huggingface/hub/models--t5-small/

# 検証
python3 scripts/download_t5.py --model t5-small --local_only
```

### 3. 完全オフライン環境での検証

```bash
# ネットワークを無効化した状態で確認
python3 scripts/download_t5.py --model t5-small --local_only
```

成功すると以下が表示される:
```
✅ Model 't5-small' ready @ ~/.cache/huggingface
   d_model: 512
   params (full): 60.5 M
   params (encoder): 35.3 M
   ✅ 正しいPyTorch重み（フルモデル）ロード成功
```

## トラブルシューティング

| エラー | 原因 | 解決方法 |
|--------|------|----------|
| `401 Unauthorized` | 認証エラー | `huggingface-cli login` |
| `ConnectionError` | プロキシ・ネットワーク | `export HTTPS_PROXY=http://proxy:8080` |
| `404 Not Found` | モデルID誤り | `t5-small` を使用（`google/t5-small` ❌） |
| `OSError: Can't load` | ローカルファイル不備 | 手動ダウンロード → 配置確認 |
| `param count mismatch` | ONNX版・部分ロード | フルモデル再ダウンロード確認 |

## 代替モデル

T5-smallが利用できない場合の代替案:

```bash
# より軽量なモデル
python3 scripts/download_t5.py --model distilbert-base-uncased

# 他のT5バリエーション
python3 scripts/download_t5.py --model t5-base
python3 scripts/download_t5.py --model flan-t5-small
```

## 設定ファイル修正

代替モデルを使用する場合は設定ファイルを修正:

```yaml
# configs/t5_freeze.yaml
transfer_learning:
  lm_name_or_path: "distilbert-base-uncased"  # 代替モデル
```