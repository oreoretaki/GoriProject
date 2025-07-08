#!/usr/bin/env python3
"""
T5モデル事前ダウンロードスクリプト（汎用CLI版）
オフライン環境・プロキシ環境対応
"""

import argparse
import os
import sys
from pathlib import Path

# 社内プロキシ対応
os.environ.setdefault("HTTPS_PROXY", os.getenv("HTTPS_PROXY", ""))

try:
    from huggingface_hub import snapshot_download
    from transformers import T5Config, T5EncoderModel
except ImportError as e:
    print(f"❌ 必要なライブラリがインストールされていません: {e}")
    print("インストール: pip install transformers huggingface_hub")
    sys.exit(1)

def download_t5_model(model_name="t5-small", cache_dir=None, token=None, local_only=False):
    """T5モデルをローカルにダウンロード"""
    
    print(f"🤗 T5モデル処理中: {model_name}")
    print(f"   キャッシュディレクトリ: {cache_dir}")
    print(f"   オフラインモード: {local_only}")
    
    try:
        if local_only:
            # ―― Offline fallback ――
            print("   モード: ローカルキャッシュのみ")
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )
        else:
            # ―― Online download ――
            print("   モード: オンラインダウンロード")
            local_path = snapshot_download(
                repo_id=model_name,
                cache_dir=cache_dir,
                token=token,
                resume_download=True
            )
        
        print(f"✅ ダウンロード/ロード完了")
        print(f"   パス: {local_path}")
        
        # 動作確認（正しいPyTorch重みロード）
        print(f"🔍 モデル検証中...")
        
        # T5フルモデルで検証（60M parameters確認）
        from transformers import T5ForConditionalGeneration
        full_model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_only
        )
        full_params = full_model.num_parameters()
        
        # エンコーダーのみもロード
        encoder_model = T5EncoderModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=local_only
        )
        encoder_params = sum(p.numel() for p in encoder_model.parameters())
        
        # パラメータ数検証
        if model_name == "t5-small":
            expected_min, expected_max = 59_000_000, 62_000_000
            assert expected_min < full_params < expected_max, \
                f"T5-small param count mismatch: {full_params:,} (expected ~60M)"
        elif model_name == "t5-base":
            expected_min, expected_max = 220_000_000, 225_000_000
            assert expected_min < full_params < expected_max, \
                f"T5-base param count mismatch: {full_params:,} (expected ~223M)"
        elif model_name == "t5-large":
            expected_min, expected_max = 735_000_000, 740_000_000
            assert expected_min < full_params < expected_max, \
                f"T5-large param count mismatch: {full_params:,} (expected ~738M)"
        
        config = T5Config.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_only)
        
        print(f"✅ Model '{model_name}' ready @ {cache_dir}")
        print(f"   d_model: {config.d_model}")
        print(f"   params (full): {full_params/1e6:.1f} M")
        print(f"   params (encoder): {encoder_params/1e6:.1f} M")
        
        # パラメータ数確認
        if model_name == "t5-small" and full_params > 59_000_000:
            print(f"   ✅ 正しいPyTorch重み（T5-small）ロード成功")
        elif model_name == "t5-base" and full_params > 220_000_000:
            print(f"   ✅ 正しいPyTorch重み（T5-base）ロード成功")
        elif model_name == "t5-large" and full_params > 735_000_000:
            print(f"   ✅ 正しいPyTorch重み（T5-large）ロード成功")
        elif full_params < 10_000_000:
            print(f"   ⚠️ WARN: パラメータ数が少ない（ONNX版の可能性）")
        else:
            print(f"   ✅ モデルロード成功")
        
        return local_path
        
    except Exception as e:
        print(f"❌ download failed: {e}")
        print("エラータイプ:", type(e).__name__)
        
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("• 認証エラー → huggingface-cli login で解決")
        elif "404" in str(e) or "not found" in str(e).lower():
            print(f"• モデル '{model_name}' が見つかりません")
            print("• 正しいモデルID: t5-small, t5-base, flan-t5-small など")
        elif "connection" in str(e).lower() or "network" in str(e).lower():
            print("• ネット接続 / プロキシ / token を確認してください")
            print(f"• プロキシ設定: export HTTPS_PROXY=http://proxy:8080")
        
        print(f"\n手動ダウンロード手順:")
        print(f"1. https://huggingface.co/{model_name} から 'Download model'")
        print(f"2. 解凍して {cache_dir}/hub/models--{model_name.replace('/', '--')} に配置")
        print(f"3. python download_t5.py --model {model_name} --local_only で検証")
        
        return None

def main():
    parser = argparse.ArgumentParser(description="T5モデルダウンローダー（汎用CLI版）")
    parser.add_argument("--model", default="t5-base", 
                       help="モデルID（デフォルト: t5-base）。選択肢: t5-small, t5-base, t5-large")
    parser.add_argument("--cache_dir", default=os.getenv("HF_HOME", "~/.cache/huggingface"), 
                       help="キャッシュディレクトリ")
    parser.add_argument("--token", default=None, help="HF token if repo is private")
    parser.add_argument("--local_only", action="store_true",
                       help="Skip network, expect model to exist in cache_dir")
    args = parser.parse_args()
    
    print("🚀 T5転移学習用モデルダウンロード（汎用CLI版）")
    
    # パス展開
    cache_dir = os.path.expanduser(args.cache_dir)
    
    # ダウンロード実行
    result = download_t5_model(
        model_name=args.model,
        cache_dir=cache_dir,
        token=args.token,
        local_only=args.local_only
    )
    
    if result:
        print(f"\n🎉 準備完了！これでT5転移学習実験を実行できます:")
        print(f"   python3 scripts/train_stage1.py --config configs/t5_freeze.yaml --data_dir ../data/derived --fast_dev_run")
    else:
        print(f"\n❌ ダウンロードに失敗しました。")
        sys.exit(1)

if __name__ == "__main__":
    main()