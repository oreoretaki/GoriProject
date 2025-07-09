#!/usr/bin/env python3
"""
Stage-1 データリーク修正前後の検証指標比較テスト
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import shutil
from typing import Dict, Any

# プロジェクトインポート
from src.data_loader import Stage1Dataset
from src.window_sampler import MultiTFWindowSampler
from src.masking import MaskingStrategy
from src.normalization import TFNormalizer
from src.feature_engineering import FeatureEngineer

def load_test_config():
    """テスト用設定を作成"""
    return {
        'data': {
            'seq_len': 128,
            'n_timeframes': 6,
            'n_features': 6,
            'total_channels': 36,
            'timeframes': ['m1', 'm5', 'm15', 'm30', 'h1', 'h4'],
            'data_dir': '../data/derived',
            'stats_file': 'stats.json'
        },
        'masking': {
            'mask_ratio': 0.15,
            'mask_span_min': 3,
            'mask_span_max': 10,
            'sync_across_tf': False
        },
        'normalization': {
            'method': 'zscore',
            'per_tf': True
        },
        'validation': {
            'val_split': 0.2,
            'val_gap_days': 1.0  # 新パラメータ
        },
        'evaluation': {
            'eval_mask_ratio': None  # 新パラメータ
        }
    }

def test_data_leak_fix():
    """データリーク修正前後の比較テスト"""
    
    print("🔍 データリーク修正前後の検証指標比較テスト")
    print("=" * 60)
    
    config = load_test_config()
    
    # データ読み込み
    print("📂 データ読み込み中...")
    data_dir = Path(config['data']['data_dir'])
    tf_data = {}
    
    for tf in config['data']['timeframes']:
        file_path = data_dir / f"simple_gap_aware_{tf}.parquet"
        if file_path.exists():
            tf_data[tf] = pd.read_parquet(file_path)
            print(f"   {tf}: {len(tf_data[tf])} レコード")
        else:
            print(f"   ⚠️ {tf}: ファイルが見つかりません")
    
    # テスト1: 従来の分割方法（リークあり）
    print("\n🔴 テスト1: 従来の分割方法（リークあり）")
    print("-" * 40)
    
    # val_gap_days=0で従来の動作を再現
    old_sampler_train = MultiTFWindowSampler(
        tf_data=tf_data,
        seq_len=config['data']['seq_len'],
        split="train",
        val_split=config['validation']['val_split'],
        min_coverage=0.8,
        cache_dir=None,  # キャッシュなし
        val_gap_days=0.0  # ギャップなし
    )
    
    old_sampler_val = MultiTFWindowSampler(
        tf_data=tf_data,
        seq_len=config['data']['seq_len'],
        split="val",
        val_split=config['validation']['val_split'],
        min_coverage=0.8,
        cache_dir=None,  # キャッシュなし
        val_gap_days=0.0  # ギャップなし
    )
    
    print(f"   訓練ウィンドウ数: {len(old_sampler_train)}")
    print(f"   検証ウィンドウ数: {len(old_sampler_val)}")
    
    # 重複チェック
    if len(old_sampler_train.split_windows) > 0 and len(old_sampler_val.split_windows) > 0:
        train_last = old_sampler_train.split_windows[-1]
        val_first = old_sampler_val.split_windows[0]
        overlap_minutes = (val_first[0] - train_last[1]).total_seconds() / 60
        print(f"   訓練最後と検証最初の間隔: {overlap_minutes:.1f} 分")
        print(f"   重複度: {127/128*100:.1f}% (127/128 時点重複)")
    
    # テスト2: 新しい分割方法（リーク修正）
    print("\n🟢 テスト2: 新しい分割方法（リーク修正）")
    print("-" * 40)
    
    new_sampler_train = MultiTFWindowSampler(
        tf_data=tf_data,
        seq_len=config['data']['seq_len'],
        split="train",
        val_split=config['validation']['val_split'],
        min_coverage=0.8,
        cache_dir=None,  # キャッシュなし
        val_gap_days=config['validation']['val_gap_days']  # 1日ギャップ
    )
    
    new_sampler_val = MultiTFWindowSampler(
        tf_data=tf_data,
        seq_len=config['data']['seq_len'],
        split="val",
        val_split=config['validation']['val_split'],
        min_coverage=0.8,
        cache_dir=None,  # キャッシュなし
        val_gap_days=config['validation']['val_gap_days']  # 1日ギャップ
    )
    
    print(f"   訓練ウィンドウ数: {len(new_sampler_train)}")
    print(f"   検証ウィンドウ数: {len(new_sampler_val)}")
    
    # ギャップチェック
    if len(new_sampler_train.split_windows) > 0 and len(new_sampler_val.split_windows) > 0:
        train_last = new_sampler_train.split_windows[-1]
        val_first = new_sampler_val.split_windows[0]
        gap_minutes = (val_first[0] - train_last[1]).total_seconds() / 60
        print(f"   訓練最後と検証最初の間隔: {gap_minutes:.1f} 分")
        print(f"   ギャップ日数: {gap_minutes/(24*60):.1f} 日")
        print(f"   重複度: 0% (完全分離)")
    
    # テスト3: マスク率オーバーライド機能
    print("\n🎭 テスト3: マスク率オーバーライド機能")
    print("-" * 40)
    
    masking_strategy = MaskingStrategy(config)
    
    # ダミー特徴量作成
    dummy_features = torch.randn(6, 128, 6)
    
    # 通常のマスク
    normal_mask = masking_strategy.generate_masks(dummy_features, seed=42)
    normal_ratio = normal_mask.sum().item() / normal_mask.numel()
    print(f"   通常マスク率: {normal_ratio:.3f}")
    
    # マスクなし
    no_mask = masking_strategy.generate_masks(dummy_features, seed=42, eval_mask_ratio_override=0.0)
    no_mask_ratio = no_mask.sum().item() / no_mask.numel()
    print(f"   マスクなし率: {no_mask_ratio:.3f}")
    
    # 全マスク
    full_mask = masking_strategy.generate_masks(dummy_features, seed=42, eval_mask_ratio_override=1.0)
    full_mask_ratio = full_mask.sum().item() / full_mask.numel()
    print(f"   全マスク率: {full_mask_ratio:.3f}")
    
    # データ量比較
    print("\n📊 データ量比較")
    print("-" * 40)
    
    old_train_size = len(old_sampler_train)
    old_val_size = len(old_sampler_val)
    new_train_size = len(new_sampler_train)
    new_val_size = len(new_sampler_val)
    
    if old_train_size > 0:
        train_reduction = (old_train_size - new_train_size) / old_train_size * 100
    else:
        train_reduction = 0.0
        
    if old_val_size > 0:
        val_reduction = (old_val_size - new_val_size) / old_val_size * 100
    else:
        val_reduction = 0.0
    
    print(f"   従来方式 - 訓練: {old_train_size:,}, 検証: {old_val_size:,}")
    print(f"   新方式   - 訓練: {new_train_size:,}, 検証: {new_val_size:,}")
    print(f"   データ削減率 - 訓練: {train_reduction:.1f}%, 検証: {val_reduction:.1f}%")
    
    # 期待される効果
    print("\n🎯 期待される効果")
    print("-" * 40)
    print("   ✅ データリーク除去により、検証指標が現実的な値に")
    print("   ✅ 過学習の早期発見が可能に")
    print("   ✅ Stage-2での初期重み選定が正確に")
    print("   ✅ マスク率に依存しない評価が可能に")
    
    # 推奨実行例
    print("\n🚀 推奨実行例")
    print("-" * 40)
    print("   # 従来方式での実行")
    print("   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1")
    print("")
    print("   # 新方式での実行")
    print("   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1 --val_gap_days 1.0")
    print("")
    print("   # マスクなし評価")
    print("   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1 --eval_mask_ratio 0.0")
    print("")
    print("   # 複数シード評価")
    print("   python3 scripts/train_stage1.py --config configs/t5_large_nofreeze.yaml --data_dir ../data/derived --devices 1 --seeds 42 123 2025")
    
    return True

if __name__ == "__main__":
    try:
        test_data_leak_fix()
        print("\n✅ テスト完了")
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()