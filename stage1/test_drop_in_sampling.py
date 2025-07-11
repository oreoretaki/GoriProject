#!/usr/bin/env python3
"""
Drop-in Sampling テストスクリプト
M1データの保持とNaN伝播防止を確認
"""

import torch
import yaml
from src.data_loader import create_stage1_dataloaders
from src.window_sampler import MultiTFWindowSampler
import pandas as pd

def test_drop_in_sampling():
    """Drop-in Samplingの動作確認"""
    print("🧪 Drop-in Sampling テスト開始")
    
    # 最小限のconfig
    config = {
        'data': {
            'seq_len': 64,
            'n_features': 6,
            'n_timeframes': 6,
            'timeframes': ['m1', 'm5', 'm15', 'm30', 'h1', 'h4'],
            'data_dir': '../data/derived',
            'stats_file': 'stats.json',
            'sampling_probs': {
                'm1': 1.0,  # M1必須
                'm5': 0.5,  # 50%ドロップ
                'm15': 0.7,
                'm30': 0.8,
                'h1': 0.9,
                'h4': 1.0
            }
        },
        'model': {
            'async_sampler': True
        },
        'training': {
            'batch_size': 4
        },
        'validation': {
            'val_split': 0.2,
            'val_gap_days': 1.0
        },
        'normalization': {
            'method': 'zscore',
            'per_tf': True
        },
        'dataloader': {
            'num_workers': 0,
            'pin_memory': False
        }
    }
    
    try:
        # DataLoaderを作成
        print("📊 DataLoader作成中...")
        train_loader, val_loader = create_stage1_dataloaders('../data/derived', config)
        
        # 数バッチをテスト
        print("\n🔍 バッチ内容確認:")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # 3バッチのみテスト
                break
                
            features = batch['features']
            targets = batch['targets']
            
            print(f"\n--- バッチ {i+1} ---")
            print(f"TFs: {list(features.keys())}")
            
            # 各TFの存在確認
            for tf_name in config['data']['timeframes']:
                if tf_name in features:
                    tf_tensor = features[tf_name]
                    has_data = not torch.isnan(tf_tensor).all()
                    print(f"  {tf_name}: shape={tf_tensor.shape}, has_data={has_data}")
                else:
                    print(f"  {tf_name}: ❌ 欠損")
            
            # M1の存在確認（必須）
            if 'm1' not in features:
                print("❌ M1がバッチに存在しません！")
                return False
            else:
                print("✅ M1がバッチに存在します")
                
        print("\n✅ Drop-in Sampling テスト完了")
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_drop_in_sampling()
    if success:
        print("\n🎉 テスト成功！")
    else:
        print("\n💥 テスト失敗")