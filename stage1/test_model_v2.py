#!/usr/bin/env python3
"""
Model v2 テストスクリプト
完全チェックリストの修正が正しく動作するかを確認
"""

import os
import sys
from pathlib import Path

# プロジェクトルートをPATHに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

import torch
import numpy as np
from src.data_loader import create_stage1_dataloaders, collate_multiscale
from src.model import Stage1Model
from src.losses import Stage1CombinedLoss
from src.masking import MaskingStrategy

# 簡単なテスト設定
test_config = {
    'data': {
        'timeframes': ['m1', 'm5', 'm15', 'h1'],
        'n_timeframes': 4,
        'seq_len': 64,  # 小さくしてテスト高速化
        'n_features': 6,
        'data_dir': '../data/derived',
        'stats_file': 'normalization_stats.json'
    },
    'model': {
        'async_sampler': True,  # Model v2テスト
        'cross_pairs': [["h1", "m1"], ["m15", "m1"]],
        'tf_stem': {
            'kernel_size': 3,
            'd_model': 128  # 小さくしてテスト高速化
        },
        'encoder': {
            'n_layers': 2,
            'd_model': 128,
            'cross_attn_every': 1
        },
        'bottleneck': {
            'latent_len': 8,
            'stride': 4
        },
        'decoder': {
            'n_layers': 2
        }
    },
    'masking': {
        'mask_ratio': 0.15,
        'mask_span_min': 2,
        'mask_span_max': 8,
        'sync_across_tf': True
    },
    'loss': {
        'weights': {
            'recon_tf': 1.0,
            'spec_tf': 0.1,
            'cross': 0.1,
            'amp_phase': 0.05
        },
        'huber_delta': 1.0,
        'stft_scales': [64, 128]
    },
    'training': {
        'batch_size': 4  # 小さくしてテスト高速化
    },
    'validation': {
        'val_split': 0.2,
        'val_gap_days': 1.0
    },
    'evaluation': {
        'eval_mask_ratio': 0.15
    },
    'dataloader': {
        'num_workers': 0,  # テスト用に0
        'pin_memory': False
    }
}

def test_dict_collate():
    """collate_multiscale関数のテスト"""
    print("🧪 Testing collate_multiscale...")
    
    # サンプルバッチデータ（Dict形式）
    batch = [
        {
            'features': {
                'm1': torch.randn(64, 6),
                'm5': torch.randn(13, 6),  # 異なる長さ
                'h1': torch.randn(2, 6)
            },
            'targets': {
                'm1': torch.randn(64, 4),
                'm5': torch.randn(13, 4),
                'h1': torch.randn(2, 4)
            }
        },
        {
            'features': {
                'm1': torch.randn(64, 6),
                'm5': torch.randn(10, 6),  # 異なる長さ
                'h1': torch.randn(1, 6)
            },
            'targets': {
                'm1': torch.randn(64, 4),
                'm5': torch.randn(10, 4),
                'h1': torch.randn(1, 4)
            }
        }
    ]
    
    result = collate_multiscale(batch)
    
    print(f"   ✅ Collate結果:")
    for key, tf_dict in result.items():
        print(f"      {key}:")
        for tf, tensor in tf_dict.items():
            print(f"        {tf}: {tensor.shape}")
    
    return result

def test_masking_strategy_dict():
    """MaskingStrategy Dict対応テスト"""
    print("🧪 Testing MaskingStrategy Dict support...")
    
    masking = MaskingStrategy(test_config, n_features=6)
    
    # Dict形式のfeatures
    features = {
        'm1': torch.randn(2, 64, 6),
        'm5': torch.randn(2, 13, 6),
        'h1': torch.randn(2, 2, 6)
    }
    
    # NaNパディングをシミュレート
    features['m5'][:, 10:, :] = float('nan')
    features['h1'][:, 1:, :] = float('nan')
    
    masks = masking.generate_masks(features, seed=42)
    
    print(f"   ✅ Masking結果:")
    for tf, mask in masks.items():
        print(f"      {tf}: {mask.shape}, mask_ratio={mask.float().mean():.3f}")
    
    # マスク適用テスト
    masked_features = masking.apply_mask_to_features(features, masks)
    
    print(f"   ✅ Mask適用完了")
    
    return masks, masked_features

def test_model_v2_forward():
    """Stage1Model Dict対応テスト"""
    print("🧪 Testing Stage1Model Dict forward...")
    
    model = Stage1Model(test_config)
    model.eval()
    
    # Dict形式の入力
    batch = {
        'm1': torch.randn(2, 64, 6),
        'm5': torch.randn(2, 13, 6),
        'h1': torch.randn(2, 2, 6)
    }
    
    # NaNパディングをシミュレート
    batch['m5'][:, 10:, :] = float('nan')
    batch['h1'][:, 1:, :] = float('nan')
    
    with torch.no_grad():
        outputs = model(batch, eval_mask_ratio=0.15)
    
    print(f"   ✅ Model forward結果:")
    for tf, output in outputs.items():
        print(f"      {tf}: {output.shape}")
    
    return outputs

def test_loss_dict():
    """Stage1CombinedLoss Dict対応テスト"""
    print("🧪 Testing Stage1CombinedLoss Dict support...")
    
    criterion = Stage1CombinedLoss(test_config)
    
    # サンプルデータ
    pred = {
        'm1': torch.randn(2, 64, 4),
        'm5': torch.randn(2, 13, 4),
        'h1': torch.randn(2, 2, 4)
    }
    
    target = {
        'm1': torch.randn(2, 64, 4),
        'm5': torch.randn(2, 13, 4),
        'h1': torch.randn(2, 2, 4)
    }
    
    masks = {
        'm1': torch.rand(2, 64) < 0.15,
        'm5': torch.rand(2, 13) < 0.15,
        'h1': torch.rand(2, 2) < 0.15
    }
    
    # NaNパディングをシミュレート
    pred['m5'][:, 10:, :] = float('nan')
    target['m5'][:, 10:, :] = float('nan')
    pred['h1'][:, 1:, :] = float('nan')
    target['h1'][:, 1:, :] = float('nan')
    
    losses = criterion(pred, target, masks, m1_data={'m1': target['m1']})
    
    print(f"   ✅ Loss計算結果:")
    for loss_name, loss_value in losses.items():
        if hasattr(loss_value, 'item'):
            print(f"      {loss_name}: {loss_value.item():.6f}")
        else:
            print(f"      {loss_name}: {loss_value:.6f}")
    
    return losses

def test_integration():
    """統合テスト"""
    print("🧪 Testing integration (Model v2 full pipeline)...")
    
    # 1. collate test
    collated = test_dict_collate()
    
    # 2. masking test
    masks, masked_features = test_masking_strategy_dict()
    
    # 3. model forward test
    outputs = test_model_v2_forward()
    
    # 4. loss calculation test
    losses = test_loss_dict()
    
    print("✅ 統合テスト完了!")
    return True

if __name__ == "__main__":
    print("🚀 Model v2 テスト開始...")
    
    try:
        success = test_integration()
        print("\n🎉 全テスト成功! Model v2は正常に動作しています。")
    except Exception as e:
        print(f"\n❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)