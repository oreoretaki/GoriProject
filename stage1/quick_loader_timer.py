#!/usr/bin/env python3
"""
DataLoader単体の速度テスト
GPU計算を除外して、純粋なデータ読み込み速度を測定
"""

import time
import torch
import yaml
from pathlib import Path
import sys
sys.path.append('src')

from src.data_loader import create_stage1_dataloaders

def test_dataloader_speed():
    print("🔍 DataLoader単体速度テスト開始")
    
    # 設定読み込み
    with open("configs/t5_large_nofreeze.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 継承設定処理
    if 'extends' in config:
        base_path = Path("configs") / config['extends']
        with open(base_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # さらに継承がある場合
        if 'extends' in base_config:
            root_path = Path("configs") / base_config['extends']
            with open(root_path, 'r') as f:
                root_config = yaml.safe_load(f)
            root_config.update(base_config)
            base_config = root_config
        
        base_config.update(config)
        config = base_config
    
    # DataLoader作成
    print("📊 DataLoader作成中...")
    train_loader, _ = create_stage1_dataloaders("../data/derived", config)
    
    print(f"   総バッチ数: {len(train_loader):,}")
    print(f"   バッチサイズ: {config['training']['batch_size']}")
    print(f"   num_workers: {config['dataloader']['num_workers']}")
    print(f"   pin_memory: {config['dataloader']['pin_memory']}")
    print(f"   persistent_workers: {config['dataloader']['persistent_workers']}")
    print(f"   prefetch_factor: {config['dataloader']['prefetch_factor']}")
    
    # 速度測定 (200バッチ)
    test_batches = min(200, len(train_loader))
    print(f"\n🚀 速度測定開始 ({test_batches}バッチ)")
    
    t0 = time.time()
    for i, batch in zip(range(test_batches), train_loader):
        # バッチデータを軽く触る（遅延読み込み回避）
        if isinstance(batch, dict):
            # Model v2のDict形式の場合
            if 'features' in batch:
                for tf_name, tf_tensor in batch['features'].items():
                    if hasattr(tf_tensor, 'shape'):
                        _ = tf_tensor.shape  # shape確認だけ（GPU転送なし）
            if 'targets' in batch:
                for tf_name, tf_tensor in batch['targets'].items():
                    if hasattr(tf_tensor, 'shape'):
                        _ = tf_tensor.shape  # shape確認だけ（GPU転送なし）
        else:
            # Legacy形式の場合
            if hasattr(batch, 'shape'):
                _ = batch.shape
        
        # プログレス表示
        if i % 50 == 0:
            elapsed = time.time() - t0
            if elapsed > 0:
                current_speed = (i + 1) / elapsed
                print(f"   {i+1:3d}/{test_batches} バッチ - {current_speed:.2f} it/s")
    
    # 結果計算
    dt = time.time() - t0
    speed = test_batches / dt
    
    print(f"\n📈 結果:")
    print(f"   測定時間: {dt:.2f}秒")
    print(f"   DataLoader単体速度: {speed:.2f} it/s")
    
    # 診断
    print(f"\n🔍 診断:")
    if speed >= 5.0:
        print("   ✅ DataLoaderは問題なし → モデル計算に焦点")
        print("   💡 次の手: torch.compile試行 / batch_size増加")
    elif speed >= 1.0:
        print("   ⚠️ DataLoaderがやや遅い → 設定調整推奨")
        print("   💡 次の手: num_workers調整 / prefetch_factor変更")
    else:
        print("   ❌ DataLoaderがボトルネック → 設定見直し必要")
        print("   💡 次の手: ストレージ速度確認 / num_workers削減")
    
    # 参考値
    print(f"\n📊 参考値:")
    print(f"   期待値: >5 it/s (DataLoader問題なし)")
    print(f"   現在値: {speed:.2f} it/s")
    
    return speed

if __name__ == "__main__":
    test_dataloader_speed()