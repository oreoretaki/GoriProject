#!/usr/bin/env python3
"""
Training profiler to identify bottlenecks (CPU vs GPU)
"""

import os
import sys
sys.path.append('src')

import torch
import torch.profiler
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import yaml
import argparse
from pathlib import Path

# Import model and data components
from model import Stage1Model
from data_loader import Stage1Dataset
from losses import Stage1CombinedLoss
from lm_adapter import create_differential_learning_rate_groups
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

class Stage1Lightning(LightningModule):
    """Minimal Lightning wrapper for profiling"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = Stage1Model(config)
        self.criterion = Stage1CombinedLoss(config)
        
    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.model(batch, eval_mask_ratio=None)
        
        # Extract m1_data for cross-TF loss
        m1_data = batch.get('m1') if isinstance(batch, dict) else None
        
        # Calculate losses
        losses = self.criterion(
            pred=outputs,
            target=batch,
            masks=None,  # Training masks are handled internally
            m1_data={'m1': m1_data} if m1_data is not None else None
        )
        
        return losses['total']
    
    def configure_optimizers(self):
        # Simple optimizer for profiling
        return torch.optim.Adam(self.parameters(), lr=1e-3)

def profile_training(config_path: str, data_dir: str):
    """Profile training to identify bottlenecks"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle config inheritance
    if 'extends' in config:
        base_config_path = Path(config_path).parent / config['extends']
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Handle nested inheritance
        if 'extends' in base_config:
            root_config_path = Path(config_path).parent / base_config['extends']
            with open(root_config_path, 'r') as f:
                root_config = yaml.safe_load(f)
            root_config.update(base_config)
            base_config = root_config
            
        base_config.update(config)
        config = base_config
    
    # Override data directory
    config['data']['data_dir'] = data_dir
    
    print("🔧 Profiling Training Performance...")
    print(f"   Config: {config_path}")
    print(f"   Data: {data_dir}")
    print(f"   Batch size: {config['training']['batch_size']}")
    print(f"   Async sampler: {config.get('model', {}).get('async_sampler', False)}")
    
    # Create model
    pl_module = Stage1Lightning(config)
    
    # Create dataset and dataloader
    print("📊 Creating DataLoader...")
    train_dataset = Stage1Dataset(
        config=config,
        split='train',
        return_dict=True  # Model v2 requires dict format
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config.get('dataloader', {}).get('num_workers', 4),
        pin_memory=config.get('dataloader', {}).get('pin_memory', True),
        persistent_workers=config.get('dataloader', {}).get('persistent_workers', False),
        prefetch_factor=config.get('dataloader', {}).get('prefetch_factor', 2)
    )
    
    print(f"   Train samples: {len(train_dataset):,}")
    print(f"   Batches per epoch: {len(train_loader):,}")
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl_module = pl_module.to(device)
    
    # Create profiler output directory
    prof_dir = Path("log/prof")
    prof_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔥 Starting profiling (50 batches)...")
    print(f"   Output: {prof_dir}")
    
    # Profile training
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=10),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(prof_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for i, batch in enumerate(train_loader):
            if i >= 50:
                break
                
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Training step
            loss = pl_module.training_step(batch, i)
            
            # Backward pass
            loss.backward()
            
            # Step profiler
            prof.step()
            
            if i % 10 == 0:
                print(f"   Batch {i}/50, Loss: {loss.item():.4f}")
    
    print("✅ Profiling complete!")
    print(f"📊 View results with: tensorboard --logdir={prof_dir}")
    
    # Print key metrics
    print("\n📈 Key Metrics:")
    key_metrics = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
    print(key_metrics)
    
    # Analyze CPU vs GPU time
    cpu_time = sum([item.self_cpu_time_total for item in prof.key_averages()])
    cuda_time = sum([item.self_cuda_time_total for item in prof.key_averages()])
    
    if cuda_time > 0:
        gpu_utilization = cuda_time / (cpu_time + cuda_time) * 100
        print(f"\n🔍 GPU Utilization: {gpu_utilization:.1f}%")
        
        if gpu_utilization < 50:
            print("⚠️ Low GPU utilization - DataLoader/preprocessing is likely the bottleneck")
        elif gpu_utilization > 80:
            print("✅ Good GPU utilization - Model computation is the main workload")
        else:
            print("⚡ Moderate GPU utilization - Mixed CPU/GPU bottleneck")

def main():
    parser = argparse.ArgumentParser(description='Profile Stage 1 training')
    parser.add_argument('--config', type=str, default='configs/t5_large_nofreeze.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../data/derived',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    profile_training(args.config, args.data_dir)

if __name__ == '__main__':
    main()