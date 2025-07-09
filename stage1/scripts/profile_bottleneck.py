#!/usr/bin/env python3
"""
GPU使用率ボトルネック分析スクリプト
"""

import torch
import torch.profiler
import time
import psutil
import threading
from pathlib import Path
import sys
import os

# プロジェクトパスを追加
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from src.data_loader import create_stage1_dataloaders
from scripts.train_stage1 import load_config, Stage1LightningModule

def monitor_system(stop_event, results):
    """システムリソースを監視"""
    cpu_usage = []
    memory_usage = []
    
    while not stop_event.is_set():
        cpu_usage.append(psutil.cpu_percent(interval=1))
        memory_usage.append(psutil.virtual_memory().percent)
        time.sleep(1)
    
    results['cpu_avg'] = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
    results['memory_avg'] = sum(memory_usage) / len(memory_usage) if memory_usage else 0

def profile_data_loading(train_loader, num_batches=10):
    """データローディング性能を測定"""
    print("🔍 データローディング性能測定中...")
    
    load_times = []
    transfer_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        
        load_start = time.time()
        # データロード完了
        load_end = time.time()
        
        # GPU転送時間測定
        transfer_start = time.time()
        if torch.cuda.is_available():
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        transfer_end = time.time()
        
        load_time = load_end - load_start
        transfer_time = transfer_end - transfer_start
        total_time = load_time + transfer_time
        
        load_times.append(load_time)
        transfer_times.append(transfer_time)
        
        print(f"  Batch {i+1}: Load {load_time:.3f}s + Transfer {transfer_time:.3f}s = {total_time:.3f}s")
    
    avg_load = sum(load_times) / len(load_times)
    avg_transfer = sum(transfer_times) / len(transfer_times)
    avg_total = avg_load + avg_transfer
    
    print(f"📊 平均データロード時間: {avg_load:.3f}s/batch")
    print(f"📊 平均GPU転送時間: {avg_transfer:.3f}s/batch")
    print(f"📊 平均総時間: {avg_total:.3f}s/batch")
    return avg_total

def profile_model_forward(model, train_loader, num_batches=5):
    """モデルforward pass性能を測定"""
    print("🔍 モデルforward pass性能測定中...")
    
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= num_batches:
                break
            
            if torch.cuda.is_available():
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            
            start_time = time.time()
            
            # Forward pass
            features = batch['features']
            masks = batch['masks']
            outputs = model.model(features, masks)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # GPU同期
            
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"  Forward {i+1}: {times[-1]:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"📊 平均forward時間: {avg_time:.3f}s/batch")
    return avg_time

def profile_with_pytorch_profiler(model, train_loader):
    """PyTorch Profilerで詳細分析"""
    print("🔍 PyTorch Profiler実行中...")
    
    model.train()
    
    def trace_handler(prof):
        # プロファイル結果を保存
        prof.export_chrome_trace("trace.json")
        prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
        
        # 上位処理を表示
        print("\n📊 GPU使用率上位処理:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        print("\n📊 CPU使用率上位処理:")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3,
            repeat=2
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 10:  # 10バッチで終了
                break
            
            if torch.cuda.is_available():
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            features = batch['features']
            targets = batch['targets']
            masks = batch['masks']
            
            outputs = model.model(features, masks)
            reconstructed = outputs['reconstructed']
            
            # 損失計算
            m1_data = targets[:, 0]
            losses = model.criterion(reconstructed, targets, masks, m1_data)
            
            prof.step()

def main():
    print("🚀 GPU使用率ボトルネック分析開始")
    print("=" * 50)
    
    # 設定読み込み
    config_path = "configs/t5_large_nofreeze.yaml"
    config = load_config(config_path)
    config['data']['data_dir'] = "../data/derived"
    
    # バッチサイズ256用の設定確認
    print(f"📊 バッチサイズ: {config['training']['batch_size']}")
    print(f"📊 accumulate_grad_batches: {config['training']['accumulate_grad_batches']}")
    print(f"📊 実効バッチサイズ: {config['training']['batch_size'] * config['training']['accumulate_grad_batches']}")
    
    # バッチサイズ1024用のテスト設定
    config['development'] = {
        'limit_train_batches': 10,    # バッチサイズ1024用に減少
        'limit_val_batches': 2        # バッチサイズ1024用に減少
    }
    
    print("📊 データローダー作成中...")
    train_loader, val_loader = create_stage1_dataloaders("../data/derived", config)
    
    print("🧠 モデル初期化中...")
    model = Stage1LightningModule(config)
    
    # PyTorch 2.0 コンパイル最適化（事前ウォームアップ付き）
    if torch.__version__ >= '2.0.0':
        print("🚀 PyTorch 2.0 コンパイル最適化を適用中...")
        try:
            # 事前ウォームアップでコンパイル時間を隠蔽
            print("🔥 ダミー入力でウォームアップ実行中...")
            with torch.no_grad():
                # バッチサイズ1でダミー入力作成
                dummy_features = torch.randn(1, 6, 128, 36, device=model.device, dtype=torch.bfloat16)
                dummy_masks = torch.ones(1, 6, 128, device=model.device, dtype=torch.bool)
                
                # ウォームアップ実行
                _ = model.model(dummy_features, dummy_masks)
                print("✅ ウォームアップ完了")
            
            # コンパイル適用
            model.model = torch.compile(model.model, backend="inductor", mode="max-autotune")
            print("✅ TorchCompile適用完了（事前ウォームアップ済み）")
            
        except Exception as e:
            print(f"⚠️ TorchCompile失敗、通常モード: {e}")
    else:
        print("⚠️ PyTorch 2.0+が必要です（TorchCompileをスキップ）")
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✅ CUDA使用: {torch.cuda.get_device_name()}")
        print(f"📊 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # システム監視開始
    stop_event = threading.Event()
    monitor_results = {}
    monitor_thread = threading.Thread(target=monitor_system, args=(stop_event, monitor_results))
    monitor_thread.start()
    
    try:
        # 1. データローディング性能測定
        data_load_time = profile_data_loading(train_loader, num_batches=10)
        
        # 2. モデルforward性能測定
        forward_time = profile_model_forward(model, train_loader, num_batches=5)
        
        # 3. PyTorch Profiler実行
        profile_with_pytorch_profiler(model, train_loader)
        
        # 分析結果
        print("\n" + "=" * 50)
        print("📊 ボトルネック分析結果")
        print("=" * 50)
        print(f"📈 データロード時間: {data_load_time:.3f}s/batch")
        print(f"🔄 Forward時間: {forward_time:.3f}s/batch")
        print(f"⚡ 総処理時間: {data_load_time + forward_time:.3f}s/batch")
        
        # GPU使用率が低い原因推定
        if data_load_time > forward_time * 2:
            print("🔴 ボトルネック: データローディング (CPU/IO bound)")
            print("💡 対策: num_workers増加、pin_memory=True、SSD使用")
        elif forward_time > data_load_time * 2:
            print("🔴 ボトルネック: モデル計算 (GPU bound)")
            print("💡 対策: バッチサイズ増加、mixed precision使用")
        else:
            print("🟡 バランス型: データロードとモデル計算が均衡")
            print("💡 対策: 並列度調整、prefetch最適化")
        
    finally:
        # システム監視停止
        stop_event.set()
        monitor_thread.join()
        
        print(f"\n📊 システムリソース使用率:")
        print(f"  CPU平均: {monitor_results.get('cpu_avg', 0):.1f}%")
        print(f"  Memory平均: {monitor_results.get('memory_avg', 0):.1f}%")
        
        print("\n📁 結果ファイル:")
        print("  - trace.json (Chrome Tracing用)")
        print("  - profiler_stacks.txt (詳細スタック)")

if __name__ == "__main__":
    main()