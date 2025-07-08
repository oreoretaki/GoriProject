#!/usr/bin/env python3
"""
リアルタイムGPU/CPU監視スクリプト
"""

import time
import subprocess
import psutil
import threading
from datetime import datetime

def get_gpu_stats():
    """NVIDIA GPUの使用率とメモリを取得"""
    try:
        # nvidia-smi コマンド実行
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = line.split(', ')
            return {
                'gpu_util': int(parts[0]),
                'mem_used': int(parts[1]),
                'mem_total': int(parts[2]),
                'temperature': int(parts[3]),
                'power': float(parts[4])
            }
    except Exception as e:
        print(f"GPU stats error: {e}")
    
    return None

def get_cpu_stats():
    """CPU使用率とメモリを取得"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=None),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3)
    }

def monitor_system(duration_minutes=5, interval_seconds=2):
    """システムリソースを監視"""
    print(f"🔍 {duration_minutes}分間のリアルタイム監視開始 (更新間隔: {interval_seconds}秒)")
    print("=" * 80)
    print(f"{'時刻':<8} {'GPU%':<6} {'GPU-Mem':<10} {'GPU-Temp':<8} {'CPU%':<6} {'RAM%':<6} {'RAM-GB':<10}")
    print("=" * 80)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    gpu_utils = []
    cpu_utils = []
    
    while time.time() < end_time:
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # GPU統計
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            gpu_util = gpu_stats['gpu_util']
            gpu_mem = f"{gpu_stats['mem_used']}/{gpu_stats['mem_total']}"
            gpu_temp = f"{gpu_stats['temperature']}°C"
            gpu_utils.append(gpu_util)
        else:
            gpu_util = "N/A"
            gpu_mem = "N/A"
            gpu_temp = "N/A"
        
        # CPU統計
        cpu_stats = get_cpu_stats()
        cpu_util = cpu_stats['cpu_percent']
        ram_percent = cpu_stats['memory_percent']
        ram_gb = f"{cpu_stats['memory_used_gb']:.1f}/{cpu_stats['memory_total_gb']:.1f}"
        
        cpu_utils.append(cpu_util)
        
        # 出力
        print(f"{current_time:<8} {gpu_util:<6} {gpu_mem:<10} {gpu_temp:<8} {cpu_util:<6.1f} {ram_percent:<6.1f} {ram_gb:<10}")
        
        time.sleep(interval_seconds)
    
    # サマリー
    print("=" * 80)
    print("📊 監視結果サマリー:")
    if gpu_utils:
        print(f"  GPU使用率 - 平均: {sum(gpu_utils)/len(gpu_utils):.1f}%, 最大: {max(gpu_utils)}%, 最小: {min(gpu_utils)}%")
    if cpu_utils:
        print(f"  CPU使用率 - 平均: {sum(cpu_utils)/len(cpu_utils):.1f}%, 最大: {max(cpu_utils):.1f}%, 最小: {min(cpu_utils):.1f}%")
    
    # ボトルネック推定
    if gpu_utils:
        avg_gpu = sum(gpu_utils) / len(gpu_utils)
        avg_cpu = sum(cpu_utils) / len(cpu_utils)
        
        print("\n💡 ボトルネック推定:")
        if avg_gpu < 50 and avg_cpu > 80:
            print("  🔴 CPU bound: CPUがボトルネック (num_workers調整推奨)")
        elif avg_gpu < 50 and avg_cpu < 50:
            print("  🔴 IO bound: データローディングがボトルネック (SSD, prefetch調整)")
        elif avg_gpu > 80:
            print("  🟢 GPU bound: GPUを効率的に使用中")
        else:
            print("  🟡 Mixed: 複数要因のボトルネック")

def check_dataloader_workers():
    """DataLoaderワーカー数の推奨値を計算"""
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    
    print("🔧 DataLoader最適化推奨値:")
    print(f"  CPU数: {cpu_count} (論理), {cpu_physical} (物理)")
    print(f"  推奨num_workers: {min(cpu_count, 16)} (過剰な並列化を避ける)")
    print(f"  推奨prefetch_factor: 2-4")
    print(f"  pin_memory: True (GPU使用時)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPU/CPUリアルタイム監視")
    parser.add_argument('--duration', type=int, default=5, help='監視時間(分)')
    parser.add_argument('--interval', type=float, default=2.0, help='更新間隔(秒)')
    parser.add_argument('--check-config', action='store_true', help='DataLoader設定推奨値を表示')
    
    args = parser.parse_args()
    
    if args.check_config:
        check_dataloader_workers()
        return
    
    try:
        monitor_system(args.duration, args.interval)
    except KeyboardInterrupt:
        print("\n⏹️ 監視を停止しました")

if __name__ == "__main__":
    main()