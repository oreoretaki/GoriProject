#!/usr/bin/env python3
"""
T5実験結果の取りまとめスクリプト
複数の実験ログからval_loss/val_corr/時間/VRAMを表形式で出力
"""

import os
import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def extract_metrics_from_log(log_file_path):
    """学習ログから最終メトリクスを抽出"""
    metrics = {
        'config_name': None,
        'experiment_name': None,
        'final_epoch': None,
        'duration_seconds': None,
        'final_val_loss': None,
        'final_val_corr': None,
        'best_val_corr': None,
        'final_grad_norm': None,
        'amp_overflow_count': 0,
        'early_stopped': False
    }
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # 設定名を抽出
        config_match = re.search(r'--config configs/([^\.]+)\.yaml', log_content)
        if config_match:
            metrics['config_name'] = config_match.group(1)
        
        # 実験時間を抽出
        duration_match = re.search(r'実行時間: (\d+)秒', log_content)
        if duration_match:
            metrics['duration_seconds'] = int(duration_match.group(1))
        
        # 最終エポック数を抽出
        epoch_matches = re.findall(r'Epoch (\d+):', log_content)
        if epoch_matches:
            metrics['final_epoch'] = int(epoch_matches[-1])
        
        # 最終val_lossとval_corrを抽出
        val_patterns = [
            r'val_loss_ep=([0-9.]+)',
            r'val_corr_ep=([0-9.-]+)',
            r'val_correlation_mean\s*([0-9.-]+)',
            r'grad_norm=([0-9.e+-]+)'
        ]
        
        # 最後の値を取得
        for pattern in val_patterns:
            matches = re.findall(pattern, log_content)
            if matches:
                try:
                    if 'val_loss' in pattern:
                        metrics['final_val_loss'] = float(matches[-1])
                    elif 'val_corr' in pattern or 'correlation' in pattern:
                        metrics['final_val_corr'] = float(matches[-1])
                    elif 'grad_norm' in pattern:
                        metrics['final_grad_norm'] = float(matches[-1])
                except ValueError:
                    pass
        
        # ベストval_correlationを抽出
        best_corr_match = re.search(r'最良チェックポイント.*val_correlation_mean=([0-9.-]+)', log_content)
        if best_corr_match:
            metrics['best_val_corr'] = float(best_corr_match.group(1))
        
        # AMP overflow カウント
        amp_overflow_matches = re.findall(r'amp_overflow', log_content)
        metrics['amp_overflow_count'] = len(amp_overflow_matches)
        
        # Early stopping検出
        if 'EarlyStopping' in log_content and 'stopped' in log_content:
            metrics['early_stopped'] = True
            
    except Exception as e:
        print(f"⚠️ ログ解析エラー {log_file_path}: {e}")
    
    return metrics

def find_experiment_results(results_dir):
    """実験結果ディレクトリから全ログファイルを検索"""
    results_path = Path(results_dir)
    log_files = []
    
    if not results_path.exists():
        print(f"❌ 結果ディレクトリが見つかりません: {results_dir}")
        return log_files
    
    # タイムスタンプディレクトリを検索
    for timestamp_dir in results_path.iterdir():
        if timestamp_dir.is_dir():
            # 各実験のtraining.logを検索
            for exp_dir in timestamp_dir.iterdir():
                if exp_dir.is_dir():
                    log_file = exp_dir / 'training.log'
                    if log_file.exists():
                        log_files.append(log_file)
    
    return sorted(log_files)

def create_comparison_table(metrics_list):
    """メトリクス一覧から比較表を作成"""
    if not metrics_list:
        return pd.DataFrame()
    
    df = pd.DataFrame(metrics_list)
    
    # カラム順序を整理
    columns_order = [
        'config_name', 'final_epoch', 'duration_seconds',
        'final_val_loss', 'final_val_corr', 'best_val_corr',
        'final_grad_norm', 'amp_overflow_count', 'early_stopped'
    ]
    
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    
    # 並び替え（ベースライン→T5系）
    config_order = ['t5_baseline_10ep', 't5_frozen_all_10ep', 't5_freeze_10ep', 't5_nofreeze_10ep']
    df['config_order'] = df['config_name'].map({config: i for i, config in enumerate(config_order)})
    df = df.sort_values('config_order').drop('config_order', axis=1)
    
    return df

def main():
    parser = argparse.ArgumentParser(description='T5実験結果取りまとめ')
    parser.add_argument('--results_dir', type=str, default='./t5_experiment_results',
                       help='実験結果ディレクトリ')
    parser.add_argument('--output', type=str, default='t5_comparison_summary.csv',
                       help='出力CSVファイル名')
    parser.add_argument('--format', type=str, choices=['csv', 'markdown', 'both'], default='both',
                       help='出力形式')
    
    args = parser.parse_args()
    
    print("📊 T5実験結果取りまとめ開始")
    print(f"   結果ディレクトリ: {args.results_dir}")
    
    # ログファイル検索
    log_files = find_experiment_results(args.results_dir)
    print(f"   見つかったログファイル: {len(log_files)}")
    
    if not log_files:
        print("❌ ログファイルが見つかりませんでした")
        return
    
    # メトリクス抽出
    metrics_list = []
    for log_file in log_files:
        print(f"   解析中: {log_file}")
        metrics = extract_metrics_from_log(log_file)
        if metrics['config_name']:
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("❌ 有効なメトリクスが抽出できませんでした")
        return
    
    # 比較表作成
    df = create_comparison_table(metrics_list)
    
    print(f"\n📈 実験結果サマリー ({len(metrics_list)}実験)")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # ファイル出力
    if args.format in ['csv', 'both']:
        csv_file = args.output
        df.to_csv(csv_file, index=False, float_format='%.6f')
        print(f"\n💾 CSV保存: {csv_file}")
    
    if args.format in ['markdown', 'both']:
        md_file = args.output.replace('.csv', '.md')
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# T5転移学習実験結果比較\n\n")
            f.write(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## 結果サマリー\n\n")
            f.write(df.to_markdown(index=False, floatfmt='.4f'))
            f.write("\n\n## 設定説明\n\n")
            f.write("| 設定名 | 説明 |\n")
            f.write("|--------|------|\n")
            f.write("| t5_baseline_10ep | ランダム初期化（従来手法） |\n")
            f.write("| t5_frozen_all_10ep | T5完全凍結（表現のみ利用） |\n") 
            f.write("| t5_freeze_10ep | T5を2エポック凍結後解凍 |\n")
            f.write("| t5_nofreeze_10ep | T5を最初から学習 |\n")
        
        print(f"💾 Markdown保存: {md_file}")
    
    # 簡易分析
    print("\n🔍 簡易分析")
    if len(df) > 1:
        baseline_idx = df[df['config_name'] == 't5_baseline_10ep'].index
        if len(baseline_idx) > 0:
            baseline_corr = df.loc[baseline_idx[0], 'final_val_corr']
            print(f"   ベースライン val_corr: {baseline_corr:.4f}")
            
            for _, row in df.iterrows():
                if row['config_name'] != 't5_baseline_10ep':
                    improvement = row['final_val_corr'] - baseline_corr
                    print(f"   {row['config_name']}: {improvement:+.4f} ({improvement/abs(baseline_corr)*100:+.1f}%)")

if __name__ == "__main__":
    main()