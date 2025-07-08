#!/usr/bin/env python3
"""
Stage 1 評価スクリプト
Correlation, Consistency ratio, Spectral delta計算
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import json
import warnings
warnings.filterwarnings('ignore')

# TF32を有効化（RTX 30xx/40xx/A100で高速化）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
print("✅ TF32とTensor Core最適化を有効化")

# プロジェクトルートをPATHに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

from src.data_loader import create_stage1_dataloaders
from scripts.train_stage1 import Stage1LightningModule

class Stage1Evaluator:
    """Stage 1 評価クラス"""
    
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.timeframes = config['data']['timeframes']
        self.device = next(model.parameters()).device
        
    def evaluate_model(self, dataloader) -> Dict:
        """モデル評価実行"""
        
        print("📊 Stage 1 モデル評価開始")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_masks = []
        
        # 予測収集
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"   バッチ {batch_idx}/{len(dataloader)} 処理中...")
                    
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # 予測
                outputs = self.model(features, masks)
                predictions = outputs['reconstructed']
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(masks.cpu())
                
        # テンソル結合
        all_predictions = torch.cat(all_predictions, dim=0)  # [n_samples, n_tf, seq_len, 4]
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        print(f"   評価データ形状: {all_predictions.shape}")
        
        # メトリクス計算
        metrics = self._calculate_metrics(all_predictions, all_targets, all_masks)
        
        return metrics
        
    def _calculate_metrics(self, predictions, targets, masks) -> Dict:
        """各種メトリクス計算"""
        
        metrics = {
            'correlation_per_tf': {},
            'consistency_ratio': {},
            'spectral_delta': {},
            'reconstruction_quality': {}
        }
        
        n_samples, n_tf, seq_len, n_features = predictions.shape
        
        for tf_idx, tf_name in enumerate(self.timeframes):
            print(f"   {tf_name.upper()} メトリクス計算中...")
            
            pred_tf = predictions[:, tf_idx]  # [n_samples, seq_len, 4]
            target_tf = targets[:, tf_idx]
            mask_tf = masks[:, tf_idx]  # [n_samples, seq_len]
            
            # 1. Correlation@TF
            correlations = self._calculate_correlation_per_tf(pred_tf, target_tf, mask_tf)
            metrics['correlation_per_tf'][tf_name] = correlations
            
            # 2. Reconstruction Quality (MSE, MAE)
            quality = self._calculate_reconstruction_quality(pred_tf, target_tf, mask_tf)
            metrics['reconstruction_quality'][tf_name] = quality
            
            # 3. Spectral Delta
            spectral_delta = self._calculate_spectral_delta(pred_tf, target_tf, mask_tf)
            metrics['spectral_delta'][tf_name] = spectral_delta
            
        # 4. Consistency ratio (M1ベースで他TFとの整合性)
        if n_tf > 1:
            consistency = self._calculate_consistency_ratio(predictions, targets, masks)
            metrics['consistency_ratio'] = consistency
            
        # 5. 全体サマリー
        metrics['summary'] = self._calculate_summary(metrics)
        
        return metrics
        
    def _calculate_correlation_per_tf(self, pred, target, mask) -> Dict:
        """TFごとの相関計算"""
        
        correlations = {'ohlc': []}
        ohlc_names = ['open', 'high', 'low', 'close']
        
        for feat_idx, feat_name in enumerate(ohlc_names):
            # マスクされた部分のみ
            masked_pred = pred[mask.bool(), feat_idx]
            masked_target = target[mask.bool(), feat_idx]
            
            if len(masked_pred) > 1:
                # ピアソン相関
                corr_matrix = np.corrcoef(masked_pred.numpy(), masked_target.numpy())
                correlation = corr_matrix[0, 1] if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                correlation = 0.0
                
            correlations['ohlc'].append(correlation)
            correlations[feat_name] = correlation
            
        # 平均相関
        correlations['mean'] = np.mean(correlations['ohlc'])
        
        return correlations
        
    def _calculate_reconstruction_quality(self, pred, target, mask) -> Dict:
        """再構築品質メトリクス"""
        
        # マスクされた部分のみ
        masked_pred = pred[mask.bool()]  # [n_masked, 4]
        masked_target = target[mask.bool()]
        
        if len(masked_pred) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}
            
        # MSE, MAE, RMSE
        mse = torch.mean((masked_pred - masked_target) ** 2).item()
        mae = torch.mean(torch.abs(masked_pred - masked_target)).item()
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(masked_pred)
        }
        
    def _calculate_spectral_delta(self, pred, target, mask) -> Dict:
        """スペクトラム差分計算"""
        
        spectral_deltas = []
        
        for feat_idx in range(4):  # OHLC
            # マスクされた部分のシーケンスを抽出
            valid_sequences = []
            
            for sample_idx in range(pred.size(0)):
                sample_mask = mask[sample_idx]
                if sample_mask.sum() > 10:  # 最低10ポイント必要
                    pred_seq = pred[sample_idx, sample_mask.bool(), feat_idx]
                    target_seq = target[sample_idx, sample_mask.bool(), feat_idx]
                    
                    if len(pred_seq) > 10:
                        valid_sequences.append((pred_seq, target_seq))
                        
            if valid_sequences:
                seq_deltas = []
                for pred_seq, target_seq in valid_sequences[:100]:  # 最大100シーケンス
                    # FFTでスペクトラム計算
                    pred_fft = torch.fft.fft(pred_seq)
                    target_fft = torch.fft.fft(target_seq)
                    
                    # パワースペクトラム
                    pred_power = torch.abs(pred_fft) ** 2
                    target_power = torch.abs(target_fft) ** 2
                    
                    # 対数パワースペクトラムの差分
                    log_pred = torch.log(pred_power + 1e-8)
                    log_target = torch.log(target_power + 1e-8)
                    
                    delta = torch.mean(torch.abs(log_pred - log_target)).item()
                    seq_deltas.append(delta)
                    
                spectral_deltas.append(np.mean(seq_deltas))
            else:
                spectral_deltas.append(float('inf'))
                
        return {
            'per_feature': spectral_deltas,
            'mean': np.mean([d for d in spectral_deltas if d != float('inf')])
        }
        
    def _calculate_consistency_ratio(self, predictions, targets, masks) -> Dict:
        """整合性比率計算（TF間の一貫性）"""
        
        # M1を基準として、他のTFとの整合性をチェック
        m1_pred = predictions[:, 0]  # [n_samples, seq_len, 4]
        m1_target = targets[:, 0]
        
        consistency_ratios = {}
        
        for tf_idx in range(1, predictions.size(1)):
            tf_name = self.timeframes[tf_idx]
            tf_pred = predictions[:, tf_idx]
            tf_target = targets[:, tf_idx]
            
            # 簡易整合性チェック：予測と実際の差分が許容範囲内か
            tolerance = 0.01  # 1%許容差
            
            consistent_samples = 0
            total_samples = 0
            
            for sample_idx in range(tf_pred.size(0)):
                tf_sample = tf_pred[sample_idx]
                target_sample = tf_target[sample_idx]
                
                if torch.any(target_sample != 0):  # 有効なデータが存在
                    relative_error = torch.abs(tf_sample - target_sample) / (torch.abs(target_sample) + 1e-8)
                    if torch.mean(relative_error) < tolerance:
                        consistent_samples += 1
                    total_samples += 1
                    
            if total_samples > 0:
                ratio = consistent_samples / total_samples
            else:
                ratio = 0.0
                
            consistency_ratios[tf_name] = {
                'ratio': ratio,
                'consistent_samples': consistent_samples,
                'total_samples': total_samples
            }
            
        return consistency_ratios
        
    def _calculate_summary(self, metrics: Dict) -> Dict:
        """全体サマリー計算"""
        
        # 平均相関
        correlations = [tf_metrics['mean'] for tf_metrics in metrics['correlation_per_tf'].values()]
        mean_correlation = np.mean(correlations)
        
        # 平均再構築品質
        mse_values = [tf_metrics['mse'] for tf_metrics in metrics['reconstruction_quality'].values() 
                     if tf_metrics['mse'] != float('inf')]
        mean_mse = np.mean(mse_values) if mse_values else float('inf')
        
        # 平均スペクトラム差分
        spectral_values = [tf_metrics['mean'] for tf_metrics in metrics['spectral_delta'].values()
                          if tf_metrics['mean'] != float('inf')]
        mean_spectral_delta = np.mean(spectral_values) if spectral_values else float('inf')
        
        # 平均整合性比率
        if metrics['consistency_ratio']:
            consistency_values = [tf_metrics['ratio'] for tf_metrics in metrics['consistency_ratio'].values()]
            mean_consistency = np.mean(consistency_values)
        else:
            mean_consistency = 0.0
            
        return {
            'mean_correlation': mean_correlation,
            'mean_mse': mean_mse,
            'mean_spectral_delta': mean_spectral_delta,
            'mean_consistency_ratio': mean_consistency,
            'n_timeframes': len(self.timeframes)
        }
        
    def save_results(self, metrics: Dict, output_path: str):
        """結果保存"""
        
        # JSON保存
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.ndarray) else x)
            
        print(f"💾 評価結果保存: {output_path}")
        
        # サマリー表示
        summary = metrics['summary']
        print("📈 評価サマリー:")
        print(f"   平均相関: {summary['mean_correlation']:.4f}")
        print(f"   平均MSE: {summary['mean_mse']:.6f}")
        print(f"   平均スペクトラム差分: {summary['mean_spectral_delta']:.4f}")
        print(f"   平均整合性比率: {summary['mean_consistency_ratio']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Stage 1 Evaluation')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルパス')
    parser.add_argument('--ckpt', type=str, required=True, help='チェックポイントパス')
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリ')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='出力ファイル')
    
    args = parser.parse_args()
    
    # 設定読み込み
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['data_dir'] = args.data_dir
    
    print("🔍 Stage 1 評価開始")
    print(f"   チェックポイント: {args.ckpt}")
    print(f"   データディレクトリ: {args.data_dir}")
    
    # データローダー作成
    _, val_loader = create_stage1_dataloaders(args.data_dir, config)
    
    # モデル読み込み
    print("📂 モデル読み込み中...")
    model = Stage1LightningModule.load_from_checkpoint(args.ckpt, config=config)
    model.eval()
    
    # 評価実行
    evaluator = Stage1Evaluator(model, config)
    metrics = evaluator.evaluate_model(val_loader)
    
    # 結果保存
    evaluator.save_results(metrics, args.output)
    
    print("✅ 評価完了")

if __name__ == "__main__":
    main()