#!/usr/bin/env python3
"""
Stage 1 損失関数
Multi-resolution STFT, Cross-TF consistency, Amplitude-Phase correlation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class HuberLoss(nn.Module):
    """Huber損失（TFごとのOHLC再構築）"""
    
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測OHLC
            target: [batch, n_tf, seq_len, 4] 正解OHLC
            mask: [batch, n_tf, seq_len] マスク（1=損失計算対象）
            
        Returns:
            loss: スカラー損失
        """
        loss = F.huber_loss(pred, target, delta=self.delta, reduction='none')  # [batch, n_tf, seq_len, 4]
        
        if mask is not None:
            # マスク部分のみ損失計算
            mask = mask.unsqueeze(-1)  # [batch, n_tf, seq_len, 1]
            loss = loss * mask
            return loss.sum() / (mask.sum() * 4 + 1e-8)
        else:
            return loss.mean()

class STFTLoss(nn.Module):
    """マルチ解像度STFT損失（スペクトログラム特徴保持）"""
    
    def __init__(self, scales: List[int] = [256, 512, 1024], hop_ratio: float = 0.25):
        super().__init__()
        self.scales = scales
        self.hop_ratio = hop_ratio
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測OHLC
            target: [batch, n_tf, seq_len, 4] 正解OHLC
            mask: [batch, n_tf, seq_len] マスク
            
        Returns:
            loss: STFT損失
        """
        batch_size, n_tf, seq_len, n_features = pred.shape
        total_loss = 0.0
        
        # 各特徴量（OHLC）とTFについてSTFT損失計算
        for tf_idx in range(n_tf):
            for feat_idx in range(n_features):
                pred_signal = pred[:, tf_idx, :, feat_idx]  # [batch, seq_len]
                target_signal = target[:, tf_idx, :, feat_idx]  # [batch, seq_len]
                
                # 各スケールでSTFT損失計算
                for scale in self.scales:
                    if seq_len < scale:
                        continue
                        
                    hop_length = int(scale * self.hop_ratio)
                    
                    # STFT計算（16bit混合精度との互換性のため32bitに変換）
                    pred_signal_32 = pred_signal.to(torch.float32)
                    target_signal_32 = target_signal.to(torch.float32)
                    pred_stft = torch.stft(
                        pred_signal_32, 
                        n_fft=scale, 
                        hop_length=hop_length, 
                        return_complex=True
                    )
                    target_stft = torch.stft(
                        target_signal_32, 
                        n_fft=scale, 
                        hop_length=hop_length, 
                        return_complex=True
                    )
                    
                    # マグニチュード損失
                    pred_mag = torch.abs(pred_stft)
                    target_mag = torch.abs(target_stft)
                    mag_loss = F.l1_loss(pred_mag, target_mag)
                    
                    # 位相損失（複素数として）
                    phase_loss = F.l1_loss(pred_stft.real, target_stft.real) + \
                                F.l1_loss(pred_stft.imag, target_stft.imag)
                    
                    total_loss += mag_loss + 0.1 * phase_loss
                    
        # 正規化
        normalization = len(self.scales) * n_tf * n_features
        return total_loss / normalization

class CrossTFConsistencyLoss(nn.Module):
    """クロスTF整合性損失（デコードTF vs M1集約）"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, m1_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測されたマルチTF OHLC
            m1_data: [batch, seq_len, 4] M1 OHLC（ベース）
            
        Returns:
            loss: クロスTF整合性損失
        """
        batch_size, n_tf, seq_len, _ = pred.shape
        total_loss = 0.0
        
        # M1は pred[:, 0] と仮定
        pred_m1 = pred[:, 0]  # [batch, seq_len, 4]
        
        # 各上位TFについて、対応するM1集約と比較
        tf_intervals = {1: 5, 2: 15, 3: 30, 4: 60, 5: 240}  # M5, M15, M30, H1, H4の間隔（分）
        
        for tf_idx in range(1, min(n_tf, 6)):  # M5以上のTFについて
            if tf_idx not in tf_intervals:
                continue
                
            interval = tf_intervals[tf_idx]
            pred_tf = pred[:, tf_idx]  # [batch, seq_len, 4]
            
            # M1から期待される集約値を計算
            expected_tf = self._aggregate_m1_to_tf(m1_data, interval, seq_len)
            
            # MSE損失
            tf_loss = F.mse_loss(pred_tf, expected_tf)
            total_loss += tf_loss
            
        return total_loss / max(1, min(n_tf - 1, 5))
        
    def _aggregate_m1_to_tf(self, m1_data: torch.Tensor, interval: int, target_len: int) -> torch.Tensor:
        """
        M1データを指定間隔で集約してTFデータを生成
        
        Args:
            m1_data: [batch, seq_len, 4] M1 OHLC
            interval: 集約間隔
            target_len: 出力シーケンス長
            
        Returns:
            aggregated: [batch, target_len, 4] 集約済みOHLC
        """
        batch_size, seq_len, _ = m1_data.shape
        
        # 簡易集約（実際にはより精密な実装が必要）
        if interval >= seq_len:
            # 全期間集約
            open_val = m1_data[:, 0, 0:1]  # 最初の始値
            high_val = m1_data[:, :, 1].max(dim=1, keepdim=True)[0]  # 最高値
            low_val = m1_data[:, :, 2].min(dim=1, keepdim=True)[0]  # 最安値
            close_val = m1_data[:, -1, 3:4]  # 最後の終値
            
            aggregated_bar = torch.cat([open_val, high_val, low_val, close_val], dim=1)
            return aggregated_bar.unsqueeze(1).expand(-1, target_len, -1)
        else:
            # 区間ごとに集約
            n_chunks = target_len
            chunk_size = seq_len // n_chunks
            
            aggregated = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, seq_len)
                
                if start_idx >= seq_len:
                    # パディング
                    last_bar = aggregated[-1] if aggregated else m1_data[:, -1]
                    aggregated.append(last_bar)
                else:
                    chunk = m1_data[:, start_idx:end_idx]
                    if chunk.size(1) == 0:
                        chunk = m1_data[:, -1:] 
                        
                    open_val = chunk[:, 0, 0]  # 最初の始値
                    high_val = chunk[:, :, 1].max(dim=1)[0]  # 最高値
                    low_val = chunk[:, :, 2].min(dim=1)[0]  # 最安値  
                    close_val = chunk[:, -1, 3]  # 最後の終値
                    
                    bar = torch.stack([open_val, high_val, low_val, close_val], dim=1)
                    aggregated.append(bar)
                    
            return torch.stack(aggregated, dim=1)  # [batch, n_chunks, 4]

class AmplitudePhaseCorrelationLoss(nn.Module):
    """振幅・位相相関損失"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測OHLC
            target: [batch, n_tf, seq_len, 4] 正解OHLC
            
        Returns:
            loss: 振幅・位相相関損失
        """
        batch_size, n_tf, seq_len, n_features = pred.shape
        total_loss = 0.0
        
        for tf_idx in range(n_tf):
            for feat_idx in range(n_features):
                pred_signal = pred[:, tf_idx, :, feat_idx]  # [batch, seq_len]
                target_signal = target[:, tf_idx, :, feat_idx]  # [batch, seq_len]
                
                # FFT計算（16bit混合精度との互換性のため32bitに変換）
                pred_signal_32 = pred_signal.to(torch.float32)
                target_signal_32 = target_signal.to(torch.float32)
                pred_fft = torch.fft.fft(pred_signal_32, dim=-1)
                target_fft = torch.fft.fft(target_signal_32, dim=-1)
                
                # 振幅
                pred_amp = torch.abs(pred_fft)
                target_amp = torch.abs(target_fft)
                
                # 位相
                pred_phase = torch.angle(pred_fft)
                target_phase = torch.angle(target_fft)
                
                # 相関損失
                amp_corr = self._correlation_loss(pred_amp, target_amp)
                phase_corr = self._correlation_loss(pred_phase, target_phase)
                
                total_loss += amp_corr + phase_corr
                
        return total_loss / (n_tf * n_features)
        
    def _correlation_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ピアソン相関係数の負値を損失として計算"""
        
        # 平均除去
        x_centered = x - x.mean(dim=-1, keepdim=True)
        y_centered = y - y.mean(dim=-1, keepdim=True)
        
        # 相関係数計算
        numerator = (x_centered * y_centered).sum(dim=-1)
        x_std = torch.sqrt((x_centered ** 2).sum(dim=-1) + 1e-8)
        y_std = torch.sqrt((y_centered ** 2).sum(dim=-1) + 1e-8)
        
        correlation = numerator / (x_std * y_std + 1e-8)
        
        # 相関が高いほど損失が小さくなるように負値を返す
        return (1 - correlation).mean()

class Stage1CombinedLoss(nn.Module):
    """Stage 1 統合損失関数"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # 損失重み
        self.weights = config['loss']['weights']
        
        # 個別損失関数
        self.huber_loss = HuberLoss(delta=config['loss']['huber_delta'])
        self.stft_loss = STFTLoss(scales=config['loss']['stft_scales'])
        self.cross_loss = CrossTFConsistencyLoss()
        self.amp_phase_loss = AmplitudePhaseCorrelationLoss()
        
        print(f"🎯 Stage1CombinedLoss初期化")
        print(f"   損失重み: {self.weights}")
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        masks: torch.Tensor,
        m1_data: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測OHLC
            target: [batch, n_tf, seq_len, 4] 正解OHLC  
            masks: [batch, n_tf, seq_len] マスク
            m1_data: [batch, seq_len, 4] M1データ（クロス損失用）
            
        Returns:
            losses: {
                'total': 総損失,
                'recon_tf': Huber損失,
                'spec_tf': STFT損失,
                'cross': クロス整合性損失,
                'amp_phase': 振幅位相損失
            }
        """
        
        # 各損失計算
        recon_loss = self.huber_loss(pred, target, masks)
        spec_loss = self.stft_loss(pred, target, masks)
        amp_phase_loss = self.amp_phase_loss(pred, target)
        
        # クロス損失（M1データが提供された場合のみ）
        if m1_data is not None:
            cross_loss = self.cross_loss(pred, m1_data)
        else:
            cross_loss = torch.tensor(0.0, device=pred.device)
            
        # 重み付き総損失
        total_loss = (
            self.weights['recon_tf'] * recon_loss +
            self.weights['spec_tf'] * spec_loss +
            self.weights['cross'] * cross_loss +
            self.weights['amp_phase'] * amp_phase_loss
        )
        
        return {
            'total': total_loss,
            'recon_tf': recon_loss,
            'spec_tf': spec_loss, 
            'cross': cross_loss,
            'amp_phase': amp_phase_loss
        }