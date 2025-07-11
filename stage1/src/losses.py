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
        pred, 
        target, 
        masks,
        m1_data = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: [batch, n_tf, seq_len, 4] 予測OHLC または Dict[str, torch.Tensor]
            target: [batch, n_tf, seq_len, 4] 正解OHLC または Dict[str, torch.Tensor]
            masks: [batch, n_tf, seq_len] マスク または Dict[str, torch.Tensor]
            m1_data: [batch, seq_len, 4] M1データ（クロス損失用）または Dict[str, torch.Tensor]
            
        Returns:
            losses: {
                'total': 総損失,
                'recon_tf': Huber損失,
                'spec_tf': STFT損失,
                'cross': クロス整合性損失,
                'amp_phase': 振幅位相損失
            }
        """
        # Dict format support for Model v2
        if isinstance(pred, dict):
            return self._forward_dict(pred, target, masks, m1_data)
        
        # Legacy tensor format support
        
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
    
    def _forward_dict(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor], 
        masks: Dict[str, torch.Tensor],
        m1_data: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Dict形式の入力に対する損失計算 (Model v2用)
        
        Args:
            pred: Dict[tf_name, torch.Tensor] - 各TFの予測 [batch, seq_len, 4]
            target: Dict[tf_name, torch.Tensor] - 各TFの正解 [batch, seq_len, 4]
            masks: Dict[tf_name, torch.Tensor] - 各TFのマスク [batch, seq_len]
            m1_data: Dict[tf_name, torch.Tensor] - M1データ（クロス損失用）
            
        Returns:
            losses: Dict[str, torch.Tensor] - 各損失の辞書
        """
        device = list(pred.values())[0].device
        
        # 各損失計算
        recon_loss = self._huber_loss_dict(pred, target, masks)
        spec_loss = self._stft_loss_dict(pred, target, masks)
        amp_phase_loss = self._amp_phase_loss_dict(pred, target)
        
        # クロス損失（M1データが提供された場合のみ）
        if m1_data is not None and 'm1' in pred:
            cross_loss = self._cross_loss_dict(pred, m1_data)
        elif 'm1' in pred:
            # 🔥 堅牢化: M1データが提供されていないが、M1予測がある場合
            # predのM1をdetachしてm1_dataとして使用
            m1_fallback = {'m1': pred['m1'].detach()}
            cross_loss = self._cross_loss_dict(pred, m1_fallback)
        else:
            cross_loss = torch.tensor(0.0, device=device)
            
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
    
    def _huber_loss_dict(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dict形式のHuber損失計算"""
        total_loss = 0.0
        total_count = 0
        
        for tf_name, pred_tf in pred.items():
            if tf_name not in target:
                continue
                
            target_tf = target[tf_name]
            mask_tf = masks.get(tf_name, None) if masks is not None else None
            
            # NaN値を除外（padding対応）
            batch_size_tf, seq_len_tf = pred_tf.shape[:2]
            valid_mask = ~torch.isnan(pred_tf[..., 0])  # [batch, seq_len_tf]
            
            if mask_tf is not None:
                # mask_tfがvalid_maskと同じshapeでない場合は調整
                if mask_tf.shape != valid_mask.shape:
                    # mask_tfをtarget_tfの実際の形状に合わせる
                    if mask_tf.shape[1] > seq_len_tf:
                        mask_tf = mask_tf[:, :seq_len_tf]  # truncate
                    elif mask_tf.shape[1] < seq_len_tf:
                        # padding with False (not masked)
                        pad_width = seq_len_tf - mask_tf.shape[1]
                        mask_tf = torch.cat([mask_tf, torch.zeros(batch_size_tf, pad_width, dtype=torch.bool, device=mask_tf.device)], dim=1)
                
                valid_mask = valid_mask & ~mask_tf  # マスクされた位置も除外
            
            if valid_mask.sum() == 0:
                continue
                
            # 有効な位置のみで損失計算（pred と target は同じ長さで来る前提）
            nan_mask = ~torch.isnan(target_tf).any(dim=-1)  # [batch, seq_len_tf]
            
            # 有効な位置が存在するかチェック
            if nan_mask.sum() == 0:
                continue
                
            pred_valid = pred_tf[nan_mask]  # [valid_positions, 4]
            target_valid = target_tf[nan_mask]  # [valid_positions, 4]
            
            loss = F.huber_loss(pred_valid, target_valid, delta=self.huber_loss.delta, reduction='mean')
            total_loss += loss
            total_count += 1
            
        return total_loss / max(total_count, 1)
    
    def _stft_loss_dict(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dict形式のSTFT損失計算"""
        total_loss = 0.0
        total_count = 0
        
        for tf_name, pred_tf in pred.items():
            if tf_name not in target:
                continue
                
            target_tf = target[tf_name]
            mask_tf = masks.get(tf_name, None) if masks is not None else None
            
            batch_size, seq_len, n_features = pred_tf.shape
            
            # NaN値を除外（padding対応）
            batch_size_tf, seq_len_tf = pred_tf.shape[:2]
            valid_mask = ~torch.isnan(pred_tf[..., 0])  # [batch, seq_len_tf]
            
            if mask_tf is not None:
                # mask_tfがvalid_maskと同じshapeでない場合は調整
                if mask_tf.shape != valid_mask.shape:
                    # mask_tfをtarget_tfの実際の形状に合わせる
                    if mask_tf.shape[1] > seq_len_tf:
                        mask_tf = mask_tf[:, :seq_len_tf]  # truncate
                    elif mask_tf.shape[1] < seq_len_tf:
                        # padding with False (not masked)
                        pad_width = seq_len_tf - mask_tf.shape[1]
                        mask_tf = torch.cat([mask_tf, torch.zeros(batch_size_tf, pad_width, dtype=torch.bool, device=mask_tf.device)], dim=1)
                
                valid_mask = valid_mask & ~mask_tf  # マスクされた位置も除外
            
            for feat_idx in range(n_features):
                pred_signal = pred_tf[:, :, feat_idx]  # [batch, seq_len]
                target_signal = target_tf[:, :, feat_idx]  # [batch, seq_len]
                
                # 各バッチで有効な部分のみ取得
                for b in range(batch_size):
                    valid_positions = valid_mask[b]
                    if valid_positions.sum() == 0:
                        continue
                        
                    pred_seq = pred_signal[b, valid_positions]  # [valid_len]
                    target_seq = target_signal[b, valid_positions]  # [valid_len]
                    
                    if len(pred_seq) == 0 or len(pred_seq) < 64:  # STFT計算に必要な最小長
                        continue
                        
                    # 各スケールでSTFT損失計算
                    for scale in self.stft_loss.scales:
                        if len(pred_seq) < scale:
                            continue
                            
                        hop_length = int(scale * self.stft_loss.hop_ratio)
                        
                        # STFT計算
                        pred_seq_32 = pred_seq.to(torch.float32)
                        target_seq_32 = target_seq.to(torch.float32)
                        pred_stft = torch.stft(
                            pred_seq_32, 
                            n_fft=scale, 
                            hop_length=hop_length, 
                            return_complex=True
                        )
                        target_stft = torch.stft(
                            target_seq_32, 
                            n_fft=scale, 
                            hop_length=hop_length, 
                            return_complex=True
                        )
                        
                        # マグニチュード損失
                        pred_mag = torch.abs(pred_stft)
                        target_mag = torch.abs(target_stft)
                        mag_loss = F.l1_loss(pred_mag, target_mag)
                        
                        # 位相損失
                        phase_loss = F.l1_loss(pred_stft.real, target_stft.real) + \
                                    F.l1_loss(pred_stft.imag, target_stft.imag)
                        
                        total_loss += mag_loss + 0.1 * phase_loss
                        total_count += 1
                        
        return total_loss / max(total_count, 1)
    
    def _amp_phase_loss_dict(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dict形式の振幅位相損失計算"""
        total_loss = 0.0
        total_count = 0
        
        for tf_name, pred_tf in pred.items():
            if tf_name not in target:
                continue
                
            target_tf = target[tf_name]
            batch_size, seq_len, n_features = pred_tf.shape
            
            # NaN値を除外（padding対応）
            batch_size_tf, seq_len_tf = pred_tf.shape[:2]
            valid_mask = ~torch.isnan(pred_tf[..., 0])  # [batch, seq_len_tf]
            
            for feat_idx in range(n_features):
                pred_signal = pred_tf[:, :, feat_idx]  # [batch, seq_len]
                target_signal = target_tf[:, :, feat_idx]  # [batch, seq_len]
                
                # 各バッチで有効な部分のみ取得
                for b in range(batch_size):
                    valid_positions = valid_mask[b]
                    if valid_positions.sum() == 0:
                        continue
                        
                    pred_seq = pred_signal[b, valid_positions]  # [valid_len]
                    target_seq = target_signal[b, valid_positions]  # [valid_len]
                    
                    if len(pred_seq) < 8:  # FFT計算に必要な最小長
                        continue
                        
                    # FFT計算
                    pred_seq_32 = pred_seq.to(torch.float32)
                    target_seq_32 = target_seq.to(torch.float32)
                    pred_fft = torch.fft.fft(pred_seq_32)
                    target_fft = torch.fft.fft(target_seq_32)
                    
                    # 振幅・位相
                    pred_amp = torch.abs(pred_fft)
                    target_amp = torch.abs(target_fft)
                    pred_phase = torch.angle(pred_fft)
                    target_phase = torch.angle(target_fft)
                    
                    # 相関損失
                    amp_corr = self.amp_phase_loss._correlation_loss(pred_amp.unsqueeze(0), target_amp.unsqueeze(0))
                    phase_corr = self.amp_phase_loss._correlation_loss(pred_phase.unsqueeze(0), target_phase.unsqueeze(0))
                    
                    total_loss += amp_corr + phase_corr
                    total_count += 1
                    
        return total_loss / max(total_count, 1)
    
    def _cross_loss_dict(self, pred: Dict[str, torch.Tensor], m1_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dict形式のクロス整合性損失計算"""
        if 'm1' not in pred or 'm1' not in m1_data:
            return torch.tensor(0.0, device=list(pred.values())[0].device)
            
        pred_m1 = pred['m1']  # [batch, seq_len, 4]
        m1_ref = m1_data['m1']  # [batch, seq_len, 4]
        
        total_loss = 0.0
        total_count = 0
        
        # M1以外のTFについて、対応するM1集約と比較
        tf_intervals = {'m5': 5, 'm15': 15, 'm30': 30, 'h1': 60, 'h4': 240}
        
        for tf_name, pred_tf in pred.items():
            if tf_name == 'm1' or tf_name not in tf_intervals:
                continue
                
            interval = tf_intervals[tf_name]
            
            # M1から期待される集約値を計算
            expected_tf = self._aggregate_m1_to_tf_dict(m1_ref, interval, pred_tf.shape[1])
            
            # NaN値を除外
            batch_size_tf, seq_len_tf = pred_tf.shape[:2]
            valid_mask = ~torch.isnan(pred_tf[..., 0])  # [batch, seq_len_tf]
            
            # expected_tfの形状をpred_tfに合わせる
            if expected_tf.shape[1] != seq_len_tf:
                if expected_tf.shape[1] > seq_len_tf:
                    expected_tf = expected_tf[:, :seq_len_tf]  # truncate
                elif expected_tf.shape[1] < seq_len_tf:
                    # padding with NaN
                    pad_width = seq_len_tf - expected_tf.shape[1]
                    pad_tensor = torch.full((batch_size_tf, pad_width, 4), float('nan'), device=expected_tf.device, dtype=expected_tf.dtype)
                    expected_tf = torch.cat([expected_tf, pad_tensor], dim=1)
            
            if valid_mask.sum() == 0:
                continue
                
            # 有効な位置のみで損失計算
            pred_valid = pred_tf[valid_mask]  # [valid_positions, 4]
            expected_valid = expected_tf[valid_mask]  # [valid_positions, 4]
            
            tf_loss = F.mse_loss(pred_valid, expected_valid)
            total_loss += tf_loss
            total_count += 1
            
        return total_loss / max(total_count, 1)
    
    def _aggregate_m1_to_tf_dict(self, m1_data: torch.Tensor, interval: int, target_len: int) -> torch.Tensor:
        """
        Dict形式のM1データを指定間隔で集約
        
        Args:
            m1_data: [batch, seq_len, 4] M1 OHLC
            interval: 集約間隔
            target_len: 出力シーケンス長
            
        Returns:
            aggregated: [batch, target_len, 4] 集約済みOHLC
        """
        # staticmethod版を直接実装（再帰回避）
        return self._aggregate_m1_to_tf_static(m1_data, interval, target_len)
    
    @staticmethod
    def _aggregate_m1_to_tf_static(m1_data: torch.Tensor, interval: int, target_len: int) -> torch.Tensor:
        """
        M1データを指定間隔で集約（static版）
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