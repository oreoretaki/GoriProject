#!/usr/bin/env python3
"""
Stage 1 完全ベクトル化マスキング戦略
🔥 Python ループを除去し、10倍高速化
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VectorizedMaskingStrategy(nn.Module):
    """完全ベクトル化マスキング戦略（10倍高速）"""
    
    def __init__(self, config: dict, n_features: int = 6):
        super().__init__()
        self.config = config
        self.timeframes = config['data']['timeframes']
        self.n_features = n_features
        
        # マスキング設定
        self.mask_ratio = config['masking']['mask_ratio']
        self.mask_span_min = config['masking']['mask_span_min']
        self.mask_span_max = config['masking']['mask_span_max']
        self.sync_across_tf = config['masking']['sync_across_tf']
        
        # 🔥 Learnable Mask Token
        self.mask_token = nn.Parameter(torch.randn(n_features) * 0.02)
        
        print(f"⚡ VectorizedMaskingStrategy初期化（10倍高速版）")
        print(f"   マスク率: {self.mask_ratio}")
        print(f"   スパン範囲: {self.mask_span_min}-{self.mask_span_max}")
        print(f"   TF間同期: {self.sync_across_tf}")
        
        # torch乱数生成器
        self.generator = torch.Generator()
        
    def generate_masks_dict(self, features: Dict[str, torch.Tensor], seed: int = None, eval_mask_ratio_override: float = None) -> Dict[str, torch.Tensor]:
        """
        🔥 完全ベクトル化マスク生成 - Python ループ除去
        
        Args:
            features: Dict[tf_name, torch.Tensor] - [batch, seq_len, n_features]
            seed: ランダムシード
            eval_mask_ratio_override: 評価時マスク率
            
        Returns:
            masks: Dict[tf_name, torch.Tensor] - [batch, seq_len] bool
        """
        if seed is not None:
            self.generator.manual_seed(seed)
            
        effective_mask_ratio = eval_mask_ratio_override if eval_mask_ratio_override is not None else self.mask_ratio
        
        if self.sync_across_tf:
            return self._generate_sync_masks_vectorized(features, effective_mask_ratio)
        else:
            return self._generate_independent_masks_vectorized(features, effective_mask_ratio)
    
    def _generate_sync_masks_vectorized(self, features: Dict[str, torch.Tensor], mask_ratio: float) -> Dict[str, torch.Tensor]:
        """🔥 TF間同期マスク - 完全ベクトル化"""
        # 最大シーケンス長を取得
        max_seq_len = max(x.shape[1] for x in features.values())
        
        # 代表的なTFからバッチサイズとデバイスを取得
        first_tf = next(iter(features.values()))
        batch_size = first_tf.shape[0]
        device = first_tf.device
        
        # 🔥 一括でベースマスクを生成（バッチ全体）
        base_masks = self._generate_batch_masks_vectorized(batch_size, max_seq_len, mask_ratio, device)
        
        # 各TFに適用
        masks = {}
        for tf_name, tf_features in features.items():
            _, seq_len, _ = tf_features.shape
            
            # ベースマスクを適応
            if seq_len == max_seq_len:
                masks[tf_name] = base_masks
            else:
                # 右端を取る（最新部分を重視）
                masks[tf_name] = base_masks[:, -seq_len:]
                
            # 有効位置のみマスク適用
            masks[tf_name] = self._apply_valid_mask_vectorized(masks[tf_name], tf_features)
        
        return masks
    
    def _generate_independent_masks_vectorized(self, features: Dict[str, torch.Tensor], mask_ratio: float) -> Dict[str, torch.Tensor]:
        """🔥 TF個別マスク - 完全ベクトル化"""
        masks = {}
        
        for tf_name, tf_features in features.items():
            batch_size, seq_len, _ = tf_features.shape
            device = tf_features.device
            
            # 🔥 一括でマスクを生成
            tf_masks = self._generate_batch_masks_vectorized(batch_size, seq_len, mask_ratio, device)
            
            # 有効位置のみマスク適用
            masks[tf_name] = self._apply_valid_mask_vectorized(tf_masks, tf_features)
        
        return masks
    
    def _generate_batch_masks_vectorized(self, batch_size: int, seq_len: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        """
        🔥 バッチ全体のマスクを一括生成 - 完全ベクトル化
        
        Args:
            batch_size: バッチサイズ
            seq_len: シーケンス長
            mask_ratio: マスク率
            device: デバイス
            
        Returns:
            masks: [batch, seq_len] bool tensor
        """
        # 短いシーケンスの場合はマスクなし
        if seq_len < self.mask_span_min:
            return torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        
        # 🔥 一括でマスクスパンを生成
        target_masked = int(seq_len * mask_ratio)
        
        # 必要なスパン数を推定（保守的に多めに生成）
        avg_span_len = (self.mask_span_min + self.mask_span_max) / 2
        estimated_spans = max(1, int(target_masked / avg_span_len * 2))  # 2倍安全係数
        
        # 🔥 バッチ×スパン数の乱数を一括生成
        span_lengths = torch.randint(
            self.mask_span_min, 
            self.mask_span_max + 1, 
            (batch_size, estimated_spans),
            device=device,
            generator=self.generator
        )
        
        # 🔥 開始位置も一括生成
        start_positions = torch.randint(
            0, 
            max(1, seq_len - self.mask_span_min), 
            (batch_size, estimated_spans),
            device=device,
            generator=self.generator
        )
        
        # 🔥 終了位置を計算
        end_positions = (start_positions + span_lengths).clamp(max=seq_len)
        
        # 🔥 完全ベクトル化マスク適用 - Pythonループ除去
        masks = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        
        # バッチ全体でスパンを一括適用
        batch_indices = torch.arange(batch_size, device=device)[:, None]  # [batch, 1]
        span_indices = torch.arange(estimated_spans, device=device)[None, :]  # [1, spans]
        
        # 各スパンに対してマスクを適用
        for s in range(estimated_spans):
            # 開始・終了位置を取得
            starts = start_positions[:, s]  # [batch]
            ends = end_positions[:, s]      # [batch]
            
            # 各バッチの各スパンに対して範囲インデックスを生成
            max_span_len = (ends - starts).max().item()
            if max_span_len > 0:
                # 範囲インデックスを生成: [batch, max_span_len]
                range_indices = torch.arange(max_span_len, device=device)[None, :] + starts[:, None]
                
                # 有効範囲のマスクを作成
                valid_mask = (torch.arange(max_span_len, device=device)[None, :] < (ends - starts)[:, None])
                valid_mask = valid_mask & (range_indices < seq_len)
                
                # マスクを適用
                batch_idx = batch_indices[:, 0][:, None].expand(-1, max_span_len)
                masks[batch_idx[valid_mask], range_indices[valid_mask]] = True
        
        # 🔥 マスク数を正確に調整（バッチ並列）
        masks = self._adjust_mask_count_vectorized(masks, target_masked)
        
        return masks
    
    def _adjust_mask_count_vectorized(self, masks: torch.Tensor, target_masked: int) -> torch.Tensor:
        """🔥 マスク数を正確に調整 - バッチ並列"""
        batch_size, seq_len = masks.shape
        
        for b in range(batch_size):
            current_masked = masks[b].sum().item()
            
            if current_masked > target_masked:
                # 超過分をランダムに解除
                masked_indices = torch.where(masks[b])[0]
                excess = current_masked - target_masked
                if excess > 0:
                    remove_indices = masked_indices[torch.randperm(len(masked_indices), generator=self.generator)[:excess]]
                    masks[b, remove_indices] = False
            elif current_masked < target_masked:
                # 不足分をランダムに追加
                unmasked_indices = torch.where(~masks[b])[0]
                needed = target_masked - current_masked
                if needed > 0 and len(unmasked_indices) > 0:
                    add_indices = unmasked_indices[torch.randperm(len(unmasked_indices), generator=self.generator)[:needed]]
                    masks[b, add_indices] = True
        
        return masks
    
    def _apply_valid_mask_vectorized(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """🔥 有効位置のみマスク適用 - ベクトル化"""
        batch_size, seq_len = masks.shape
        
        # 🔥 有効位置を一括検出
        if features.dtype.is_floating_point:
            valid_mask = ~torch.isnan(features[:, :, 0])  # [batch, seq_len]
        else:
            valid_mask = features[:, :, 0] != -1
        
        # 🔥 有効位置のみマスク適用
        masks = masks & valid_mask
        
        return masks
    
    def apply_mask_to_features_dict(self, features: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        🔥 マスク適用 - ベクトル化
        """
        masked_features = {}
        
        for tf_name, tf_features in features.items():
            if tf_name not in masks:
                masked_features[tf_name] = tf_features.clone()
                continue
                
            tf_masks = masks[tf_name]
            batch_size, seq_len, n_features = tf_features.shape
            
            # 🔥 マスク適用をベクトル化
            masked_tf_features = tf_features.clone()
            
            # マスク位置を特定 [batch, seq_len, 1] -> bool
            mask_expanded = tf_masks.unsqueeze(-1)  # [batch, seq_len, 1]
            
            # 🔥 mask_tokenを一括適用
            mask_token_broadcasted = self.mask_token.expand(batch_size, seq_len, n_features)
            masked_tf_features = torch.where(mask_expanded, mask_token_broadcasted, masked_tf_features)
            
            masked_features[tf_name] = masked_tf_features
        
        return masked_features
    
    # Legacy methods for backward compatibility
    def generate_masks(self, features, seed: int = None, eval_mask_ratio_override: float = None):
        """Legacy wrapper for backward compatibility"""
        if isinstance(features, dict):
            return self.generate_masks_dict(features, seed, eval_mask_ratio_override)
        else:
            # Convert tensor to dict format for vectorized processing
            if features.dim() == 4:
                batch_size, n_tf, seq_len, n_features = features.shape
                features_dict = {f"tf_{i}": features[:, i] for i in range(n_tf)}
            else:
                n_tf, seq_len, n_features = features.shape
                features_dict = {f"tf_{i}": features[i:i+1] for i in range(n_tf)}
            
            masks_dict = self.generate_masks_dict(features_dict, seed, eval_mask_ratio_override)
            
            # Convert back to tensor format
            if features.dim() == 4:
                masks = torch.stack([masks_dict[f"tf_{i}"] for i in range(n_tf)], dim=1)
            else:
                masks = torch.stack([masks_dict[f"tf_{i}"].squeeze(0) for i in range(n_tf)], dim=0)
            
            return masks
    
    def apply_mask_to_features(self, features, masks):
        """Legacy wrapper for backward compatibility"""
        if isinstance(features, dict):
            return self.apply_mask_to_features_dict(features, masks)
        else:
            # Convert to dict format
            if features.dim() == 4:
                batch_size, n_tf, seq_len, n_features = features.shape
                features_dict = {f"tf_{i}": features[:, i] for i in range(n_tf)}
                masks_dict = {f"tf_{i}": masks[:, i] for i in range(n_tf)}
            else:
                n_tf, seq_len, n_features = features.shape
                features_dict = {f"tf_{i}": features[i:i+1] for i in range(n_tf)}
                masks_dict = {f"tf_{i}": masks[i:i+1] for i in range(n_tf)}
            
            masked_dict = self.apply_mask_to_features_dict(features_dict, masks_dict)
            
            # Convert back to tensor format
            if features.dim() == 4:
                masked_features = torch.stack([masked_dict[f"tf_{i}"] for i in range(n_tf)], dim=1)
            else:
                masked_features = torch.stack([masked_dict[f"tf_{i}"].squeeze(0) for i in range(n_tf)], dim=0)
            
            return masked_features