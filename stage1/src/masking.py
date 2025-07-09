#!/usr/bin/env python3
"""
Stage 1 マスキング戦略
ランダム連続ブロック・TF間同期マスキング
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MaskingStrategy:
    """マスキング戦略クラス"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.timeframes = config['data']['timeframes']
        
        # マスキング設定
        self.mask_ratio = config['masking']['mask_ratio']  # 0.15
        self.mask_span_min = config['masking']['mask_span_min']  # 5
        self.mask_span_max = config['masking']['mask_span_max']  # 60
        self.sync_across_tf = config['masking']['sync_across_tf']  # True
        
        print(f"🎭 MaskingStrategy初期化")
        print(f"   マスク率: {self.mask_ratio}")
        print(f"   スパン範囲: {self.mask_span_min}-{self.mask_span_max}")
        print(f"   TF間同期: {self.sync_across_tf}")
        
        # 乱数生成器（再現性のため）
        self.rng = np.random.RandomState()
        
    def generate_masks(self, features: torch.Tensor, seed: int = None) -> torch.Tensor:
        """
        マルチTF特徴量に対するマスクを生成
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量テンソル
            seed: ランダムシード（再現性用）
            
        Returns:
            masks: [n_tf, seq_len] マスクテンソル (1=マスク, 0=観測)
        """
        if seed is not None:
            self.rng.seed(seed)
            
        n_tf, seq_len, n_features = features.shape
        masks = torch.zeros(n_tf, seq_len)
        
        if self.sync_across_tf:
            # TF間同期マスキング: M1ベースでマスクを生成し、他のTFに適用
            base_mask = self._generate_single_mask(seq_len)
            
            for i in range(n_tf):
                # 各TFの実際の長さに応じてマスクを調整
                tf_mask = self._adapt_mask_to_tf(base_mask, features[i], seq_len)
                masks[i] = tf_mask
        else:
            # TF個別マスキング
            for i in range(n_tf):
                tf_mask = self._generate_single_mask(seq_len)
                # 各TFの実際の長さに応じて調整
                tf_mask = self._adapt_mask_to_tf(tf_mask, features[i], seq_len)
                masks[i] = tf_mask
                
        return masks
        
    def _generate_single_mask(self, seq_len: int) -> torch.Tensor:
        """
        単一シーケンスに対するマスクを生成
        
        Args:
            seq_len: シーケンス長
            
        Returns:
            mask: [seq_len] マスクテンソル
        """
        mask = torch.zeros(seq_len)
        target_masked = int(seq_len * self.mask_ratio)
        masked_count = 0
        
        # ランダム連続ブロックでマスキング
        while masked_count < target_masked:
            # ランダムなマスクスパン長
            span_len = self.rng.randint(self.mask_span_min, self.mask_span_max + 1)
            
            # ランダムな開始位置
            max_start = max(0, seq_len - span_len)
            if max_start <= 0:
                break
                
            start_pos = self.rng.randint(0, max_start + 1)
            end_pos = min(start_pos + span_len, seq_len)
            
            # マスク適用
            mask[start_pos:end_pos] = 1.0
            masked_count = mask.sum().item()
            
            # 目標マスク数を超えた場合は調整
            if masked_count > target_masked:
                # 超過分をランダムに解除
                masked_indices = torch.where(mask == 1.0)[0]
                excess = int(masked_count - target_masked)
                if excess > 0:
                    remove_indices = masked_indices[torch.randperm(len(masked_indices))[:excess]]
                    mask[remove_indices] = 0.0
                break
                
        return mask
        
    def _adapt_mask_to_tf(self, base_mask: torch.Tensor, tf_features: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        ベースマスクを特定TFの特徴量に適応
        
        Args:
            base_mask: [seq_len] ベースマスク（M1基準）
            tf_features: [seq_len, n_features] TF特徴量
            seq_len: シーケンス長
            
        Returns:
            adapted_mask: [seq_len] 適応済みマスク
        """
        # 実際にデータが存在する部分を特定（右端整列を考慮）
        valid_mask = torch.any(tf_features != 0, dim=1)
        
        if not valid_mask.any():
            # データが存在しない場合はマスクしない
            return torch.zeros(seq_len)
            
        # 有効データ範囲でのみマスキング
        adapted_mask = base_mask.clone()
        adapted_mask = adapted_mask * valid_mask.to(torch.float32)
        
        return adapted_mask
        
    def apply_mask_to_features(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        特徴量にマスクを適用
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量
            masks: [n_tf, seq_len] マスク
            
        Returns:
            masked_features: [n_tf, seq_len, n_features] マスク済み特徴量
        """
        # マスク部分を0に設定
        mask_expanded = masks.unsqueeze(-1)  # [n_tf, seq_len, 1]
        masked_features = features * (1 - mask_expanded)
        
        return masked_features
        
    def get_mask_statistics(self, masks: torch.Tensor) -> Dict:
        """マスク統計を取得（デバッグ用）"""
        
        n_tf, seq_len = masks.shape
        stats = {}
        
        for i, tf in enumerate(self.timeframes):
            tf_mask = masks[i]
            mask_ratio = tf_mask.sum().item() / seq_len
            
            # 連続マスクブロックの検出
            mask_blocks = self._find_mask_blocks(tf_mask)
            
            stats[tf] = {
                'mask_ratio': mask_ratio,
                'masked_tokens': int(tf_mask.sum().item()),
                'total_tokens': seq_len,
                'n_blocks': len(mask_blocks),
                'block_lengths': [end - start for start, end in mask_blocks],
                'avg_block_length': np.mean([end - start for start, end in mask_blocks]) if mask_blocks else 0
            }
            
        return stats
        
    def _find_mask_blocks(self, mask: torch.Tensor) -> List[Tuple[int, int]]:
        """マスクの連続ブロックを検出"""
        
        mask_np = mask.numpy().astype(bool)
        blocks = []
        
        in_block = False
        start = 0
        
        for i, is_masked in enumerate(mask_np):
            if is_masked and not in_block:
                # ブロック開始
                start = i
                in_block = True
            elif not is_masked and in_block:
                # ブロック終了
                blocks.append((start, i))
                in_block = False
                
        # 最後までマスクされている場合
        if in_block:
            blocks.append((start, len(mask_np)))
            
        return blocks