#!/usr/bin/env python3
"""
Stage 1 マスキング戦略
ランダム連続ブロック・TF間同期マスキング
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class MaskingStrategy(nn.Module):
    """マスキング戦略クラス（Learnable Mask Token版）"""
    
    def __init__(self, config: dict, n_features: int = 6):
        """
        Args:
            config: 設定辞書
            n_features: 特徴量数（mask_tokenの次元）
        """
        super().__init__()
        self.config = config
        self.timeframes = config['data']['timeframes']
        self.n_features = n_features
        
        # マスキング設定
        self.mask_ratio = config['masking']['mask_ratio']  # 0.15
        self.mask_span_min = config['masking']['mask_span_min']  # 5
        self.mask_span_max = config['masking']['mask_span_max']  # 60
        self.sync_across_tf = config['masking']['sync_across_tf']  # True
        
        # 🔥 Learnable Mask Token（学習可能なマスクトークン）
        self.mask_token = nn.Parameter(torch.randn(n_features) * 0.02)
        
        print(f"🎭 MaskingStrategy初期化（Learnable Mask Token版）")
        print(f"   マスク率: {self.mask_ratio}")
        print(f"   スパン範囲: {self.mask_span_min}-{self.mask_span_max}")
        print(f"   TF間同期: {self.sync_across_tf}")
        print(f"   💡 Learnable Mask Token: {n_features}次元（初期値: μ={self.mask_token.mean():.3f}, σ={self.mask_token.std():.3f}）")
        
        # 乱数生成器（再現性のため）
        self.rng = np.random.RandomState()
        
    def generate_masks(self, features: torch.Tensor, seed: int = None, eval_mask_ratio_override: float = None) -> torch.Tensor:
        """
        マルチTF特徴量に対するマスクを生成
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量テンソル
            seed: ランダムシード（再現性用）
            eval_mask_ratio_override: 評価時のマスク率上書き (None=通常, 0=マスクなし, 1=全マスク)
            
        Returns:
            masks: [n_tf, seq_len] マスクテンソル (1=マスク, 0=観測)
        """
        if seed is not None:
            self.rng.seed(seed)
            
        # featuresの形状を確認: [batch, n_tf, seq_len, n_features] または [n_tf, seq_len, n_features]
        if features.dim() == 4:
            batch_size, n_tf, seq_len, n_features = features.shape
        elif features.dim() == 3:
            n_tf, seq_len, n_features = features.shape
            batch_size = 1
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")
        
        # 評価時のマスク率オーバーライド処理
        effective_mask_ratio = self.mask_ratio
        if eval_mask_ratio_override is not None:
            effective_mask_ratio = eval_mask_ratio_override
            print(f"   [MASK DBG] Override: {self.mask_ratio} → {effective_mask_ratio}")
            
        # 🔥 バッチサイズに応じてマスクの形状を決定（bool型で統一）
        if features.dim() == 4:
            masks = torch.zeros(batch_size, n_tf, seq_len, device=features.device, dtype=torch.bool)
        else:
            masks = torch.zeros(n_tf, seq_len, device=features.device, dtype=torch.bool)
        
        if self.sync_across_tf:
            # TF間同期マスキング: M1ベースでマスクを生成し、他のTFに適用
            base_mask = self._generate_single_mask(seq_len, effective_mask_ratio)
            
            if features.dim() == 4:
                # バッチ処理
                base_mask = base_mask.to(features.device)
                for b in range(batch_size):
                    for i in range(n_tf):
                        tf_mask = self._adapt_mask_to_tf(base_mask, features[b, i], seq_len)
                        masks[b, i] = tf_mask
            else:
                # 単一サンプル処理
                base_mask = base_mask.to(features.device)
                for i in range(n_tf):
                    tf_mask = self._adapt_mask_to_tf(base_mask, features[i], seq_len)
                    masks[i] = tf_mask
        else:
            # TF個別マスキング
            if features.dim() == 4:
                # バッチ処理
                for b in range(batch_size):
                    for i in range(n_tf):
                        tf_mask = self._generate_single_mask(seq_len, effective_mask_ratio)
                        tf_mask = tf_mask.to(features.device)
                        tf_mask = self._adapt_mask_to_tf(tf_mask, features[b, i], seq_len)
                        masks[b, i] = tf_mask
            else:
                # 単一サンプル処理
                for i in range(n_tf):
                    tf_mask = self._generate_single_mask(seq_len, effective_mask_ratio)
                    tf_mask = tf_mask.to(features.device)
                    tf_mask = self._adapt_mask_to_tf(tf_mask, features[i], seq_len)
                    masks[i] = tf_mask
        
        # デバッグ: 実際のマスク率を確認
        if eval_mask_ratio_override is not None:
            actual_ratios = []
            for i in range(n_tf):
                mask_i = masks[i] if masks.dim() == 2 else masks[0, i]
                actual_ratio = mask_i.mean().item()
                actual_ratios.append(actual_ratio)
                print(f"   [MASK DBG] TF{i} actual mask ratio: {actual_ratio:.4f}")
            print(f"   [MASK DBG] Mean actual mask ratio: {sum(actual_ratios)/len(actual_ratios):.4f}")
                
        return masks
        
    def _generate_single_mask(self, seq_len: int, mask_ratio: float = None) -> torch.Tensor:
        """
        単一シーケンスに対するマスクを生成
        
        Args:
            seq_len: シーケンス長
            mask_ratio: マスク率（Noneの場合はself.mask_ratioを使用）
            
        Returns:
            mask: [seq_len] マスクテンソル
        """
        mask = torch.zeros(seq_len, dtype=torch.bool)  # 🔥 bool型で統一
        effective_mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        target_masked = int(seq_len * effective_mask_ratio)
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
            mask[start_pos:end_pos] = True
            masked_count = mask.sum().item()
            
            # 目標マスク数を超えた場合は調整
            if masked_count > target_masked:
                # 超過分をランダムに解除
                masked_indices = torch.where(mask)[0]  # bool型対応
                excess = int(masked_count - target_masked)
                if excess > 0:
                    remove_indices = masked_indices[torch.randperm(len(masked_indices))[:excess]]
                    mask[remove_indices] = False
                break
                
        return mask  # 🔥 既にbool型なのでそのまま返す
        
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
            return torch.zeros(seq_len, dtype=torch.bool)
            
        # 有効データ範囲でのみマスキング
        adapted_mask = base_mask.clone()
        adapted_mask = adapted_mask & valid_mask  # bool演算で統一
        
        return adapted_mask
        
    def apply_mask_to_features(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        特徴量にマスクを適用（Learnable Mask Token版）
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量
            masks: [n_tf, seq_len] マスク（1=マスク, 0=観測）
            
        Returns:
            masked_features: [n_tf, seq_len, n_features] マスク済み特徴量
        """
        # 特徴量をコピー（inplace操作のため）
        masked_features = features.clone()
        
        # マスク位置を特定 [n_tf, seq_len, 1] -> bool
        mask_expanded = masks.unsqueeze(-1).bool()  # [n_tf, seq_len, 1]
        
        # 🔥 マスク位置にLearnable Mask Tokenを設定（0乗算ではなくinplace置換）
        # mask_tokenを各マスク位置に適用
        n_tf, seq_len, n_features = features.shape
        mask_token_expanded = self.mask_token.unsqueeze(0).unsqueeze(0)  # [1, 1, n_features]
        mask_token_broadcasted = mask_token_expanded.expand(n_tf, seq_len, n_features)
        
        # マスク位置のみを置換
        masked_features[mask_expanded.expand_as(features)] = mask_token_broadcasted[mask_expanded.expand_as(features)]
        
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