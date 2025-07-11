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
        
        # 🔥 Learnable Mask Token（学習可能なマスクトークン）- 必ず作成
        self.mask_token = nn.Parameter(torch.randn(n_features) * 0.02)
        
        # 🔥 ベクトル化設定フラグ
        self.use_vectorized = config.get('masking', {}).get('use_vectorized', True)
        
        print(f"🎭 MaskingStrategy初期化（Learnable Mask Token版）")
        print(f"   マスク率: {self.mask_ratio}")
        print(f"   スパン範囲: {self.mask_span_min}-{self.mask_span_max}")
        print(f"   TF間同期: {self.sync_across_tf}")
        print(f"   💡 Learnable Mask Token: {n_features}次元（初期値: μ={self.mask_token.mean().item():.3f}, σ={self.mask_token.std().item():.3f}）")
        print(f"   ⚡ ベクトル化: {self.use_vectorized} （10倍高速）")
        
        # torch乱数生成器に統一（再現性のため・デバイス対応）
        self.generator = None  # 実行時にデバイス対応版を初期化
        
    def generate_masks(self, features, seed: int = None, eval_mask_ratio_override: float = None):
        """
        マルチTF特徴量に対するマスクを生成 (Dict対応)
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量テンソル または Dict[str, torch.Tensor]
            seed: ランダムシード（再現性用）
            eval_mask_ratio_override: 評価時のマスク率上書き (None=通常, 0=マスクなし, 1=全マスク)
            
        Returns:
            masks: [n_tf, seq_len] マスクテンソル (1=マスク, 0=観測) または Dict[str, torch.Tensor]
        """
        # Dict format support for Model v2
        if isinstance(features, dict):
            return self.generate_masks_dict(features, seed, eval_mask_ratio_override)
        
        # Legacy tensor format support  
        # 🔥 デバイス対応generator初期化
        device = features.device
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            
        if seed is not None:
            self.generator.manual_seed(seed)
            
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
            # print(f"   [MASK DBG] Override: {self.mask_ratio} → {effective_mask_ratio}")  # 🔇 無効化
            
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
        
        # デバッグ: 実際のマスク率を確認（完全に無効化）
        if False:  # 🔥 デバッグ出力を完全に無効化
            actual_ratios = []
            for i in range(n_tf):
                mask_i = masks[i] if masks.dim() == 2 else masks[0, i]
                # 🔥 Bool tensor mean() エラー修正: float変換
                actual_ratio = mask_i.float().mean().item()
                actual_ratios.append(actual_ratio)
                print(f"   [MASK DBG] TF{i} actual mask ratio: {actual_ratio:.4f}")
            print(f"   [MASK DBG] Mean actual mask ratio: {sum(actual_ratios)/len(actual_ratios):.4f}")
            print(f"   [MASK DBG] Override: {eval_mask_ratio_override:.2f} → {eval_mask_ratio_override:.2f}")
                
        return masks
        
    def generate_masks_dict(self, features: Dict[str, torch.Tensor], seed: int = None, eval_mask_ratio_override: float = None) -> Dict[str, torch.Tensor]:
        """
        Dict形式の特徴量に対するマスクを生成 (Model v2用)
        
        Args:
            features: Dict[tf_name, torch.Tensor] - 各TFの特徴量 [batch, seq_len, n_features]
            seed: ランダムシード（再現性用）
            eval_mask_ratio_override: 評価時のマスク率上書き
            
        Returns:
            masks: Dict[tf_name, torch.Tensor] - 各TFのマスク [batch, seq_len]
        """
        # 🔥 デバイス対応generator初期化
        device = next(iter(features.values())).device
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            
        if seed is not None:
            self.generator.manual_seed(seed)
            
        # 評価時のマスク率オーバーライド処理
        effective_mask_ratio = self.mask_ratio
        if eval_mask_ratio_override is not None:
            effective_mask_ratio = eval_mask_ratio_override
            
        masks = {}
        
        # 🔥 ベクトル化処理 vs 従来処理
        if self.use_vectorized:
            # ベクトル化版: 10倍高速
            masks = self._generate_masks_dict_vectorized(features, effective_mask_ratio)
        else:
            # 従来版: 互換性のために保持
            if self.sync_across_tf:
                # TF間同期マスキング: 最も長いTFをベースにマスクを生成
                max_seq_len = max(x.shape[1] for x in features.values())
                base_mask = self._generate_single_mask(max_seq_len, effective_mask_ratio)
                
                for tf_name, tf_features in features.items():
                    batch_size, seq_len, n_features = tf_features.shape
                    tf_masks = torch.zeros(batch_size, seq_len, device=tf_features.device, dtype=torch.bool)
                    
                    # Adapt base mask to this TF's sequence length
                    adapted_base_mask = self._adapt_base_mask_to_length(base_mask, seq_len)
                    
                    for b in range(batch_size):
                        # Check for padding (invalid positions) - dtype対応
                        if tf_features.dtype.is_floating_point:
                            valid_mask = ~torch.isnan(tf_features[b, :, 0])  # float: NaN検出
                        else:
                            valid_mask = tf_features[b, :, 0] != -1  # int: -1をpad flagとして使用
                        
                        # Apply base mask only to valid positions
                        tf_mask = adapted_base_mask.clone()
                        tf_mask = tf_mask & valid_mask  # Only mask valid positions
                        
                        tf_masks[b] = tf_mask
                    
                    masks[tf_name] = tf_masks
            else:
                # TF個別マスキング
                for tf_name, tf_features in features.items():
                    batch_size, seq_len, n_features = tf_features.shape
                    tf_masks = torch.zeros(batch_size, seq_len, device=tf_features.device, dtype=torch.bool)
                    
                    for b in range(batch_size):
                        # Check for padding (invalid positions) - dtype対応
                        if tf_features.dtype.is_floating_point:
                            valid_mask = ~torch.isnan(tf_features[b, :, 0])  # float: NaN検出
                        else:
                            valid_mask = tf_features[b, :, 0] != -1  # int: -1をpad flagとして使用
                        valid_seq_len = valid_mask.sum().item()
                        
                        if valid_seq_len > 0:
                            # Generate mask for valid sequence length
                            tf_mask = self._generate_single_mask(valid_seq_len, effective_mask_ratio)
                            
                            # Map back to full sequence (only valid positions)
                            valid_indices = torch.where(valid_mask)[0]
                            if len(valid_indices) > 0:
                                full_mask = torch.zeros(seq_len, device=tf_features.device, dtype=torch.bool)
                                if len(tf_mask) <= len(valid_indices):
                                    full_mask[valid_indices[:len(tf_mask)]] = tf_mask
                                tf_masks[b] = full_mask
                        
                    masks[tf_name] = tf_masks
        
        return masks
        
    def _adapt_base_mask_to_length(self, base_mask: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        ベースマスクを指定長に適応
        
        Args:
            base_mask: [base_seq_len] ベースマスク
            target_length: ターゲットシーケンス長
            
        Returns:
            adapted_mask: [target_length] 適応済みマスク
        """
        if len(base_mask) == target_length:
            return base_mask.clone()
        elif len(base_mask) > target_length:
            # 右端を取る（最新部分を重視）
            return base_mask[-target_length:].clone()
        else:
            # パディング（左端をFalseで埋める）
            adapted_mask = torch.zeros(target_length, dtype=torch.bool, device=base_mask.device)
            adapted_mask[-len(base_mask):] = base_mask
            return adapted_mask
            
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
        
        # 短いシーケンスの無限ループ防止
        if seq_len < self.mask_span_min:
            return mask  # 全てFalseのマスクを返す（マスクしない）
        
        # ランダム連続ブロックでマスキング
        while masked_count < target_masked:
            # ランダムなマスクスパン長（torch乱数使用）
            span_len = torch.randint(
                self.mask_span_min, 
                self.mask_span_max + 1, 
                (1,), 
                generator=self.generator
            ).item()
            
            # ランダムな開始位置
            max_start = max(0, seq_len - span_len)
            if max_start <= 0:
                break
                
            start_pos = torch.randint(
                0, 
                max_start + 1, 
                (1,), 
                generator=self.generator
            ).item()
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
        
    def apply_mask_to_features(self, features, masks):
        """
        特徴量にマスクを適用（Learnable Mask Token版、Dict対応）
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量 または Dict[str, torch.Tensor]
            masks: [n_tf, seq_len] マスク（1=マスク, 0=観測）または Dict[str, torch.Tensor]
            
        Returns:
            masked_features: [n_tf, seq_len, n_features] マスク済み特徴量 または Dict[str, torch.Tensor]
        """
        # Dict format support for Model v2
        if isinstance(features, dict):
            return self.apply_mask_to_features_dict(features, masks)
        
        # Legacy tensor format support
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
        
    def apply_mask_to_features_dict(self, features: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Dict形式の特徴量にマスクを適用（Learnable Mask Token版）
        
        Args:
            features: Dict[tf_name, torch.Tensor] - 各TFの特徴量 [batch, seq_len, n_features]
            masks: Dict[tf_name, torch.Tensor] - 各TFのマスク [batch, seq_len]
            
        Returns:
            masked_features: Dict[tf_name, torch.Tensor] - 各TFのマスク済み特徴量 [batch, seq_len, n_features]
        """
        masked_features = {}
        
        for tf_name, tf_features in features.items():
            if tf_name not in masks:
                # マスクがない場合は元の特徴量をそのまま使用
                masked_features[tf_name] = tf_features.clone()
                continue
                
            tf_masks = masks[tf_name]
            batch_size, seq_len, n_features = tf_features.shape
            
            # 特徴量をコピー
            masked_tf_features = tf_features.clone()
            
            # マスク位置を特定 [batch, seq_len, 1] -> bool
            mask_expanded = tf_masks.unsqueeze(-1).bool()  # [batch, seq_len, 1]
            
            # 🔥 マスク位置にLearnable Mask Tokenを設定
            # mask_tokenを各マスク位置に適用
            mask_token_expanded = self.mask_token.unsqueeze(0).unsqueeze(0)  # [1, 1, n_features]
            mask_token_broadcasted = mask_token_expanded.expand(batch_size, seq_len, n_features)
            
            # マスク位置のみを置換
            masked_tf_features[mask_expanded.expand_as(tf_features)] = mask_token_broadcasted[mask_expanded.expand_as(tf_features)]
            
            masked_features[tf_name] = masked_tf_features
        
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
    
    def _generate_masks_dict_vectorized(self, features: Dict[str, torch.Tensor], mask_ratio: float) -> Dict[str, torch.Tensor]:
        """
        🔥 ベクトル化マスク生成 - Dict形式（10倍高速）
        
        Args:
            features: Dict[tf_name, torch.Tensor] - [batch, seq_len, n_features]
            mask_ratio: マスク率
            
        Returns:
            masks: Dict[tf_name, torch.Tensor] - [batch, seq_len] bool
        """
        masks = {}
        
        if self.sync_across_tf:
            # TF間同期マスキング: 最大シーケンス長でベースマスクを生成
            max_seq_len = max(x.shape[1] for x in features.values())
            first_tf = next(iter(features.values()))
            batch_size = first_tf.shape[0]
            device = first_tf.device
            
            # 🔥 一括でベースマスクを生成
            base_masks = self._generate_batch_masks_vectorized(batch_size, max_seq_len, mask_ratio, device)
            
            # 各TFに適用
            for tf_name, tf_features in features.items():
                _, seq_len, _ = tf_features.shape
                
                # ベースマスクを適応
                if seq_len == max_seq_len:
                    tf_masks = base_masks
                else:
                    # 右端を取る（最新部分を重視）
                    tf_masks = base_masks[:, -seq_len:]
                    
                # 🔥 有効位置のみマスク適用（ベクトル化）
                tf_masks = self._apply_valid_mask_vectorized(tf_masks, tf_features)
                masks[tf_name] = tf_masks
        else:
            # TF個別マスキング
            for tf_name, tf_features in features.items():
                batch_size, seq_len, _ = tf_features.shape
                device = tf_features.device
                
                # 🔥 一括でマスクを生成
                tf_masks = self._generate_batch_masks_vectorized(batch_size, seq_len, mask_ratio, device)
                
                # 🔥 有効位置のみマスク適用（ベクトル化）
                tf_masks = self._apply_valid_mask_vectorized(tf_masks, tf_features)
                masks[tf_name] = tf_masks
        
        return masks
    
    def _apply_valid_mask_vectorized(self, masks: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """🔥 有効位置のみマスク適用 - ベクトル化"""
        # 🔥 有効位置を一括検出
        if features.dtype.is_floating_point:
            valid_mask = ~torch.isnan(features[:, :, 0])  # [batch, seq_len]
        else:
            valid_mask = features[:, :, 0] != -1
        
        # 🔥 有効位置のみマスク適用
        masks = masks & valid_mask
        
        return masks
    
    def _generate_batch_masks_vectorized(self, batch_size: int, seq_len: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        """
        🔥 バッチ全体のマスクを一括生成 - 完全ベクトル化（新機能統合）
        
        Args:
            batch_size: バッチサイズ
            seq_len: シーケンス長
            mask_ratio: マスク率
            device: デバイス
            
        Returns:
            masks: [batch, seq_len] bool tensor
        """
        # 🔥 デバイス対応generator初期化
        if self.generator is None or self.generator.device != device:
            self.generator = torch.Generator(device=device)
            
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
        
        # 🔥 完全ベクトル化: for-sループも除去
        # 1) 最大スパン長を取得
        span_lengths_actual = end_positions - start_positions  # [batch, spans]
        max_span_len = span_lengths_actual.max().item()
        
        if max_span_len > 0:
            # 2) インデックス範囲を作成
            idx = torch.arange(max_span_len, device=device)  # [L]
            
            # 3) スパンマスクを作成 [batch, spans, L]
            span_mask = (idx[None, None, :] < span_lengths_actual[:, :, None])
            
            # 4) 範囲インデックスを作成 [batch, spans, L]
            range_idx = idx[None, None, :] + start_positions[:, :, None]
            
            # 5) 有効範囲のマスクを作成
            valid_mask = span_mask & (range_idx < seq_len)
            
            # 6) scatter_でマスクを一括適用
            batch_idx = torch.arange(batch_size, device=device)[:, None, None].expand(-1, estimated_spans, max_span_len)
            masks[batch_idx[valid_mask], range_idx[valid_mask]] = True
        
        # 🔥 マスク数を正確に調整（バッチ並列）
        for b in range(batch_size):
            current_masked = masks[b].sum().item()
            
            if current_masked > target_masked:
                # 超過分をランダムに解除
                masked_indices = torch.where(masks[b])[0]
                excess = current_masked - target_masked
                if excess > 0:
                    # 🔥 CPUでrandpermを実行してからGPUに移動
                    cpu_generator = torch.Generator()
                    if hasattr(self.generator, 'get_state'):
                        cpu_generator.set_state(self.generator.get_state().cpu())
                    perm_indices = torch.randperm(len(masked_indices), generator=cpu_generator)[:excess]
                    remove_indices = masked_indices[perm_indices]
                    masks[b, remove_indices] = False
            elif current_masked < target_masked:
                # 不足分をランダムに追加
                unmasked_indices = torch.where(~masks[b])[0]
                needed = target_masked - current_masked
                if needed > 0 and len(unmasked_indices) > 0:
                    # 🔥 CPUでrandpermを実行してからGPUに移動
                    cpu_generator = torch.Generator()
                    if hasattr(self.generator, 'get_state'):
                        cpu_generator.set_state(self.generator.get_state().cpu())
                    perm_indices = torch.randperm(len(unmasked_indices), generator=cpu_generator)[:needed]
                    add_indices = unmasked_indices[perm_indices]
                    masks[b, add_indices] = True
        
        return masks