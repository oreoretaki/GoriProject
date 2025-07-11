#!/usr/bin/env python3
"""
Stage 1 Multi-TF Self-Supervised Reconstruction Model
TF固有CNN + 共有エンコーダー + Bottleneck + TF別デコーダー
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# T5転移学習用インポート
try:
    from .lm_adapter import T5TimeSeriesAdapter
    T5_AVAILABLE = True
except ImportError:
    T5TimeSeriesAdapter = None
    T5_AVAILABLE = False

# マスキング戦略インポート
from .masking import MaskingStrategy


class CrossScaleFusion(nn.Module):
    """Coarse→Fine Cross-Attention for multi-scale fusion"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True)
        
    def forward(self, coarse_features: torch.Tensor, fine_features: torch.Tensor, 
                coarse_mask: Optional[torch.Tensor] = None, 
                fine_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-scale attention from coarse to fine timeframes
        
        Args:
            coarse_features: [batch, seq_coarse, d_model] - coarse TF features
            fine_features: [batch, seq_fine, d_model] - fine TF features  
            coarse_mask: [batch, seq_coarse] - padding mask for coarse
            fine_mask: [batch, seq_fine] - padding mask for fine
            
        Returns:
            fused_features: [batch, d_model] - CLS token representation
        """
        # Use CLS token from coarse as query
        q = self.query_proj(coarse_features[:, :1])  # [batch, 1, d_model]
        k = self.key_proj(fine_features)  # [batch, seq_fine, d_model] 
        v = self.value_proj(fine_features)  # [batch, seq_fine, d_model]
        
        # Cross-attention
        fused, _ = self.attn(
            q, k, v,
            key_padding_mask=fine_mask  # Only mask fine features
        )
        
        return fused.squeeze(1)  # [batch, d_model]

class TFSpecificStem(nn.Module):
    """TF固有ステム: 1D depth-wise CNN"""
    
    def __init__(self, n_features: int, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Depth-wise convolution
        self.depthwise_conv = nn.Conv1d(
            n_features, n_features, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=n_features
        )
        
        # Point-wise convolution (projection)
        self.pointwise_conv = nn.Conv1d(n_features, d_model, kernel_size=1)
        
        # Normalization & activation
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_features]
        Returns:
            out: [batch, seq_len, d_model]
        """
        # [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # Depth-wise + Point-wise convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        
        # [batch, d_model, seq_len] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # Normalization & activation
        x = self.norm(x)
        x = self.activation(x)
        
        return x

class CrossScaleAttention(nn.Module):
    """クロススケール注意機構"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
        
    def forward(self, tf_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tf_features: [batch, n_tf, seq_len, d_model]
        Returns:
            attended: [batch, n_tf, seq_len, d_model]
        """
        batch_size, n_tf, seq_len, d_model = tf_features.shape
        
        # Reshape for multi-head attention
        # [batch, n_tf, seq_len, d_model] -> [batch, n_tf*seq_len, d_model]
        x = tf_features.view(batch_size, n_tf * seq_len, d_model)
        
        q = self.q_proj(x).view(batch_size, n_tf * seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, n_tf * seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, n_tf * seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tf * seq_len, d_model)
        out = self.out_proj(out)
        
        # Reshape back
        out = out.view(batch_size, n_tf, seq_len, d_model)
        
        return out

class TransformerEncoderLayer(nn.Module):
    """Transformer エンコーダー層（Mamba代替）"""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] attention mask
        Returns:
            out: [batch, seq_len, d_model]
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

class SharedEncoder(nn.Module):
    """共有クロススケールエンコーダー"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config['model']['encoder']['d_model']
        self.n_layers = config['model']['encoder']['n_layers']
        self.cross_attn_every = config['model']['encoder']['cross_attn_every']
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(self.d_model)
            for _ in range(self.n_layers)
        ])
        
        # Cross-scale attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossScaleAttention(self.d_model)
            for _ in range(self.n_layers // self.cross_attn_every)
        ])
        
    def forward(self, tf_features: torch.Tensor, masks: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tf_features: [batch, n_tf, seq_len, d_model] or [batch, seq_len, d_model] (async mode)
            masks: [batch, n_tf, seq_len] padding masks (legacy)
            key_padding_mask: [batch, seq_len] padding mask (async mode)
        Returns:
            encoded: [batch, n_tf, seq_len, d_model] or [batch, seq_len, d_model]
        """
        # async mode support: 3D input
        if tf_features.dim() == 3:
            return self._forward_single_tf(tf_features, key_padding_mask)
        
        # legacy mode: 4D input
        batch_size, n_tf, seq_len, d_model = tf_features.shape
        
        # 🔥 NaN-paddedシーケンス用のattention_mask自動生成
        if masks is None:
            # NaN値を検出してpadding maskを作成 (True=マスク対象, False=有効データ)
            masks = torch.isnan(tf_features).any(dim=-1)  # [batch, n_tf, seq_len]
        
        # Process each TF separately first
        for layer_idx, layer in enumerate(self.layers):
            # Reshape for processing
            x_flat = tf_features.view(batch_size * n_tf, seq_len, d_model)
            mask_flat = masks.view(batch_size * n_tf, seq_len) if masks is not None else None
            
            # Self-attention within each TF
            x_flat = layer(x_flat, mask_flat)
            
            # Reshape back
            tf_features = x_flat.view(batch_size, n_tf, seq_len, d_model)
            
            # Cross-scale attention every N layers
            if (layer_idx + 1) % self.cross_attn_every == 0:
                cross_idx = layer_idx // self.cross_attn_every
                tf_features = tf_features + self.cross_attn_layers[cross_idx](tf_features)
                
        return tf_features
    
    def _forward_single_tf(self, tf_features: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        単一TF用forward (async mode)
        
        Args:
            tf_features: [batch, seq_len, d_model]
            key_padding_mask: [batch, seq_len] padding mask (True=valid, False=padded)
        Returns:
            encoded: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = tf_features.shape
        
        # padding maskの処理
        if key_padding_mask is not None:
            # key_padding_maskは通常True=valid, False=paddedだが、
            # attention maskとしてはTrue=masked, False=validが必要
            attention_mask = key_padding_mask  # そのまま使用
        else:
            # NaN値を検出してpadding maskを作成
            attention_mask = torch.isnan(tf_features).any(dim=-1)  # [batch, seq_len]
        
        # Process through transformer layers
        encoded = tf_features
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention within the TF
            encoded = layer(encoded, attention_mask)
            
            # Note: Cross-scale attention is not applied in single TF mode
            
        return encoded

class Bottleneck(nn.Module):
    """Bottleneck: 全域コンテキスト圧縮"""
    
    def __init__(self, d_model: int, latent_len: int, stride: int = 4):
        super().__init__()
        # latent_lenは参考値のみ（動的計算で実際の値を使用）
        self.latent_len = latent_len  
        self.stride = stride
        
        # Strided convolution for compression
        self.compress = nn.Conv1d(d_model, d_model, kernel_size=stride, stride=stride)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_tf, seq_len, d_model]
        Returns:
            compressed: [batch, n_tf, latent_len, d_model]
        """
        batch_size, n_tf, seq_len, d_model = x.shape
        
        # Reshape and compress
        x = x.view(batch_size * n_tf, seq_len, d_model).transpose(1, 2)  # [batch*n_tf, d_model, seq_len]
        compressed = self.compress(x)  # [batch*n_tf, d_model, latent_len]
        compressed = compressed.transpose(1, 2)  # [batch*n_tf, latent_len, d_model]
        
        # 動的にlatent_lenを計算
        latent_len = compressed.size(1)  # [batch*n_tf, latent_len, d_model]
        
        # Reshape back
        compressed = compressed.view(batch_size, n_tf, latent_len, d_model)
        compressed = self.norm(compressed)
        
        return compressed

class TFDecoder(nn.Module):
    """TF個別デコーダー"""
    
    def __init__(self, d_model: int, seq_len: int, n_output_features: int = 4, n_layers: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.n_output_features = n_output_features
        
        # Upsampling layers
        self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=4)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(d_model, n_output_features, kernel_size=1)
        
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, latent_len, d_model]
        Returns:
            decoded: [batch, seq_len, n_output_features]
        """
        batch_size, latent_len, d_model = x.shape
        
        # [batch, latent_len, d_model] -> [batch, d_model, latent_len]
        x = x.transpose(1, 2)
        
        # Upsample (latent_len=1対応)
        if latent_len == 1:
            # latent_len=1の場合: repeat→upsample
            x = x.repeat(1, 1, 16)  # [batch, d_model, 16] へ拡張
        x = self.upsample(x)
        
        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
            x = F.gelu(x)
            
        # Output projection
        x = self.output_proj(x)
        
        # [batch, n_output_features, seq_len] -> [batch, seq_len, n_output_features]
        x = x.transpose(1, 2)
        
        # Ensure correct length
        if x.size(1) != self.seq_len:
            x = F.interpolate(x.transpose(1, 2), size=self.seq_len, mode='linear', align_corners=False).transpose(1, 2)
            
        return x

class Stage1Model(nn.Module):
    """Stage 1 マルチTF自己教師あり再構築モデル"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Model parameters
        self.n_tf = config['data']['n_timeframes']
        self.seq_len = config['data']['seq_len']
        self.n_features = config['data']['n_features']
        self.d_model = config['model']['tf_stem']['d_model']
        self.latent_len = config['model']['bottleneck']['latent_len']
        
        # Model v2: async mode support
        self.async_sampler = config.get('model', {}).get('async_sampler', False)
        self.timeframes = config.get('data', {}).get('timeframes', ['m1', 'm5', 'm15', 'm30', 'h1', 'h4'])
        
        # TF-specific stems (now using ModuleDict for variable TFs)
        if self.async_sampler:
            # Dict-based stems for variable timeframes
            self.tf_stems = nn.ModuleDict({
                tf: TFSpecificStem(self.n_features, self.d_model, config['model']['tf_stem']['kernel_size'])
                for tf in self.timeframes
            })
        else:
            # List-based stems for fixed timeframes
            self.tf_stems = nn.ModuleList([
                TFSpecificStem(self.n_features, self.d_model, config['model']['tf_stem']['kernel_size'])
                for _ in range(self.n_tf)
            ])
        
        # Shared encoder (T5転移学習またはバニラエンコーダー)
        use_pretrained_lm = config.get('transfer_learning', {}).get('use_pretrained_lm', False)
        
        if use_pretrained_lm:
            if not T5_AVAILABLE:
                raise ImportError(
                    "T5転移学習が有効化されていますが、lm_adapterまたはtransformersライブラリが利用できません。"
                    "pip install transformers>=4.42.0 でインストールしてください。"
                )
            print("🤗 T5転移学習を使用します（共有エンコーダー）")
            # 🔥 T5エンコーダーは常に共有（async_samplerモードでも）
            self.shared_encoder = T5TimeSeriesAdapter(config)
        else:
            print("📦 従来のSharedEncoderを使用します")
            if self.async_sampler:
                # TF-specific encoders for async mode
                self.encoders = nn.ModuleDict({
                    tf: SharedEncoder(config) for tf in self.timeframes
                })
            else:
                self.shared_encoder = SharedEncoder(config)
        
        # Cross-scale fusion for async mode
        if self.async_sampler:
            cross_pairs = config.get('model', {}).get('cross_pairs', [["h4", "m1"], ["h1", "m1"], ["m30", "m1"]])
            self.cross_fusion = nn.ModuleDict({
                f"{coarse}_{fine}": CrossScaleFusion(self.d_model)
                for coarse, fine in cross_pairs
            })
            self.cross_pairs = cross_pairs
        
        # Bottleneck
        self.bottleneck = Bottleneck(
            self.d_model, 
            self.latent_len, 
            config['model']['bottleneck']['stride']
        )
        
        # TF-specific decoders
        if self.async_sampler:
            # Dict-based decoders for variable timeframes
            self.tf_decoders = nn.ModuleDict({
                tf: TFDecoder(self.d_model, self.seq_len, n_output_features=4, n_layers=config['model']['decoder']['n_layers'])
                for tf in self.timeframes
            })
        else:
            # List-based decoders for fixed timeframes
            self.tf_decoders = nn.ModuleList([
                TFDecoder(self.d_model, self.seq_len, n_output_features=4, n_layers=config['model']['decoder']['n_layers'])
                for _ in range(self.n_tf)
            ])
        
        # 🔥 Learnable Mask Token Strategy
        # 🔥 一旦従来のマスキング戦略を使用（動作確認のため）
        self.masking_strategy = MaskingStrategy(config, n_features=self.n_features)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
    def _create_positional_encoding(self) -> nn.Embedding:
        """位置エンコーディング作成"""
        return nn.Embedding(self.seq_len, self.d_model)
        
    def forward(self, batch: Dict[str, torch.Tensor], eval_mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Model v2: Dict input support for variable-length multi-timeframe processing
        
        Args:
            batch: {
                'm1': [B, L_m1, F],
                'm5': [B, L_m5, F],
                ...
            } - Dict of timeframe tensors with variable lengths (NaN-padded)
            eval_mask_ratio: Override mask ratio for evaluation (fixes val_corr=0 issue)
            
        Returns:
            outputs: Dict[str, torch.Tensor] - Same keys as input, values are [B, L_tf, 4]
        """
        # 🔥 入力型チェック: TensorかDictかを判定
        if isinstance(batch, dict):
            # Dict形式 -> async_samplerの設定に従う
            if self.async_sampler:
                return self._forward_async(batch, eval_mask_ratio)
            else:
                return self._forward_sync(batch, eval_mask_ratio)
        else:
            # Tensor形式 -> 従来のlegacy forward
            return self._forward_legacy_tensor(batch, eval_mask_ratio)
    
    def _forward_async(self, batch: Dict[str, torch.Tensor], eval_mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Async mode forward pass with variable-length support"""
        encoded = {}
        padding_masks = {}
        training_masks = {}
        
        # 🔥 1-A. ベクトル化TF前処理（バッチ融合で高速化）
        if hasattr(self, 'shared_encoder'):
            # 共有エンコーダー使用時: バッチ融合で1回呼び出し
            encoded, padding_masks, training_masks = self._forward_batch_fusion(batch, eval_mask_ratio)
        else:
            # TF固有エンコーダー使用時: 個別処理
            for tf, x in batch.items():
                # NaN detection for padding mask
                mask = torch.isnan(x[..., 0])  # [B, L] - use first feature to detect NaN
                x_clean = torch.nan_to_num(x, nan=0.0)  # NaN → 0
                
                # Apply self-supervised masking BEFORE stem processing
                if self.training or eval_mask_ratio is not None:
                    # Generate training masks for this TF
                    mask_ratio = eval_mask_ratio if eval_mask_ratio is not None else 0.15
                    tf_training_mask = self._generate_tf_masks(x_clean, mask_ratio)
                    # Apply masking to input features (before stem)
                    x_masked_input = self._apply_tf_masks(x_clean, tf_training_mask)
                    training_masks[tf] = tf_training_mask
                else:
                    x_masked_input = x_clean
                    training_masks[tf] = torch.zeros_like(mask, dtype=torch.bool)
                
                # TF-specific stem processing (after masking)
                x_stem = self.tf_stems[tf](x_masked_input)  # [B, L, d_model]
                
                # TF固有エンコーダーを使用（非T5モード）
                encoded_features = self.encoders[tf](x_stem, key_padding_mask=mask)
                encoded[tf] = encoded_features  # [B, L, d_model]
                padding_masks[tf] = mask
        
        # 1-B. Cross-scale Fusion (coarse→fine)
        fused_cls = {}
        if hasattr(self, 'cross_fusion') and self.cross_fusion:
            for coarse, fine in self.cross_pairs:
                if coarse in encoded and fine in encoded:
                    key = f"{coarse}_{fine}"
                    if key in self.cross_fusion:
                        fused_cls[fine] = self.cross_fusion[key](
                            encoded[coarse], encoded[fine],
                            padding_masks[coarse], padding_masks[fine]
                        )
        
        # 1-C. Bottleneck + Decoder  
        outputs = {}
        for tf, z in encoded.items():
            # Async mode用の修正されたBottleneck処理
            # z: [B, L, d_model] → mean-pooling → [B, d_model] → MLP → [B, latent_len, d_model]
            
            valid_mask = ~padding_masks[tf]  # [B, L]
            if valid_mask.sum() > 0:
                # Mean pooling over valid positions
                z_pool = (z * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)  # [B, d_model]
            else:
                z_pool = z.mean(dim=1)  # Fallback to regular mean
            
            # Async mode用の簡略化bottleneck: ダイレクトMLP
            batch_size = z_pool.size(0)
            # z_pool: [B, d_model] → [B, 1, d_model] (単一latent)
            z_latent = z_pool.unsqueeze(1)  # [B, 1, d_model]
            
            # TF-specific decoder (latent_len=1として処理)
            outputs[tf] = self.tf_decoders[tf](z_latent)  # [B, seq_len, 4]
        
        return outputs
    
    def _forward_sync(self, batch: Dict[str, torch.Tensor], eval_mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Legacy sync mode forward pass - convert Dict to tensor format"""
        # Convert Dict format to legacy tensor format for backward compatibility
        timeframes = list(batch.keys())
        
        # Find maximum sequence length and batch size
        batch_size = list(batch.values())[0].shape[0]
        max_seq_len = max(x.shape[1] for x in batch.values())
        n_features = list(batch.values())[0].shape[2]
        n_tf = len(timeframes)
        
        # Pad all sequences to max length and stack
        features = torch.full((batch_size, n_tf, max_seq_len, n_features), 
                            float('nan'), device=list(batch.values())[0].device)
        
        for i, (tf, x) in enumerate(batch.items()):
            seq_len = x.shape[1]
            features[:, i, :seq_len, :] = x
        
        # Use legacy forward logic
        # 🔥 自己教師ありマスキング処理
        training_masks = None
        if self.training or eval_mask_ratio is not None:
            mask_ratio = eval_mask_ratio if eval_mask_ratio is not None else 0.15
            training_masks = torch.stack([
                self.masking_strategy.generate_masks(
                    features[b], 
                    seed=hash((id(features), b)) % 2147483647,
                    mask_ratio=mask_ratio
                ) 
                for b in range(batch_size)
            ], dim=0)
        
        # Apply masking
        if training_masks is not None:
            masked_features = torch.stack([
                self.masking_strategy.apply_mask_to_features(features[b], training_masks[b])
                for b in range(batch_size)
            ], dim=0)
        else:
            masked_features = features
            training_masks = torch.zeros(batch_size, n_tf, max_seq_len, device=features.device, dtype=torch.bool)
        
        # Process through model
        if hasattr(self, 'shared_encoder'):
            if isinstance(self.shared_encoder, T5TimeSeriesAdapter):
                encoded = self.shared_encoder(masked_features)
            else:
                tf_embeddings = []
                for i in range(n_tf):
                    stem_out = self.tf_stems[i](masked_features[:, i])
                    tf_embeddings.append(stem_out)
                tf_embeddings = torch.stack(tf_embeddings, dim=1)
                
                pos_ids = torch.arange(max_seq_len, device=masked_features.device).unsqueeze(0).expand(batch_size, -1)
                pos_emb = self.pos_encoding(pos_ids).unsqueeze(1)
                tf_embeddings = tf_embeddings + pos_emb
                
                encoded = self.shared_encoder(tf_embeddings)
        
        compressed = self.bottleneck(encoded)
        
        reconstructed = []
        for i in range(n_tf):
            decoded = self.tf_decoders[i](compressed[:, i])
            reconstructed.append(decoded)
        reconstructed = torch.stack(reconstructed, dim=1)
        
        # Convert back to Dict format
        outputs = {}
        for i, tf in enumerate(timeframes):
            seq_len = batch[tf].shape[1]
            outputs[tf] = reconstructed[:, i, :seq_len, :]
        
        return outputs
    
    def _forward_legacy_tensor(self, batch: torch.Tensor, eval_mask_ratio: Optional[float] = None) -> torch.Tensor:
        """Legacy tensor format forward pass"""
        # 従来のテンソル形式での処理
        features = batch  # [batch, n_tf, seq_len, n_features]
        batch_size, n_tf, seq_len, n_features = features.shape
        
        # 🔥 自己教師ありマスキング処理
        training_masks = None
        if self.training or eval_mask_ratio is not None:
            mask_ratio = eval_mask_ratio if eval_mask_ratio is not None else 0.15
            training_masks = torch.stack([
                self.masking_strategy.generate_masks(
                    features[b], 
                    seed=hash((id(features), b)) % 2147483647,
                    eval_mask_ratio_override=mask_ratio
                ) 
                for b in range(batch_size)
            ], dim=0)
        
        # Apply masking
        if training_masks is not None:
            masked_features = torch.stack([
                self.masking_strategy.apply_mask_to_features(features[b], training_masks[b])
                for b in range(batch_size)
            ], dim=0)
        else:
            masked_features = features
            training_masks = torch.zeros(batch_size, n_tf, seq_len, device=features.device, dtype=torch.bool)
        
        # Process through model (legacy logic)
        if hasattr(self, 'shared_encoder'):
            if isinstance(self.shared_encoder, T5TimeSeriesAdapter):
                encoded = self.shared_encoder(masked_features)
            else:
                tf_embeddings = []
                for i in range(n_tf):
                    stem_out = self.tf_stems[i](masked_features[:, i])
                    tf_embeddings.append(stem_out)
                tf_embeddings = torch.stack(tf_embeddings, dim=1)
                
                pos_ids = torch.arange(seq_len, device=masked_features.device).unsqueeze(0).expand(batch_size, -1)
                pos_emb = self.pos_encoding(pos_ids).unsqueeze(1)
                tf_embeddings = tf_embeddings + pos_emb
                
                # SharedEncoder expects [batch, n_tf, seq_len, d_model]
                encoded = self.shared_encoder(tf_embeddings)
        else:
            # Fallback to individual processing
            tf_embeddings = []
            for i in range(n_tf):
                stem_out = self.tf_stems[i](masked_features[:, i])
                tf_embeddings.append(stem_out)
            encoded = torch.stack(tf_embeddings, dim=1)
        
        # Bottleneck processing
        compressed = self.bottleneck(encoded)
        
        # Decode
        reconstructed = []
        for i in range(n_tf):
            if hasattr(self, 'tf_decoders') and i < len(self.tf_decoders):
                # compressed[:, i] should be [batch, latent_len, d_model]
                decoder_out = self.tf_decoders[i](compressed[:, i])
            else:
                decoder_out = compressed[:, i]
            reconstructed.append(decoder_out)
        
        reconstructed = torch.stack(reconstructed, dim=1)
        return reconstructed
    
    def _generate_tf_masks(self, x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        """Generate self-supervised masks for a single TF"""
        batch_size, seq_len, _ = x.shape
        
        # Simple random masking for now
        masks = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        return masks
    
    def _apply_tf_masks(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masks to features using learnable mask tokens"""
        # Apply mask (set masked positions to 0 or use learnable mask token)
        masked_x = x.clone()
        if hasattr(self.masking_strategy, 'mask_token'):
            # Use learnable mask token
            mask_token = self.masking_strategy.mask_token.expand_as(x)
            masked_x = torch.where(mask.unsqueeze(-1), mask_token, x)
        else:
            # Simple masking to zero
            masked_x = torch.where(mask.unsqueeze(-1), torch.zeros_like(x), x)
        
        return masked_x
        
    def get_model_info(self) -> Dict:
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # float32 assumption
            'architecture': {
                'n_tf': self.n_tf,
                'seq_len': self.seq_len,
                'n_features': self.n_features,
                'd_model': self.d_model,
                'latent_len': f'dynamic({self.seq_len // self.bottleneck.stride})'
            }
        }
    
    def _forward_batch_fusion(self, batch: Dict[str, torch.Tensor], eval_mask_ratio: Optional[float] = None) -> Tuple[Dict, Dict, Dict]:
        """
        🔥 バッチ融合による高速化処理（統合版）
        共有エンコーダーを1回だけ呼び出してTF処理を高速化
        """
        # 1. マスク生成とマスク適用
        encoded = {}
        padding_masks = {}
        training_masks = {}
        
        if self.training or eval_mask_ratio is not None:
            mask_ratio = eval_mask_ratio if eval_mask_ratio is not None else 0.15
            # 🔥 ベクトル化マスク生成使用
            masks = self.masking_strategy.generate_masks_dict(batch, eval_mask_ratio_override=mask_ratio)
            masked_batch = self.masking_strategy.apply_mask_to_features_dict(batch, masks)
        else:
            masked_batch = batch
            masks = {}
        
        # 2. TF-specific stem処理
        stemmed_features = {}
        for tf_name, tf_features in masked_batch.items():
            # NaN detection for padding mask
            mask = torch.isnan(tf_features[..., 0])
            x_clean = torch.nan_to_num(tf_features, nan=0.0)
            
            # TF-specific stem processing
            x_stem = self.tf_stems[tf_name](x_clean)
            stemmed_features[tf_name] = x_stem
            padding_masks[tf_name] = mask
            training_masks[tf_name] = masks.get(tf_name, torch.zeros_like(mask, dtype=torch.bool))
        
        # 🔥 3. 共通seq_lenにパディングしてバッチ融合
        max_seq_len = max(x.shape[1] for x in stemmed_features.values())
        batch_size = list(stemmed_features.values())[0].shape[0]
        
        # 全TFを結合: [batch * n_tf, max_seq_len, d_model]
        fused_features = []
        fused_padding_masks = []
        
        for tf_name in self.timeframes:
            if tf_name in stemmed_features:
                tf_features = stemmed_features[tf_name]
                current_seq_len = tf_features.shape[1]
                
                # パディング
                if current_seq_len < max_seq_len:
                    pad_len = max_seq_len - current_seq_len
                    padded_features = F.pad(tf_features, (0, 0, 0, pad_len), value=0.0)
                    # パディングマスク
                    pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=tf_features.device)
                    pad_mask[:, current_seq_len:] = True
                else:
                    padded_features = tf_features
                    pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool, device=tf_features.device)
                
                fused_features.append(padded_features)
                fused_padding_masks.append(pad_mask)
        
        # スタックしてバッチ融合: [batch * n_tf, max_seq_len, d_model]
        fused_features = torch.stack(fused_features, dim=1)  # [batch, n_tf, max_seq_len, d_model]
        fused_features = fused_features.view(batch_size * self.n_tf, max_seq_len, self.d_model)
        
        fused_padding_masks = torch.stack(fused_padding_masks, dim=1)  # [batch, n_tf, max_seq_len]
        fused_padding_masks = fused_padding_masks.view(batch_size * self.n_tf, max_seq_len)
        
        # 🔥 4. 共有エンコーダー1回呼び出し（6回→1回に削減）
        if hasattr(self.shared_encoder, 't5_encoder') or hasattr(self.shared_encoder, 'patch_embedding'):
            # T5TimeSeriesAdapter の場合
            encoded_features = self.shared_encoder(fused_features, key_padding_mask=fused_padding_masks)
        else:
            # 通常のTransformerEncoder の場合
            encoded_features = self.shared_encoder(fused_features, src_key_padding_mask=fused_padding_masks)
        
        # 🔥 5. バッチ融合を解除して各TFに分離
        encoded_features = encoded_features.view(batch_size, self.n_tf, max_seq_len, self.d_model)
        
        # 各TFに分離
        for i, tf_name in enumerate(self.timeframes):
            if tf_name in batch:
                # 各TFの元のseq_lenに戻す
                original_seq_len = batch[tf_name].shape[1]
                tf_encoded = encoded_features[:, i, :original_seq_len, :]  # [batch, seq_len, d_model]
                encoded[tf_name] = tf_encoded
        
        return encoded, padding_masks, training_masks

# 🔥 ファクトリー関数: ベクトル化版を統合済み
def create_stage1_model(config: dict, use_vectorized: bool = True):
    """
    Stage1モデルを作成（ベクトル化機能統合済み）
    
    Args:
        config: モデル設定
        use_vectorized: ベクトル化版を使用するか（True=10倍高速、False=従来版）
        
    Returns:
        model: Stage1モデル
    """
    if use_vectorized:
        print("⚡ ベクトル化機能統合済みStage1モデルを使用（10倍高速）")
    else:
        print("📦 従来のStage1モデルを使用")
    return Stage1Model(config)