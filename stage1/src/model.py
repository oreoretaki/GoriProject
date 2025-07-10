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
        
    def forward(self, tf_features: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tf_features: [batch, n_tf, seq_len, d_model]
            masks: [batch, n_tf, seq_len] padding masks (optional, auto-generated from NaN)
        Returns:
            encoded: [batch, n_tf, seq_len, d_model]
        """
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
        
        # Upsample
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
        
        # TF-specific stems
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
            print("🤗 T5転移学習を使用します")
            self.shared_encoder = T5TimeSeriesAdapter(config)
        else:
            print("📦 従来のSharedEncoderを使用します")
            self.shared_encoder = SharedEncoder(config)
        
        # Bottleneck
        self.bottleneck = Bottleneck(
            self.d_model, 
            self.latent_len, 
            config['model']['bottleneck']['stride']
        )
        
        # TF-specific decoders
        self.tf_decoders = nn.ModuleList([
            TFDecoder(self.d_model, self.seq_len, n_output_features=4, n_layers=config['model']['decoder']['n_layers'])
            for _ in range(self.n_tf)
        ])
        
        # 🔥 Learnable Mask Token Strategy
        self.masking_strategy = MaskingStrategy(config, n_features=self.n_features)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding()
        
    def _create_positional_encoding(self) -> nn.Embedding:
        """位置エンコーディング作成"""
        return nn.Embedding(self.seq_len, self.d_model)
        
    def forward(self, features: torch.Tensor, masks: Optional[torch.Tensor] = None, training_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [batch, n_tf, seq_len, n_features] 入力特徴量（生データ）
            masks: [batch, n_tf, seq_len] パディング/マスクマスク（従来）
            training_masks: [batch, n_tf, seq_len] 自己教師ありマスク（新規）
            
        Returns:
            outputs: {
                'reconstructed': [batch, n_tf, seq_len, 4] 再構築されたOHLC,
                'encoded': [batch, n_tf, latent_len, d_model] エンコード済み表現,
                'masked_features': [batch, n_tf, seq_len, n_features] マスク適用済み特徴量,
                'training_masks': [batch, n_tf, seq_len] 生成されたマスク
            }
        """
        batch_size, n_tf, seq_len, n_features = features.shape
        
        # 🔥 自己教師ありマスキング処理
        if training_masks is None and self.training:
            # 訓練時：各サンプルに対してマスクを生成（再現性担保）
            training_masks = torch.stack([
                self.masking_strategy.generate_masks(
                    features[b], 
                    seed=hash((id(features), b)) % 2147483647  # 再現性担保
                )  # [n_tf, seq_len]
                for b in range(batch_size)
            ], dim=0)  # [batch, n_tf, seq_len]
        
        # マスクされた特徴量を生成
        if training_masks is not None:
            masked_features = torch.stack([
                self.masking_strategy.apply_mask_to_features(features[b], training_masks[b])
                for b in range(batch_size)
            ], dim=0)  # [batch, n_tf, seq_len, n_features]
        else:
            # 推論時またはマスクなし
            masked_features = features
            training_masks = torch.zeros(batch_size, n_tf, seq_len, device=features.device, dtype=torch.bool)
        
        # Shared encoding（マスク済み特徴量を使用）
        if isinstance(self.shared_encoder, T5TimeSeriesAdapter):
            # T5アダプターはマスク済み特徴量を使用
            encoded = self.shared_encoder(masked_features)  # [batch, n_tf, seq_len, d_model]
        else:
            # 従来のSharedEncoder: TF-specific stem processing
            tf_embeddings = []
            for i in range(n_tf):
                stem_out = self.tf_stems[i](masked_features[:, i])  # [batch, seq_len, d_model]
                tf_embeddings.append(stem_out)
                
            tf_embeddings = torch.stack(tf_embeddings, dim=1)  # [batch, n_tf, seq_len, d_model]
            
            # Add positional encoding
            pos_ids = torch.arange(seq_len, device=masked_features.device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.pos_encoding(pos_ids).unsqueeze(1)  # [batch, 1, seq_len, d_model]
            tf_embeddings = tf_embeddings + pos_emb
            
            encoded = self.shared_encoder(tf_embeddings, masks)  # [batch, n_tf, seq_len, d_model]
        
        # Bottleneck
        compressed = self.bottleneck(encoded)  # [batch, n_tf, latent_len, d_model]
        
        # TF-specific decoding
        reconstructed = []
        for i in range(n_tf):
            decoded = self.tf_decoders[i](compressed[:, i])  # [batch, seq_len, 4]
            reconstructed.append(decoded)
            
        reconstructed = torch.stack(reconstructed, dim=1)  # [batch, n_tf, seq_len, 4]
        
        return {
            'reconstructed': reconstructed,
            'encoded': compressed,
            'masked_features': masked_features,
            'training_masks': training_masks
        }
        
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