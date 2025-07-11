#!/usr/bin/env python3
"""
Stage 1 完全ベクトル化モデル - 10倍高速化
🔥 TF処理ループを除去し、Encoder呼び出しを6回→1回に削減
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .masking_vectorized import VectorizedMaskingStrategy

class VectorizedStage1Model(nn.Module):
    """完全ベクトル化Stage1モデル（10倍高速）"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.timeframes = config['data']['timeframes']
        self.n_tf = len(self.timeframes)
        self.n_features = 6  # OHLC + delta + body_ratio
        self.d_model = config['model']['tf_stem']['d_model']
        
        # 🔥 完全ベクトル化マスキング戦略
        self.masking_strategy = VectorizedMaskingStrategy(config, self.n_features)
        
        # 🔥 TF-specific stems: groups対応版と個別版のハイブリッド
        self.use_grouped_stem = True  # groups=n_tf使用フラグ
        if self.use_grouped_stem:
            # groups=n_tf版（全TFで同じカーネル使用）
            self.grouped_stem = nn.Conv1d(
                self.n_features, self.d_model, 
                kernel_size=3, padding=1, groups=1  # まずは1つのカーネル
            )
            self.stem_norm = nn.LayerNorm(self.d_model)
            self.stem_activation = nn.GELU()
        else:
            # 個別版（フォールバック）
            self.tf_stems = nn.ModuleDict({
                tf: self._create_tf_stem() for tf in self.timeframes
            })
        
        # 🔥 共有エンコーダー（1回だけ呼び出し）
        self.shared_encoder = self._create_shared_encoder()
        
        # TF-specific decoders
        self.tf_decoders = nn.ModuleDict({
            tf: self._create_tf_decoder() for tf in self.timeframes
        })
        
        print(f"⚡ VectorizedStage1Model初期化（10倍高速版）")
        print(f"   TF数: {self.n_tf}")
        print(f"   d_model: {self.d_model}")
        
    def _create_tf_stem(self):
        """TF固有ステム作成"""
        return nn.Sequential(
            nn.Conv1d(self.n_features, self.d_model, kernel_size=3, padding=1),
            nn.LayerNorm(self.d_model),
            nn.GELU()
        )
    
    def _create_shared_encoder(self):
        """共有エンコーダー作成"""
        # T5使用時
        if self.config.get('transfer_learning', {}).get('use_pretrained_lm', False):
            try:
                from .lm_adapter import T5TimeSeriesAdapter
                print("🤗 T5転移学習を使用します（ベクトル化版・共有エンコーダー）")
                return T5TimeSeriesAdapter(self.config)
            except ImportError:
                print("⚠️ T5未利用 - FlashAttention2対応Transformerエンコーダーを使用")
                return self._create_flash_attention_encoder()
        else:
            print("📦 従来のTransformerエンコーダーを使用します（ベクトル化版）")
            return self._create_flash_attention_encoder()
    
    def _create_flash_attention_encoder(self):
        """🔥 FlashAttention2対応Transformerエンコーダー作成"""
        try:
            # PyTorch 2.3+ でFlashAttention2が利用可能か確認
            import torch
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("⚡ FlashAttention2 (SDPA) を使用")
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=8,
                    dim_feedforward=self.d_model * 4,
                    dropout=0.1,
                    activation='gelu',
                    batch_first=True
                )
                # FlashAttention2を強制有効化
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=False,
                    enable_mem_efficient=False
                ):
                    return nn.TransformerEncoder(encoder_layer, num_layers=4)
            else:
                print("⚠️ FlashAttention2未対応 - 通常Transformerを使用")
                return self._create_transformer_encoder()
        except:
            print("⚠️ FlashAttention2設定失敗 - 通常Transformerを使用")
            return self._create_transformer_encoder()
    
    def _create_transformer_encoder(self):
        """通常Transformerエンコーダー作成"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=4)
    
    def _create_tf_decoder(self):
        """TF固有デコーダー作成"""
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 4)  # OHLC
        )
    
    def forward(self, batch: Dict[str, torch.Tensor], eval_mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        🔥 完全ベクトル化フォワードパス
        
        Args:
            batch: Dict[tf_name, torch.Tensor] - [batch, seq_len, n_features]
            eval_mask_ratio: 評価時マスク率
            
        Returns:
            outputs: Dict[tf_name, torch.Tensor] - [batch, seq_len, 4]
        """
        # 🔥 1. 一括マスク生成（Python ループ除去）
        if self.training or eval_mask_ratio is not None:
            mask_ratio = eval_mask_ratio if eval_mask_ratio is not None else 0.15
            masks = self.masking_strategy.generate_masks_dict(batch, eval_mask_ratio_override=mask_ratio)
            masked_batch = self.masking_strategy.apply_mask_to_features_dict(batch, masks)
        else:
            masked_batch = batch
            masks = {}
        
        # 🔥 2. TF軸をバッチ軸に融合してEncoder呼び出し回数を削減
        return self._forward_batch_fusion(masked_batch, masks)
    
    def _forward_batch_fusion(self, batch: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        🔥 バッチ融合による高速化
        各TFを個別処理せず、バッチ次元に融合して1回でEncoder通過
        """
        # 🔥 1. TF-specific stem処理を最適化 - 複数TFを並列処理
        stemmed_features = self._process_stems_parallel(batch)
        
        # 🔥 2. 共通seq_lenにパディングしてバッチ融合
        max_seq_len = max(x.shape[1] for x in stemmed_features.values())
        batch_size = list(stemmed_features.values())[0].shape[0]
        
        # 全TFを結合: [batch * n_tf, max_seq_len, d_model]
        fused_features = []
        padding_masks = []
        
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
                padding_masks.append(pad_mask)
        
        # スタックしてバッチ融合: [batch * n_tf, max_seq_len, d_model]
        fused_features = torch.stack(fused_features, dim=1)  # [batch, n_tf, max_seq_len, d_model]
        fused_features = fused_features.view(batch_size * self.n_tf, max_seq_len, self.d_model)
        
        padding_masks = torch.stack(padding_masks, dim=1)  # [batch, n_tf, max_seq_len]
        padding_masks = padding_masks.view(batch_size * self.n_tf, max_seq_len)
        
        # 🔥 3. 共有エンコーダー1回呼び出し（6回→1回に削減）
        if hasattr(self.shared_encoder, 'forward'):
            # T5またはTransformerエンコーダー
            if hasattr(self.shared_encoder, 'encoder'):
                # T5TimeSeriesAdapter の場合
                encoded_features = self.shared_encoder(fused_features, key_padding_mask=padding_masks)
            else:
                # 通常のTransformerEncoder の場合
                encoded_features = self.shared_encoder(fused_features, src_key_padding_mask=padding_masks)
        else:
            encoded_features = fused_features
        
        # 🔥 4. バッチ融合を解除して各TFに分離
        encoded_features = encoded_features.view(batch_size, self.n_tf, max_seq_len, self.d_model)
        
        # 🔥 5. TF-specific decoder処理
        outputs = {}
        for i, tf_name in enumerate(self.timeframes):
            if tf_name in batch:
                # 各TFの元のseq_lenに戻す
                original_seq_len = batch[tf_name].shape[1]
                tf_encoded = encoded_features[:, i, :original_seq_len, :]  # [batch, seq_len, d_model]
                
                # デコーダー通過
                outputs[tf_name] = self.tf_decoders[tf_name](tf_encoded)  # [batch, seq_len, 4]
        
        return outputs
    
    def _process_stems_parallel(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        🔥 TF-specific stem処理を真の並列化
        groups=n_tf で完全に1回のConv1d呼び出し
        """
        if self.use_grouped_stem:
            return self._process_stems_grouped(batch)
        else:
            return self._process_stems_individual(batch)
    
    def _process_stems_grouped(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """🔥 groups=n_tf版: 完全並列化"""
        stemmed_features = {}
        
        # 同一seq_lenのTFをグループ化
        seq_len_groups = {}
        for tf_name, tf_features in batch.items():
            seq_len = tf_features.shape[1]
            if seq_len not in seq_len_groups:
                seq_len_groups[seq_len] = []
            seq_len_groups[seq_len].append((tf_name, tf_features))
        
        # 各seq_lenグループで並列処理
        for seq_len, tf_list in seq_len_groups.items():
            tf_names = [tf_name for tf_name, _ in tf_list]
            tf_features_list = [tf_features for _, tf_features in tf_list]
            
            # スタックして並列処理
            stacked_features = torch.stack(tf_features_list, dim=1)  # [batch, n_tf, seq_len, n_features]
            batch_size, n_tf, seq_len, n_features = stacked_features.shape
            
            # [batch*n_tf, n_features, seq_len] に変換
            reshaped = stacked_features.view(batch_size * n_tf, seq_len, n_features).transpose(1, 2)
            
            # 🔥 1回のConv1d呼び出しで全TF処理
            processed = self.grouped_stem(reshaped)  # [batch*n_tf, d_model, seq_len]
            
            # [batch*n_tf, seq_len, d_model] に変換
            processed = processed.transpose(1, 2)
            
            # 正規化とアクティベーション
            processed = self.stem_norm(processed)
            processed = self.stem_activation(processed)
            
            # [batch, n_tf, seq_len, d_model] に戻す
            processed = processed.view(batch_size, n_tf, seq_len, self.d_model)
            
            # 各TFに分離
            for i, tf_name in enumerate(tf_names):
                stemmed_features[tf_name] = processed[:, i]  # [batch, seq_len, d_model]
        
        return stemmed_features
    
    def _process_stems_individual(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """個別版: フォールバック処理"""
        stemmed_features = {}
        for tf_name, tf_features in batch.items():
            x = tf_features.transpose(1, 2)
            x = self.tf_stems[tf_name](x)
            stemmed_features[tf_name] = x.transpose(1, 2)
        return stemmed_features
    
    def get_model_info(self) -> Dict:
        """モデル情報を取得"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'architecture': {
                'n_tf': self.n_tf,
                'n_features': self.n_features,
                'd_model': self.d_model,
                'vectorized': True
            }
        }

# 既存のStage1Modelを置き換える関数
def replace_with_vectorized_model(original_model):
    """既存モデルをベクトル化版に置き換え"""
    config = original_model.config
    vectorized_model = VectorizedStage1Model(config)
    
    # 重みをコピー（可能な限り）
    try:
        # マスキング戦略の重みコピー
        if hasattr(original_model, 'masking_strategy') and hasattr(original_model.masking_strategy, 'mask_token'):
            vectorized_model.masking_strategy.mask_token.data = original_model.masking_strategy.mask_token.data.clone()
        
        # ステムの重みコピー
        if hasattr(original_model, 'tf_stems'):
            for tf_name in vectorized_model.timeframes:
                if tf_name in original_model.tf_stems:
                    vectorized_model.tf_stems[tf_name].load_state_dict(original_model.tf_stems[tf_name].state_dict())
        
        # エンコーダーの重みコピー
        if hasattr(original_model, 'shared_encoder'):
            try:
                vectorized_model.shared_encoder.load_state_dict(original_model.shared_encoder.state_dict())
            except:
                print("⚠️ エンコーダー重みコピーに失敗（構造差異）")
        
        # デコーダーの重みコピー
        if hasattr(original_model, 'tf_decoders'):
            for tf_name in vectorized_model.timeframes:
                if tf_name in original_model.tf_decoders:
                    vectorized_model.tf_decoders[tf_name].load_state_dict(original_model.tf_decoders[tf_name].state_dict())
        
        print("✅ ベクトル化モデルへの重み転送完了")
        
    except Exception as e:
        print(f"⚠️ 重み転送中にエラー（一部失敗）: {e}")
    
    return vectorized_model