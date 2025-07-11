#!/usr/bin/env python3
"""
T5 Language Model Adapter for Time Series
T5エンコーダーを時系列データに適応させるアダプター
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import T5EncoderModel, T5Config
    from huggingface_hub.errors import RepositoryNotFoundError
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    T5EncoderModel = None
    T5Config = None
    RepositoryNotFoundError = OSError  # フォールバック


class PatchEmbedding(nn.Module):
    """時系列データをパッチ化してT5互換の埋め込みに変換"""
    
    def __init__(self, 
                 n_features: int,
                 patch_len: int, 
                 d_model: int,
                 n_timeframes: int):
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.d_model = d_model
        self.n_timeframes = n_timeframes
        
        # パッチサイズの特徴量次元
        patch_dim = n_features * patch_len
        
        # 各タイムフレーム用の投影層
        self.patch_projections = nn.ModuleList([
            nn.Linear(patch_dim, d_model) for _ in range(n_timeframes)
        ])
        
        # 位置エンコーディング
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, d_model, dtype=torch.float32) * 0.02)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, n_tf, seq_len, n_features]
        Returns:
            patches: [batch, n_tf, n_patches, d_model]
            attention_mask: [batch, n_tf, n_patches]
        """
        batch_size, n_tf, seq_len, n_features = x.shape
        
        # パッチ数を計算（ceil使用で適切な数を確保）
        n_patches = max(1, math.ceil(seq_len / self.patch_len))
        required_len = n_patches * self.patch_len
        
        # パディングが必要かチェック
        if seq_len < required_len:
            pad_len = required_len - seq_len
            # NaNパディング（右端合わせ）
            x = F.pad(x, (0, 0, 0, pad_len, 0, 0), value=float('nan'))  # [batch, n_tf, seq+pad, n_feat]
        elif seq_len > required_len:
            # 長すぎる場合は切り詰め
            x = x[:, :, :required_len, :]
        
        effective_len = required_len
        
        # パッチ化: [batch, n_tf, n_patches, patch_len, n_features]
        x = x.view(batch_size, n_tf, n_patches, self.patch_len, n_features)
        
        # パッチを平坦化: [batch, n_tf, n_patches, patch_dim]
        patch_dim = self.patch_len * n_features
        x = x.view(batch_size, n_tf, n_patches, patch_dim)
        
        # 各タイムフレームごとに投影
        patches = []
        for tf_idx in range(n_tf):
            tf_x = x[:, tf_idx]  # [batch, n_patches, patch_dim]
            tf_patches = self.patch_projections[tf_idx](tf_x)  # [batch, n_patches, d_model]
            patches.append(tf_patches)
        
        patches = torch.stack(patches, dim=1)  # [batch, n_tf, n_patches, d_model]
        
        # 位置エンコーディング追加
        pos_emb = self.pos_embedding[:, :n_patches, :].unsqueeze(1)  # [1, 1, n_patches, d_model]
        patches = patches + pos_emb
        
        # T5互換正規化とスケーリング
        patches = self.layer_norm(patches)
        patches = patches / math.sqrt(self.d_model)  # T5想定分布に合わせる
        patches = self.dropout(patches)
        
        # アテンションマスク (すべて有効)
        attention_mask = torch.ones(batch_size, n_tf, n_patches, 
                                   dtype=torch.bool, device=x.device)
        
        return patches, attention_mask
    
    def forward_single_tf(self, x: torch.Tensor, tf_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        単一TF用のパッチ埋め込み (async mode)
        
        Args:
            x: [batch, seq_len, n_features]
            tf_idx: タイムフレームインデックス (projection選択用)
        Returns:
            patches: [batch, n_patches, d_model]
            attention_mask: [batch, n_patches]
        """
        batch_size, seq_len, n_features = x.shape
        
        # パッチ数を計算（ceil使用で適切な数を確保）
        n_patches = max(1, math.ceil(seq_len / self.patch_len))
        required_len = n_patches * self.patch_len
        
        # パディングが必要かチェック
        if seq_len < required_len:
            pad_len = required_len - seq_len
            # NaNパディング（右端合わせ）
            x = F.pad(x, (0, 0, 0, pad_len), value=float('nan'))  # [batch, seq+pad, n_feat]
        elif seq_len > required_len:
            # 長すぎる場合は切り詰め
            x = x[:, :required_len, :]
        
        effective_len = required_len
        
        # パッチ化: [batch, n_patches, patch_len, n_features]
        x = x.view(batch_size, n_patches, self.patch_len, n_features)
        
        # パッチを平坦化: [batch, n_patches, patch_dim]
        # 🔥 実際のpatch_dimを正確に計算（16*6=96, not 16384）
        patch_dim = self.patch_len * n_features
        x = x.view(batch_size, n_patches, patch_dim)
        
        # Dynamic patch projection for async mode
        # async modeでは実際のinput featuresに合わせて動的に処理
        expected_patch_dim = self.patch_len * self.n_features
        if patch_dim != expected_patch_dim:
            # 動的にlinear layerを作成（初回のみ）
            dynamic_key = f"dynamic_patch_proj_{tf_idx}_{patch_dim}"
            if not hasattr(self, dynamic_key):
                dynamic_proj = nn.Linear(patch_dim, self.d_model).to(x.device)
                setattr(self, dynamic_key, dynamic_proj)
                print(f"🔧 Dynamic patch projection created: {patch_dim} -> {self.d_model}")
            
            patches = getattr(self, dynamic_key)(x)
        else:
            # 通常の投影層を使用
            tf_idx = min(tf_idx, len(self.patch_projections) - 1)  # 範囲外対策
            patches = self.patch_projections[tf_idx](x)  # [batch, n_patches, d_model]
        
        # 位置エンコーディング追加
        pos_emb = self.pos_embedding[:, :n_patches, :]  # [1, n_patches, d_model]
        patches = patches + pos_emb
        
        # T5互換正規化とスケーリング
        patches = self.layer_norm(patches)
        patches = patches / math.sqrt(self.d_model)
        patches = self.dropout(patches)
        
        # アテンションマスク (すべて有効)
        attention_mask = torch.ones(batch_size, n_patches, 
                                   dtype=torch.bool, device=x.device)
        
        return patches, attention_mask


class T5TimeSeriesAdapter(nn.Module):
    """T5エンコーダーを時系列データに適応させるアダプター"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for T5 adapter. "
                "Install with: pip install transformers>=4.42.0"
            )
        
        # 設定の抽出
        transfer_config = config.get('transfer_learning', {})
        self.lm_name_or_path = transfer_config.get('lm_name_or_path', 'google/t5-small')
        self.patch_len = transfer_config.get('patch_len', 16)
        self.freeze_lm_epochs = transfer_config.get('freeze_lm_epochs', 3)
        
        # データ次元
        self.n_features = config['data']['n_features']
        self.n_timeframes = config['data']['n_timeframes']
        self.seq_len = config['data']['seq_len']
        
        # T5の設定を取得（オフライン対応）
        try:
            self.t5_config = T5Config.from_pretrained(
                self.lm_name_or_path,
                local_files_only=False,  # まずオンラインを試行
                use_auth_token=False
            )
        except (OSError, RepositoryNotFoundError) as e:
            # 事前学習済みT5が必要なので、ロードに失敗したら停止
            raise RuntimeError(
                f"❌ {self.lm_name_or_path} の事前学習済み重みが見つかりません。\n"
                f"転移学習には事前学習済みモデルが必須です。\n\n"
                f"解決方法:\n"
                f"1. オンライン環境: インターネット接続・認証トークンを確認\n"
                f"2. オフライン環境: 以下でモデルを事前ダウンロード\n"
                f"   huggingface-cli download google/t5-small\n\n"
                f"元のエラー: {e}"
            )
            
        self.d_model = self.t5_config.d_model  # 通常512 (T5-small)
        
        # パッチ埋め込み層
        self.patch_embedding = PatchEmbedding(
            n_features=self.n_features,
            patch_len=self.patch_len,
            d_model=self.d_model,
            n_timeframes=self.n_timeframes
        )
        
        # T5エンコーダーをロード（オフライン対応）
        try:
            print(f"🤗 T5エンコーダーをロード中: {self.lm_name_or_path}")
            self.t5_encoder = T5EncoderModel.from_pretrained(
                self.lm_name_or_path,
                local_files_only=False,
                use_auth_token=False
            )
        except (OSError, RepositoryNotFoundError) as e:
            # 事前学習済みT5が必要なので、ロードに失敗したら停止
            raise RuntimeError(
                f"❌ T5エンコーダー {self.lm_name_or_path} の事前学習済み重みが見つかりません。\n"
                f"転移学習には事前学習済みモデルが必須です。\n"
                f"元のエラー: {e}"
            )
        
        # 出力投影層（T5のd_modelからStage-1のd_modelに変換）
        stage1_d_model = config['model']['encoder']['d_model']
        self.output_projection = nn.Linear(self.d_model, stage1_d_model)
        
        # 🔥 T5エンコーダーは最初から解凍状態（凍結機能廃止）
        print(f"🔓 T5エンコーダーは最初から解凍状態で開始")
        
        print(f"✅ T5TimeSeriesAdapter初期化完了")
        print(f"   T5 d_model: {self.d_model}")
        print(f"   Stage-1 d_model: {stage1_d_model}")
        print(f"   Patch length: {self.patch_len}")
        print(f"   凍結機能: 廃止（常に解凍状態）")
    
    # 🔥 凍結機能は廃止 - T5エンコーダーは常に解凍状態
    
    def get_model_info(self) -> Dict:
        """モデル情報を返す"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        t5_params = sum(p.numel() for p in self.t5_encoder.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            't5_parameters': t5_params,
            't5_frozen': not next(self.t5_encoder.parameters()).requires_grad,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # float32を仮定
        }
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, n_tf, seq_len, n_features] or [batch, seq_len, n_features] (async mode)
            key_padding_mask: [batch, seq_len] padding mask (optional, async mode only)
        Returns:
            encoded: [batch, n_tf, seq_len, d_model] or [batch, seq_len, d_model] (Stage-1互換)
        """
        # 入力の次元数で処理方法を分岐
        if x.dim() == 3:
            # Async mode: [batch, seq_len, n_features]
            return self._forward_single_tf(x, key_padding_mask)
        elif x.dim() == 4:
            # Legacy mode: [batch, n_tf, seq_len, n_features]
            return self._forward_multi_tf(x)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
    
    def _forward_single_tf(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        単一TF用forward (async mode)
        
        Args:
            x: [batch, seq_len, n_features] 
            key_padding_mask: [batch, seq_len] padding mask (True=valid, False=padded)
        Returns:
            encoded: [batch, seq_len, d_model]
        """
        batch_size, seq_len, n_features = x.shape
        
        # 単一TF用にreshape: [batch, 1, seq_len, n_features]
        x_reshaped = x.unsqueeze(1)
        
        # パッチ埋め込み（単一TF用）
        patches, attention_mask = self.patch_embedding.forward_single_tf(x_reshaped[:, 0], tf_idx=0)
        # patches: [batch, n_patches, d_model]
        # attention_mask: [batch, n_patches]
        
        # key_padding_maskがある場合、パッチレベルのマスクに変換
        if key_padding_mask is not None:
            # key_padding_maskを反転（T5は True=valid, False=padded を期待）
            valid_mask = ~key_padding_mask  # [batch, seq_len]
            
            # パッチレベルのマスクに変換
            n_patches_actual = patches.size(1)
            if n_patches_actual > 0:
                patch_stride = max(1, seq_len // n_patches_actual) if seq_len > 0 else 1
                patch_mask = torch.zeros(batch_size, n_patches_actual, device=x.device, dtype=torch.bool)
                
                for p in range(n_patches_actual):
                    start_idx = p * patch_stride
                    end_idx = min(start_idx + self.patch_len, seq_len)
                    if start_idx < seq_len and end_idx > start_idx:
                        # パッチ内に有効データが少なくとも1つあればvalid
                        patch_mask[:, p] = valid_mask[:, start_idx:end_idx].any(dim=1)
                    else:
                        # 範囲外の場合は無効
                        patch_mask[:, p] = False
                
                # attention_maskと統合
                attention_mask = attention_mask & patch_mask
        
        # T5エンコーダーに入力
        encoder_outputs = self.t5_encoder(
            inputs_embeds=patches,
            attention_mask=attention_mask.to(torch.float32)
        )
        
        # 最後の隠れ状態を取得
        encoded = encoder_outputs.last_hidden_state  # [batch, n_patches, d_model]
        
        # パッチからシーケンスに復元（補間）
        encoded = self._patches_to_sequence(encoded, seq_len)  # [batch, seq_len, d_model]
        
        # Stage-1のd_modelに投影
        encoded = self.output_projection(encoded)
        
        return encoded
        
    def _forward_multi_tf(self, x: torch.Tensor) -> torch.Tensor:
        """
        マルチTF用forward (legacy mode)
        
        Args:
            x: [batch, n_tf, seq_len, n_features]
        Returns:
            encoded: [batch, n_tf, seq_len, d_model]
        """
        batch_size, n_tf, seq_len, n_features = x.shape
        
        # パッチ埋め込み
        patches, attention_mask = self.patch_embedding(x)
        # patches: [batch, n_tf, n_patches, d_model]
        
        # 各タイムフレームごとにT5エンコーダーを適用
        encoded_patches = []
        
        for tf_idx in range(n_tf):
            tf_patches = patches[:, tf_idx]  # [batch, n_patches, d_model]
            tf_mask = attention_mask[:, tf_idx]  # [batch, n_patches]
            
            # T5エンコーダーに入力
            encoder_outputs = self.t5_encoder(
                inputs_embeds=tf_patches,
                attention_mask=tf_mask.to(torch.float32)
            )
            
            # 最後の隠れ状態を取得
            encoded = encoder_outputs.last_hidden_state  # [batch, n_patches, d_model]
            encoded_patches.append(encoded)
        
        # タイムフレーム次元を復元
        encoded_patches = torch.stack(encoded_patches, dim=1)  # [batch, n_tf, n_patches, d_model]
        
        # パッチからシーケンスに復元（補間）
        n_patches = encoded_patches.size(2)
        patch_seq_len = n_patches * self.patch_len
        
        if patch_seq_len != seq_len:
            # 長さが異なる場合は補間
            encoded_patches = F.interpolate(
                encoded_patches.permute(0, 1, 3, 2),  # [batch, n_tf, d_model, n_patches]
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 1, 3, 2)  # [batch, n_tf, seq_len, d_model]
        else:
            # パッチを展開
            encoded_patches = encoded_patches.repeat_interleave(self.patch_len, dim=2)
            encoded_patches = encoded_patches[:, :, :seq_len, :]
        
        # Stage-1のd_modelに投影
        output = self.output_projection(encoded_patches)
        
        return output
    
    def _patches_to_sequence(self, patches: torch.Tensor, target_seq_len: int) -> torch.Tensor:
        """
        パッチからシーケンスに復元（補間）
        
        Args:
            patches: [batch, n_patches, d_model]
            target_seq_len: 目標シーケンス長
        Returns:
            sequence: [batch, target_seq_len, d_model]
        """
        batch_size, n_patches, d_model = patches.shape
        patch_seq_len = n_patches * self.patch_len
        
        if patch_seq_len != target_seq_len:
            # 長さが異なる場合は補間
            sequence = F.interpolate(
                patches.permute(0, 2, 1),  # [batch, d_model, n_patches]
                size=target_seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [batch, target_seq_len, d_model]
        else:
            # パッチを展開
            sequence = patches.repeat_interleave(self.patch_len, dim=1)
            sequence = sequence[:, :target_seq_len, :]
        
        return sequence


try:
    from pytorch_lightning.callbacks import Callback
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    # フォールバック用のダミーCallbackクラス
    class Callback:
        pass

class GradualUnfreezingCallback(Callback):
    """PyTorch Lightning Callback for gradual unfreezing of T5 encoder"""
    
    def __init__(self, freeze_epochs: int = 3):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.unfrozen = False
    
    def on_train_epoch_start(self, trainer, pl_module):
        """エポック開始時にT5エンコーダーの凍結状態を制御"""
        current_epoch = trainer.current_epoch
        
        print(f"🔍 GradualUnfreezingCallback: epoch={current_epoch}, freeze_epochs={self.freeze_epochs}, unfrozen={self.unfrozen}")
        
        # T5アダプターが使用されているかチェック
        if hasattr(pl_module.model, 'shared_encoder') and \
           isinstance(pl_module.model.shared_encoder, T5TimeSeriesAdapter):
            
            print(f"🔍 T5TimeSeriesAdapter検出済み")
            
            if current_epoch >= self.freeze_epochs and not self.unfrozen:
                pl_module.model.shared_encoder.unfreeze_t5_encoder()
                self.unfrozen = True
                print(f"🔓 エポック{current_epoch}: T5エンコーダーの凍結を解除")
            elif current_epoch < self.freeze_epochs and self.unfrozen:
                # 再凍結（通常は発生しないが念のため）
                pl_module.model.shared_encoder.freeze_t5_encoder()
                self.unfrozen = False
                print(f"🔒 エポック{current_epoch}: T5エンコーダーを再凍結")
            else:
                print(f"🔍 解凍条件不満: current_epoch={current_epoch}, freeze_epochs={self.freeze_epochs}, unfrozen={self.unfrozen}")
        else:
            print(f"🔍 T5TimeSeriesAdapter未検出")


def create_differential_learning_rate_groups(model, base_lr: float, t5_lr_factor: float = 0.1, 
                                          layerwise_lr_decay: float = None, t5_lr_top: float = None):
    """T5部分とその他で異なる学習率を設定するためのパラメータグループを作成
    
    Args:
        model: モデル
        base_lr: ヘッド & Adapter 用の基準 LR
        t5_lr_factor: T5学習率係数（layerwise_lr_decayが指定されていない場合に使用）
        layerwise_lr_decay: 下位層ほどLRを減衰させる係数（例: 0.85）
        t5_lr_top: T5最上層の学習率（layerwise_lr_decayと組み合わせて使用）
    """
    
    param_groups = []
    other_params = []
    
    # T5アダプターの検出（共有エンコーダーを優先）
    t5_encoder = None
    if hasattr(model, 'shared_encoder') and isinstance(model.shared_encoder, T5TimeSeriesAdapter):
        t5_encoder = model.shared_encoder.t5_encoder
    elif hasattr(model, 'encoders') and isinstance(model.encoders, nn.ModuleDict):
        # async_samplerモードでのフォールバック（非T5モード）
        for tf, encoder in model.encoders.items():
            if isinstance(encoder, T5TimeSeriesAdapter):
                t5_encoder = encoder.t5_encoder
                break
    
    if t5_encoder is not None:
        
        # Layerwise LR Decayが指定された場合
        if layerwise_lr_decay is not None and t5_lr_top is not None:
            # T5エンコーダーのブロック数取得
            if hasattr(t5_encoder, 'encoder') and hasattr(t5_encoder.encoder, 'block'):
                num_layers = len(t5_encoder.encoder.block)
                
                # 各ブロックごとに異なる学習率を設定
                for i, block in enumerate(t5_encoder.encoder.block):
                    # i=0が最下層、i=num_layers-1が最上層
                    decay_factor = layerwise_lr_decay ** (num_layers - 1 - i)
                    lr_i = t5_lr_top * decay_factor
                    
                    param_groups.append({
                        'params': list(block.parameters()),
                        'lr': lr_i,
                        'name': f't5_block_{i}'
                    })
                
                # その他のT5パラメータ（embedding, final_layer_normなど）
                t5_block_params = {p for block in t5_encoder.encoder.block for p in block.parameters()}
                other_t5_params = [p for p in t5_encoder.parameters() if p not in t5_block_params]
                
                if other_t5_params:
                    param_groups.append({
                        'params': other_t5_params,
                        'lr': t5_lr_top,
                        'name': 't5_other'
                    })
                
                print(f"🔧 Layerwise LR Decay適用: {num_layers}層, top_lr={t5_lr_top:.2e}, decay={layerwise_lr_decay}")
                
                # 最下層と最上層のLRを表示
                bottom_lr = t5_lr_top * (layerwise_lr_decay ** (num_layers - 1))
                print(f"   └─ T5 block_0 (最下層): lr={bottom_lr:.2e}")
                print(f"   └─ T5 block_{num_layers-1} (最上層): lr={t5_lr_top:.2e}")
                
            else:
                # ブロック構造が見つからない場合は従来方式にフォールバック
                print("⚠️ T5ブロック構造が見つかりません。従来の差分学習率を使用")
                t5_params = [p for p in t5_encoder.parameters() if p.requires_grad]
                if t5_params:
                    param_groups.append({
                        'params': t5_params,
                        'lr': t5_lr_top,
                        'name': 'T5_encoder'
                    })
        else:
            # 従来の単純な差分学習率
            t5_params = [p for p in t5_encoder.parameters() if p.requires_grad]
            if t5_params:
                param_groups.append({
                    'params': t5_params,
                    'lr': base_lr * t5_lr_factor,
                    'name': 'T5_encoder'
                })
        
        # その他のパラメータ（T5エンコーダー以外）
        t5_encoder_params = {p for p in t5_encoder.parameters()}
        for name, param in model.named_parameters():
            if param.requires_grad and param not in t5_encoder_params:
                other_params.append(param)
    else:
        # T5が使用されていない場合は全て通常学習率
        other_params = [p for p in model.parameters() if p.requires_grad]
    
    # ヘッド & Adapter パラメータ
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'head_and_adapter'
        })
    
    return param_groups