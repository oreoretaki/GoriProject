#!/usr/bin/env python3
"""
Stage 1 訓練スクリプト
Multi-TF Self-Supervised Reconstruction Training
"""

import os
# PyTorch分散学習を完全無効化（importより前に設定）
os.environ['PYTORCH_LIGHTNING_DISABLE_MPI'] = '1'
os.environ['PL_DISABLE_FORK'] = '1' 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['NCCL_DISABLE_WARN'] = '1'
os.environ['TORCH_DISTRIBUTED_DETAIL'] = 'OFF'

import sys
import argparse
import yaml
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Tensor Core最適化（PyTorch 2.0+）
torch.set_float32_matmul_precision('high')

# TF32を有効化（RTX 30xx/40xx/A100で高速化）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("✅ TF32とTensor Core最適化を有効化")
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# WSL/MPI互換性のための環境変数設定（importの直後に実行）
os.environ['PL_DISABLE_FORK'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
# MPI検出を完全無効化
os.environ['PYTORCH_LIGHTNING_DISABLE_MPI'] = '1'
os.environ['SLURM_DISABLED'] = '1'

# プロジェクトルートをPATHに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

from src.data_loader import create_stage1_dataloaders
from src.model import Stage1Model
from src.losses import Stage1CombinedLoss

# T5転移学習用インポート
try:
    from src.lm_adapter import GradualUnfreezingCallback, create_differential_learning_rate_groups
    T5_CALLBACKS_AVAILABLE = True
except ImportError:
    T5_CALLBACKS_AVAILABLE = False
    GradualUnfreezingCallback = None
    create_differential_learning_rate_groups = None

class CustomProgressBar(TQDMProgressBar):
    """◆ カスタムプログレスバー：重要メトリクスのみ表示"""
    
    def __init__(self, refresh_rate: int = 10):
        super().__init__(refresh_rate=refresh_rate)
    
    def get_metrics(self, trainer, pl_module):
        # 既定メトリクスを取得
        metrics = super().get_metrics(trainer, pl_module)
        
        # Lightningが自動で付けるsuffixとのマッピング
        rename_map = {
            "train_loss_step": "train_loss",
            "train_loss_epoch": "train_loss_ep",  # エポック終了時用に分離
            "val_loss": "val_loss_ep",
            "val_loss_live": "val_loss",  # ライブ版を主表示
            "val_correlation": "val_corr_ep",
            "val_corr_live": "val_corr",  # ライブ版を主表示
            "lr-AdamW": "lr",
            "grad_norm": "grad_norm",
            "amp_overflow": "amp",  # 短縮表示
        }
        
        filtered = {}
        for src_key, dst_key in rename_map.items():
            if src_key in metrics:
                val = metrics[src_key]
                # ★ Tensor → float変換
                if isinstance(val, torch.Tensor):
                    val = val.item()
                # フォーマット整形
                if isinstance(val, float):
                    if dst_key in {"train_loss", "train_loss_ep", "val_loss", "val_loss_ep"}:
                        val = f"{val:.4f}"
                    elif dst_key in {"val_corr", "val_corr_ep"}:
                        val = f"{val:+.3f}"
                    elif dst_key == "grad_norm":
                        val = f"{val:.2e}"  # 指数表記で桁幅を抑える
                    elif dst_key == "lr":
                        val = f"{val:.2e}"  # 指数表記
                    elif dst_key == "amp":
                        val = f"{int(val)}"
                filtered[dst_key] = val
        
        return filtered

class Stage1LightningModule(pl.LightningModule):
    """Stage 1 PyTorch Lightning モジュール"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # モデル
        self.model = Stage1Model(config)
        
        # 損失関数
        self.criterion = Stage1CombinedLoss(config)
        
        # メトリクス
        self.train_losses = []
        self.val_losses = []
        
        print("⚡ Stage1LightningModule初期化完了")
        print(f"   モデル情報: {self.model.get_model_info()}")
        
    def forward(self, features, masks=None):
        return self.model(features, masks)
        
    def training_step(self, batch, batch_idx):
        features = batch['features']  # [batch, n_tf, seq_len, n_features]
        targets = batch['targets']    # [batch, n_tf, seq_len, 4]
        masks = batch['masks']        # [batch, n_tf, seq_len]
        
        # Forward pass
        outputs = self.model(features, masks)
        reconstructed = outputs['reconstructed']
        
        # M1データを抽出（クロス損失用）
        m1_data = targets[:, 0]  # [batch, seq_len, 4]
        
        # 損失計算
        losses = self.criterion(reconstructed, targets, masks, m1_data)
        
        # ◆ 学習損失を毎ステップでログ（プログレスバーに表示）
        loss = losses['total']
        self.log("train_loss_step", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # 学習率もログ（プログレスバーに表示）
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr-AdamW", current_lr, on_step=True, prog_bar=True, logger=True)
        
        # 詳細損失もログ（エポック単位のみ）
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':  # totalは上記で既にログ済み
                self.log(f'train_{loss_name}', loss_value, 
                        on_step=False, on_epoch=True, prog_bar=False, logger=True)
            
        return loss
    
    def on_train_epoch_start(self):
        """エポック開始時の初期化"""
        self.overflow_count = 0
        self._amp_scale_start = None
        # BF16では scaler が None のため、安全にアクセス
        if (hasattr(self.trainer, 'precision_plugin') and 
            hasattr(self.trainer.precision_plugin, 'scaler') and 
            self.trainer.precision_plugin.scaler is not None):
            self._amp_scale_start = self.trainer.precision_plugin.scaler.get_scale()
    
    def on_after_backward(self) -> None:
        """勾配ノルム監視 & AMPオーバーフロー検知"""
        
        # ---- 1) grad_norm 計測 & ログ ----
        grad_norms = [p.grad.detach().float().norm() 
                      for p in self.parameters() if p.grad is not None]
        if grad_norms:
            grad_norm = torch.linalg.vector_norm(torch.stack(grad_norms), ord=2)
        else:
            grad_norm = torch.tensor(0.0)
            
        # inf/nanを1e3に丸めて可視化だけは続行
        if not torch.isfinite(grad_norm):
            grad_norm = torch.tensor(1e3, device=self.device)
            
        self.log("grad_norm", grad_norm,
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # ---- 2) AMPオーバーフロー検知（GradScalerから正確に取得）----
        overflow = 0.0
        if (hasattr(self.trainer, 'precision_plugin') and 
            hasattr(self.trainer.precision_plugin, 'scaler') and 
            self.trainer.precision_plugin.scaler is not None):
            scaler = self.trainer.precision_plugin.scaler
            # スケールが0ならオーバーフロー発生
            if hasattr(scaler, '_scale') and scaler._scale is not None:
                overflow = 1.0 if scaler._scale.item() == 0 else 0.0
                
        self.log("amp_overflow", overflow,
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # ---- 3) 勾配クリッピング（オーバーフロー対策もLightningに任せる）----
        # クリッピング前後の値を記録
        grad_norm_before = grad_norm if torch.isfinite(grad_norm) else torch.tensor(1e3)
        grad_norm_after = torch.nn.utils.clip_grad_norm_(self.parameters(), 
                                                         max_norm=self.config['training']['gradient_clip'])
        # クリップされたかを記録
        if grad_norm_before > self.config['training']['gradient_clip']:
            clipped_ratio = grad_norm_after / grad_norm_before
            self.log("grad_norm_clipped", clipped_ratio,
                     on_step=True, on_epoch=False, prog_bar=False, logger=True)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """オプティマイザーステップ"""
        # AMP/GradScalerの内部ロジックを壊さないようLightningに任せる
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        targets = batch['targets']
        masks = batch['masks']
        
        # Forward pass
        outputs = self.model(features, masks)
        reconstructed = outputs['reconstructed']
        
        # M1データを抽出
        m1_data = targets[:, 0]
        
        # 損失計算
        losses = self.criterion(reconstructed, targets, masks, m1_data)
        
        # ◆ 検証損失をプログレスバーに表示（エポック終了時）
        self.log("val_loss", losses['total'],
                 on_epoch=True, prog_bar=True, logger=True)
        
        # ◆ リアルタイム版（検証中に即座に表示）
        self.log("val_loss_live", losses['total'],
                 on_step=True, on_epoch=False, prog_bar=True, logger=False)
        
        # 詳細損失（プログレスバーには表示しない）
        for loss_name, loss_value in losses.items():
            if loss_name != 'total':  # totalは上記で既にログ済み
                self.log(f'val_{loss_name}', loss_value, 
                        on_epoch=True, prog_bar=False, logger=True)
            
        # 相関メトリクス計算（検証のみ）
        correlations = self._calculate_correlations(reconstructed, targets, masks)
        for tf_idx, corr in enumerate(correlations):
            tf_name = self.config['data']['timeframes'][tf_idx]
            self.log(f'val_corr_{tf_name}', corr, on_epoch=True, prog_bar=False, logger=True)
            
        # ◆ 平均相関をプログレスバーに表示（エポック終了時）
        mean_corr = torch.mean(torch.stack(correlations))
        self.log('val_correlation', mean_corr, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_correlation_mean', mean_corr, on_epoch=True, prog_bar=False, logger=True)  # 後方互換性
        
        # ◆ リアルタイム版（検証中に即座に表示）
        self.log("val_corr_live", mean_corr,
                 on_step=True, on_epoch=False, prog_bar=True, logger=False)
        
        return losses['total']
        
    def _calculate_correlations(self, pred, target, masks):
        """TFごとの相関を計算"""
        correlations = []
        
        for tf_idx in range(pred.size(1)):
            pred_tf = pred[:, tf_idx]  # [batch, seq_len, 4]
            target_tf = target[:, tf_idx]
            mask_tf = masks[:, tf_idx]  # [batch, seq_len]
            
            # マスクされた部分のみで相関計算
            if mask_tf.sum() > 0:
                pred_masked = pred_tf[mask_tf.bool()]  # [n_masked, 4]
                target_masked = target_tf[mask_tf.bool()]
                
                if pred_masked.numel() > 0:
                    # ピアソン相関（4つのOHLC特徴量の平均）
                    corr_ohlc = []
                    for feat_idx in range(4):
                        pred_feat = pred_masked[:, feat_idx]
                        target_feat = target_masked[:, feat_idx]
                        
                        if pred_feat.numel() > 1:
                            corr = torch.corrcoef(torch.stack([pred_feat, target_feat]))[0, 1]
                            if not torch.isnan(corr):
                                corr_ohlc.append(corr)
                                
                    if corr_ohlc:
                        mean_corr = torch.mean(torch.stack(corr_ohlc))
                        correlations.append(mean_corr)
                    else:
                        correlations.append(torch.tensor(0.0, device=pred.device))
                else:
                    correlations.append(torch.tensor(0.0, device=pred.device))
            else:
                correlations.append(torch.tensor(0.0, device=pred.device))
                
        return correlations
    
    def on_validation_epoch_end(self):
        """検証エポック終了時：val_corrの平均をプログレスバーに表示"""
        # 既にログされたval_correlation_meanをプログレスバーにも表示
        if 'val_correlation_mean' in self.trainer.callback_metrics:
            val_corr_mean = self.trainer.callback_metrics['val_correlation_mean']
            self.log('val_corr_mean', val_corr_mean, 
                     on_step=False, on_epoch=True, prog_bar=True, logger=False)
    
    def on_after_backward(self):
        """勾配計算後：TensorBoardに勾配ヒストグラムを記録"""
        # ヒストグラム記録を一時的に無効化（empty histogram エラー回避）
        # 重要なレイヤーの勾配ノルムのみ記録
        if hasattr(self.model, 'shared_encoder'):
            # エンコーダーの勾配ノルム
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.shared_encoder.parameters(), 
                float('inf')
            )
            self.log('grad_norm/encoder', encoder_grad_norm, on_step=True, on_epoch=False)
        
        # T5使用時はT5部分の勾配ノルムも記録
        if (hasattr(self.model, 'shared_encoder') and 
            hasattr(self.model.shared_encoder, 't5_encoder')):
            t5_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.shared_encoder.t5_encoder.parameters(), 
                float('inf')
            )
            self.log('grad_norm/t5', t5_grad_norm, on_step=True, on_epoch=False)
    
    def on_train_epoch_end(self):
        """エポック終了時のAMPスケール変化ログ"""
        if hasattr(self.trainer, 'precision_plugin') and hasattr(self.trainer.precision_plugin, 'scaler'):
            current_scale = self.trainer.precision_plugin.scaler.get_scale()
            if self._amp_scale_start is not None:
                scale_change = current_scale / self._amp_scale_start
                self.log('amp_scale_change', scale_change, on_epoch=True, prog_bar=False, logger=True)
                # スケールが大きく下がった場合のみ警告
                if scale_change < 0.5:
                    print(f"⚠️ AMPスケールが大幅に減少: {self._amp_scale_start:.1f} → {current_scale:.1f}")
    
    def test_step(self, batch, batch_idx):
        """テストステップ（検証ステップと同じ）"""
        return self.validation_step(batch, batch_idx)
        
    def configure_optimizers(self):
        """オプティマイザーとスケジューラー設定（T5 Layerwise LR Decay対応）"""
        
        # 基準学習率の取得
        base_lr = self.config['training']['optimizer'].get('lr', self.config['training']['scheduler']['max_lr'])
        
        # T5転移学習が有効な場合は差分学習率を適用
        if (T5_CALLBACKS_AVAILABLE and 
            self.config.get('transfer_learning', {}).get('use_pretrained_lm', False)):
            
            # Layerwise LR Decay設定の取得
            layerwise_lr_decay = self.config['training'].get('layerwise_lr_decay')
            t5_lr_top = self.config['training'].get('t5_lr_top')
            t5_lr_factor = self.config.get('transfer_learning', {}).get('lm_learning_rate_factor', 0.1)
            
            param_groups = create_differential_learning_rate_groups(
                self.model, 
                base_lr=base_lr, 
                t5_lr_factor=t5_lr_factor,
                layerwise_lr_decay=layerwise_lr_decay,
                t5_lr_top=t5_lr_top
            )
            
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=self.config['training']['optimizer']['betas'],
                weight_decay=self.config['training']['optimizer']['weight_decay']
            )
            
            # 設定情報をログ出力
            if layerwise_lr_decay is not None and t5_lr_top is not None:
                print(f"🔧 Layerwise LR Decay: base_lr={base_lr:.2e}, t5_top_lr={t5_lr_top:.2e}, decay={layerwise_lr_decay}")
            else:
                print(f"🤗 T5差分学習率: base_lr={base_lr:.2e}, t5_lr={base_lr*t5_lr_factor:.2e}")
            
            # パラメータグループの詳細をログ出力
            for i, group in enumerate(param_groups):
                print(f"  ParamGroup[{i}] ({group.get('name', 'unknown')}): lr={group['lr']:.2e}")
            
        else:
            # 従来の単一学習率
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=base_lr,
                betas=self.config['training']['optimizer']['betas'],
                weight_decay=self.config['training']['optimizer']['weight_decay']
            )
            print(f"📐 単一学習率: lr={base_lr:.2e}")
        
        # スケジューラー
        scheduler_config = self.config['training']['scheduler']
        total_steps = self.trainer.estimated_stepping_batches
        
        # Linear Warmup + Cosine Decay スケジューラーの実装
        if scheduler_config['name'].lower() == 'linear_with_warmup':
            # Warmup設定の計算
            warmup_epochs = self.config['training'].get('warmup_epochs', 3)
            steps_per_epoch = total_steps // self.config['training']['epochs']
            num_warmup_steps = steps_per_epoch * warmup_epochs  # 970 * 3 = 2910
            
            print(f"📐 Linear Warmup スケジューラー設定:")
            print(f"  - Warmup steps: {num_warmup_steps} ({warmup_epochs} epochs)")
            print(f"  - Total steps: {total_steps}")
            print(f"  - Steps per epoch: {steps_per_epoch}")
            
            # PyTorchの標準的なLinear Warmup + Cosine Decayの実装
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, num_warmup_steps))
                else:
                    # Cosine decay
                    progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            # 従来のOneCycleLR
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=scheduler_config['max_lr'],
                total_steps=total_steps,
                div_factor=scheduler_config['div_factor'],
                final_div_factor=scheduler_config['final_div_factor'],
                pct_start=scheduler_config['pct_start']
            )
        
        # スケジューラー設定取得
        interval = scheduler_config.get('interval', 'epoch')
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': interval
            }
        }

def load_config(config_path: str) -> dict:
    """設定ファイル読み込み（継承対応）"""
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # extends指定がある場合は親設定を読み込んでマージ
    if 'extends' in config:
        parent_path = config_path.parent / config['extends']
        parent_config = load_config(str(parent_path))
        # 親設定をベースに現在の設定で上書き
        merged_config = deep_merge(parent_config, config)
        # extends自体は最終設定から削除
        merged_config.pop('extends', None)
        return merged_config
    
    return config

def deep_merge(base: dict, override: dict) -> dict:
    """設定を再帰的にマージ"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def main():
    # PyTorch分散学習を完全無効化してWSL互換性を向上
    os.environ['PL_DISABLE_FORK'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'gloo'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['PYTORCH_LIGHTNING_DISABLE_MPI'] = '1'
    os.environ['SLURM_DISABLED'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['NCCL_DISABLE_WARN'] = '1'
    os.environ['TORCH_DISTRIBUTED_DETAIL'] = 'OFF'
    
    parser = argparse.ArgumentParser(description='Stage 1 Training')
    parser.add_argument('--config', type=str, required=True, help='設定ファイルパス')
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリ')
    parser.add_argument('--devices', type=int, default=1, help='デバイス数')
    parser.add_argument('--resume_from', type=str, default=None, help='チェックポイントから再開')
    parser.add_argument('--fast_dev_run', action='store_true', help='開発用高速実行（1バッチのみ）')
    parser.add_argument('--max_epochs', type=int, default=None, help='最大エポック数上書き')
    parser.add_argument('--profiler', type=str, default=None, help='プロファイラー（simple, advanced）')
    parser.add_argument('--dry_run', action='store_true', help='ドライラン（データ確認のみ）')
    parser.add_argument('--plot_lr', action='store_true', help='学習率カーブをプロット')
    parser.add_argument('--batch_size', type=int, default=None, help='バッチサイズ上書き')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='勾配クリッピング値上書き')
    parser.add_argument('--check_early_stop', action='store_true', help='早期停止動作テスト')
    
    args = parser.parse_args()
    
    # 設定読み込み
    config = load_config(args.config)
    config['data']['data_dir'] = args.data_dir
    
    # 引数での設定上書き
    if args.max_epochs:
        config['training']['epochs'] = args.max_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.gradient_clip_val:
        config['training']['gradient_clip'] = args.gradient_clip_val
    
    print("🚀 Stage 1 訓練開始")
    print(f"   設定ファイル: {args.config}")
    print(f"   データディレクトリ: {args.data_dir}")
    print(f"   デバイス数: {args.devices}")
    
    # シード設定
    pl.seed_everything(config['runtime']['seed'])
    
    # データローダー作成
    print("📊 データローダー作成中...")
    train_loader, val_loader = create_stage1_dataloaders(args.data_dir, config)
    
    # モデル作成
    print("🧠 モデル初期化中...")
    model = Stage1LightningModule(config)
    
    # PyTorch 2.0 コンパイル最適化（事前ウォームアップ付き）
    if torch.__version__ >= '2.0.0':
        print("🚀 PyTorch 2.0 コンパイル最適化を適用中...")
        try:
            # GPUに移動
            if torch.cuda.is_available():
                model = model.cuda()
            
            # 1) コンパイル用にFP32に統一
            print("🔧 コンパイル用にFP32に統一...")
            model.model = model.model.to(torch.float32)
            
            # 2) 事前ウォームアップでコンパイル時間を隠蔽（FP32で）
            print("🔥 ダミー入力でウォームアップ実行中...")
            with torch.no_grad():
                # バッチサイズ1でダミー入力作成（FP32）
                dummy_features = torch.randn(1, 6, 128, 36, device=model.device, dtype=torch.float32)
                dummy_masks = torch.ones(1, 6, 128, device=model.device, dtype=torch.bool)
                
                # ウォームアップ実行
                _ = model.model(dummy_features, dummy_masks)
                print("✅ ウォームアップ完了")
            
            # 3) コンパイル適用（FP32モデルで）
            model.model = torch.compile(model.model, backend="inductor", mode="max-autotune")
            print("✅ TorchCompile適用完了（事前ウォームアップ済み）")
            
        except Exception as e:
            print(f"⚠️ TorchCompile失敗、通常モード: {e}")
    else:
        print("⚠️ PyTorch 2.0+が必要です（TorchCompileをスキップ）")
    
    # コールバック設定
    callbacks = []
    
    # チェックポイント保存
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.config).parent.parent / 'checkpoints',
        filename='stage1-{epoch:02d}-{val_correlation_mean:.4f}',
        monitor='val_correlation_mean',
        mode='max',
        save_top_k=config['logging']['save_top_k'],
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早期停止
    early_stopping = EarlyStopping(
        monitor='val_correlation_mean',
        mode='max',
        patience=config['training']['early_stop']['patience'],
        min_delta=config['training']['early_stop']['min_delta']
    )
    callbacks.append(early_stopping)
    
    # 学習率モニタリング
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # ◆ カスタムプログレスバー（10ステップごとに更新）
    custom_progress = CustomProgressBar(refresh_rate=10)
    callbacks.append(custom_progress)
    
    # T5転移学習用コールバック
    if T5_CALLBACKS_AVAILABLE and config.get('transfer_learning', {}).get('use_pretrained_lm', False):
        freeze_epochs = config.get('transfer_learning', {}).get('freeze_lm_epochs', 3)
        unfreezing_callback = GradualUnfreezingCallback(freeze_epochs=freeze_epochs)
        callbacks.append(unfreezing_callback)
        print(f"🤗 T5段階的解凍コールバックを追加 (freeze_epochs={freeze_epochs})")
    
    # ロガー
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=Path(args.config).parent.parent / 'logs',
            name='stage1'
        )
    except ImportError:
        print("⚠️ TensorBoard未インストール - CSVロガーを使用")
        from pytorch_lightning.loggers import CSVLogger
        logger = CSVLogger(
            save_dir=Path(args.config).parent.parent / 'logs',
            name='stage1'
        )
    
    # トレーナー設定（MPI完全回避）
    trainer_kwargs = {
        'max_epochs': config['training']['epochs'],
        'devices': 1 if torch.cuda.is_available() and args.devices > 0 else 'auto',
        'accelerator': 'gpu' if torch.cuda.is_available() and args.devices > 0 else 'cpu',
        'strategy': 'auto',  # DDPを避けてautoに設定
        'precision': config['training']['precision'],
        'accumulate_grad_batches': config['training']['accumulate_grad_batches'],
        'gradient_clip_val': config['training']['gradient_clip'],
        'callbacks': callbacks,
        'logger': logger,
        'log_every_n_steps': config['logging']['log_every_n_steps'],
        'check_val_every_n_epoch': 1,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'num_nodes': 1,  # 単一ノード強制
        'sync_batchnorm': False,  # バッチ正規化同期無効
        'use_distributed_sampler': False,  # 分散サンプラー無効
    }
    
    # オプション引数を追加
    if args.fast_dev_run:
        trainer_kwargs['fast_dev_run'] = True
    if args.profiler:
        trainer_kwargs['profiler'] = args.profiler
        
    # 開発用設定（バッチ数制限）
    if 'development' in config:
        if 'limit_train_batches' in config['development']:
            trainer_kwargs['limit_train_batches'] = config['development']['limit_train_batches']
        if 'limit_val_batches' in config['development']:
            trainer_kwargs['limit_val_batches'] = config['development']['limit_val_batches']
        
    trainer = pl.Trainer(**trainer_kwargs)
    
    # LR Finder実行（設定で有効化されている場合）
    if config.get('lr_finder', {}).get('enabled', False):
        print("🔍 Learning Rate Finder実行中...")
        import matplotlib.pyplot as plt
        
        # LR Finder設定
        lr_finder_config = config['lr_finder']
        min_lr = float(lr_finder_config.get('min_lr', 1e-8))
        max_lr = float(lr_finder_config.get('max_lr', 1.0))
        num_training = int(lr_finder_config.get('num_training', 100))
        save_path = lr_finder_config.get('save_path', 'lr_finder_results')
        
        # 結果保存ディレクトリ作成
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        print(f"   範囲: {min_lr:.1e} ～ {max_lr:.1e}")
        print(f"   ステップ数: {num_training}")
        
        try:
            # PyTorch Lightning 2.4.0 対応の LR Finder実行
            from pytorch_lightning.tuner import Tuner
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                model,
                train_dataloaders=train_loader,
                min_lr=min_lr,
                max_lr=max_lr,
                num_training=num_training,
                mode='exponential',
                early_stop_threshold=4.0,  # 損失が4倍になったら停止
                update_attr=False  # モデルの学習率は更新しない
            )
            
            # 結果プロット
            fig = lr_finder.plot(suggest=True, show=False)
            
            # 推奨学習率を取得
            suggested_lr = lr_finder.suggestion()
            
            # プロット保存
            plot_path = save_dir / "lr_finder_plot.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # 結果ファイル保存
            results_path = save_dir / "lr_finder_results.txt"
            with open(results_path, 'w') as f:
                f.write(f"Learning Rate Finder Results\n")
                f.write(f"=" * 40 + "\n")
                f.write(f"Configuration:\n")
                f.write(f"  Min LR: {min_lr:.1e}\n")
                f.write(f"  Max LR: {max_lr:.1e}\n")
                f.write(f"  Steps: {num_training}\n")
                f.write(f"\nResults:\n")
                f.write(f"  Suggested LR: {suggested_lr:.1e}\n")
                f.write(f"  Current base_lr: {config['training']['optimizer']['lr']:.1e}\n")
                if 't5_lr_top' in config['training']:
                    f.write(f"  Current t5_lr_top: {config['training']['t5_lr_top']:.1e}\n")
                f.write(f"\nRecommendations:\n")
                f.write(f"  - Set optimizer.lr to: {suggested_lr:.1e}\n")
                if 't5_lr_top' in config['training']:
                    t5_suggested = suggested_lr * 0.25  # T5用により低い学習率
                    f.write(f"  - Set t5_lr_top to: {t5_suggested:.1e}\n")
            
            print(f"✅ LR Finder完了")
            print(f"   推奨学習率: {suggested_lr:.1e}")
            print(f"   現在の学習率: {config['training']['optimizer']['lr']:.1e}")
            print(f"   プロット: {plot_path}")
            print(f"   詳細結果: {results_path}")
            
            # T5使用時の推奨値も表示
            if 't5_lr_top' in config['training']:
                current_t5_lr = config['training']['t5_lr_top']
                t5_suggested = suggested_lr * 0.25
                print(f"   T5現在値: {current_t5_lr:.1e}")
                print(f"   T5推奨値: {t5_suggested:.1e}")
            
            print("\n💡 推奨アクション:")
            print(f"   1. shared_base.yaml の optimizer.lr を {suggested_lr:.1e} に設定")
            if 't5_lr_top' in config['training']:
                print(f"   2. shared_base.yaml の t5_lr_top を {suggested_lr * 0.25:.1e} に設定")
            print("   3. 設定を保存して再度訓練を実行")
            
            # LR Finder実行後は終了
            print("\n🔄 LR Finder完了。設定を更新して再実行してください。")
            return
            
        except Exception as e:
            print(f"❌ LR Finder実行エラー: {e}")
            print("⚠️  通常の訓練を続行します...")
    
    # ドライラン処理
    if args.dry_run:
        print("🧪 ドライラン実行中...")
        print(f"   データローダー情報:")
        print(f"     訓練バッチ数: {len(train_loader)}")
        print(f"     検証バッチ数: {len(val_loader)}")
        
        # 1バッチテスト
        sample_batch = next(iter(train_loader))
        print(f"     バッチ形状: {[k + ': ' + str(v.shape) for k, v in sample_batch.items()]}")
        
        if args.plot_lr:
            # LRスケジュール可視化（実装省略）
            print("   LRスケジュール: OneCycleLR設定済み")
        
        print("✅ ドライラン完了")
        return
    
    # 訓練実行
    if args.resume_from:
        print(f"📂 チェックポイントから再開: {args.resume_from}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # 最良モデル情報
    print("✅ 訓練完了")
    print(f"   最良チェックポイント: {checkpoint_callback.best_model_path}")
    print(f"   最良スコア: {checkpoint_callback.best_model_score}")
    
    # 最終評価（fast_dev_runでは省略）
    if not args.fast_dev_run:
        print("📈 最終評価実行中...")
        # 開発モードの場合はテストもバッチ数制限
        if 'development' in config and 'limit_val_batches' in config['development']:
            # 新しいトレーナーを作成（テスト用にlimit_test_batchesを設定）
            test_trainer_kwargs = trainer_kwargs.copy()
            test_trainer_kwargs['limit_test_batches'] = config['development']['limit_val_batches']
            test_trainer = pl.Trainer(**test_trainer_kwargs)
            test_trainer.test(model, val_loader)
        else:
            trainer.test(model, val_loader)

if __name__ == "__main__":
    main()