#!/usr/bin/env python3
"""
Stage 1 最適化データローダー
高速化最適化付きDataLoader実装
"""

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import os
from pathlib import Path

from .window_sampler import MultiTFWindowSampler
from .feature_engineering import FeatureEngineer
from .normalization import TFNormalizer

class Stage1Dataset(Dataset):
    """Stage 1 データセット（最適化版）"""
    
    def __init__(self, data_dir: str, config: dict, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # configにdata_dirを設定
        self.config['data']['data_dir'] = str(self.data_dir)
        
        print(f"🔄 Stage 1 Dataset初期化 ({split})")
        print(f"   データディレクトリ: {data_dir}")
        
        # TFデータ読み込み
        self.tf_data = self._load_tf_data()
        
        # 特徴量エンジニアリング
        self.feature_engineer = FeatureEngineer(config)
        
        # 正規化
        self.normalizer = TFNormalizer(
            config=config,
            cache_stats=True
        )
        
        # 統計情報をロード（train専用統計優先）
        try:
            if split == "train":
                # trainの場合、まずtrain専用統計を試行
                try:
                    self.normalizer.load_stats(split="train")
                    print(f"   📊 train専用統計をロード")
                except FileNotFoundError:
                    print(f"   📊 train専用統計を新規計算中...")
                    self.normalizer.fit(self.tf_data)
                    self.normalizer.save_stats(split="train")
            else:
                # valの場合、train専用統計を読み込み
                try:
                    self.normalizer.load_stats(split="train")
                    print(f"   📊 train専用統計をロード（val用）")
                except FileNotFoundError:
                    print(f"   ⚠️ train専用統計が見つかりません。全期間統計を使用")
                    self.normalizer.load_stats()
        except FileNotFoundError:
            print(f"   📊 正規化統計を新規計算中...")
            self.normalizer.fit(self.tf_data)
        
        # ウィンドウサンプリング（ベクトル化+キャッシュ）
        cache_dir = self.data_dir / "cache"
        self.window_sampler = MultiTFWindowSampler(
            tf_data=self.tf_data,
            seq_len=config['data']['seq_len'],
            split=split,
            val_split=config['validation']['val_split'],
            min_coverage=0.8,
            cache_dir=str(cache_dir),
            val_gap_days=config['validation'].get('val_gap_days', 1.0)
        )
        
        # 注意：マスキングはモデル内で実行（data_loader側では生データを返す）
        
    def _load_tf_data(self) -> Dict[str, pd.DataFrame]:
        """TFデータを高速読み込み"""
        tf_data = {}
        timeframes = self.config['data']['timeframes']
        
        print(f"   時間足: {timeframes}")
        print(f"   シーケンス長: {self.config['data']['seq_len']}")
        
        for tf in timeframes:
            file_path = self.data_dir / f"simple_gap_aware_{tf}.parquet"
            
            if file_path.exists():
                # Parquet高速読み込み
                df = pd.read_parquet(file_path)
                
                # インデックス修正: M1以外はtimestamp列をインデックスに使用
                if tf == 'm1':
                    df.index = pd.to_datetime(df.index)
                else:
                    # timestamp列をインデックスに設定
                    if 'timestamp' in df.columns:
                        df.index = pd.to_datetime(df['timestamp'])
                        df = df.drop('timestamp', axis=1)  # timestamp列を削除
                    else:
                        df.index = pd.to_datetime(df.index)
                
                # タイムゾーン統一 (UTCに統一)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                elif str(df.index.tz) != 'UTC':
                    df.index = df.index.tz_convert('UTC')
                
                tf_data[tf] = df
                print(f"   {tf.upper()}: {len(df):,}レコード, 期間: {df.index[0]} - {df.index[-1]}")
            else:
                raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
                
        return tf_data
    
    def __len__(self) -> int:
        return len(self.window_sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """最適化されたバッチ取得"""
        # ウィンドウデータ取得
        window_data = self.window_sampler[idx]
        
        # 特徴量エンジニアリング
        features, targets = self.feature_engineer.process_window(window_data)
        
        # 正規化（features用）
        features = self.normalizer.normalize(features)
        
        # ターゲット用の正規化（OHLC用）
        targets = self.normalizer.normalize_targets(targets)
        
        # 既にテンソル形式（BF16互換）
        features_tensor = features.to(torch.float32)
        targets_tensor = targets.to(torch.float32)
        
        # 🔥 生データを返す（マスキングはモデル内で実行）
        return {
            'features': features_tensor,  # 生の特徴量（マスクなし）
            'targets': targets_tensor
        }

def create_stage1_dataloaders(data_dir: str, config: dict) -> Tuple[DataLoader, DataLoader]:
    """最適化されたDataLoader作成"""
    
    # DataLoader設定取得（デフォルト値付き）
    dataloader_config = config.get('dataloader', {})
    batch_size = config['training']['batch_size']
    
    # 最適化設定
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': dataloader_config.get('num_workers', 8),
        'pin_memory': dataloader_config.get('pin_memory', True),
        'persistent_workers': dataloader_config.get('persistent_workers', True),
        'prefetch_factor': dataloader_config.get('prefetch_factor', 4),
        'drop_last': True,  # バッチサイズ統一
    }
    
    # データセット作成
    train_dataset = Stage1Dataset(data_dir, config, split="train")
    val_dataset = Stage1Dataset(data_dir, config, split="val")
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )
    
    print(f"📊 DataLoader作成完了")
    print(f"   訓練: {len(train_loader)}バッチ ({len(train_dataset)}サンプル)")
    print(f"   検証: {len(val_loader)}バッチ ({len(val_dataset)}サンプル)")
    print(f"   最適化: num_workers={dataloader_kwargs['num_workers']}, prefetch={dataloader_kwargs['prefetch_factor']}")
    
    return train_loader, val_loader