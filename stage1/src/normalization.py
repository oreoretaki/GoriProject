#!/usr/bin/env python3
"""
Stage 1 正規化
TFごとのz-score正規化・統計キャッシュ
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class TFNormalizer:
    """TF別正規化クラス"""
    
    def __init__(self, config: dict, cache_stats: bool = True):
        """
        Args:
            config: 設定辞書
            cache_stats: 統計をファイルにキャッシュするか
        """
        self.config = config
        self.cache_stats = cache_stats
        self.timeframes = config['data']['timeframes']
        self.n_features = config['data']['n_features']
        
        # 統計ファイルパス
        data_dir = Path(config['data']['data_dir'])
        self.stats_file = data_dir / config['data']['stats_file']
        self.stats_train_file = data_dir / 'stats_train.json'  # train専用統計ファイル
        
        # 正規化統計 {tf: {'mean': [...], 'std': [...]}}
        self.stats = {}
        
        # 特徴量名（デバッグ用）
        self.feature_names = ['open', 'high', 'low', 'close', 'delta_close', 'body_ratio']
        
        print(f"📏 TFNormalizer初期化")
        print(f"   統計ファイル: {self.stats_file}")
        print(f"   キャッシュ有効: {cache_stats}")
        
    def fit(self, tf_data: Dict[str, pd.DataFrame]) -> None:
        """
        訓練データから正規化統計を計算
        
        Args:
            tf_data: {tf_name: DataFrame} TFデータ
        """
        print(f"📊 正規化統計計算中...")
        
        # 特徴量エンジニアが必要なのでインポート
        from .feature_engineering import FeatureEngineer
        feature_engineer = FeatureEngineer(self.config)
        
        for tf_name in self.timeframes:
            if tf_name not in tf_data:
                continue
                
            df = tf_data[tf_name]
            print(f"   {tf_name.upper()}: {len(df):,}レコード")
            
            # 特徴量計算（大量データの場合はサンプリング）
            if len(df) > 100000:
                # 大きなデータセットの場合は10%サンプリング
                sample_size = len(df) // 10
                indices = np.random.choice(len(df) - 1, sample_size, replace=False)
                indices = np.sort(indices)  # 時系列順を保持
                sample_df = df.iloc[indices]
            else:
                sample_df = df
                
            # 特徴量計算
            features = feature_engineer._calculate_features(sample_df)  # [n_samples, n_features]
            
            # 統計計算
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            
            # ゼロ分散回避
            std = np.where(std < 1e-8, 1.0, std)
            
            self.stats[tf_name] = {
                'mean': mean.tolist(),
                'std': std.tolist(),
                'n_samples': len(features)
            }
            
            print(f"     平均: {mean[:4].round(6)}")  # 最初の4特徴量のみ表示
            print(f"     標準偏差: {std[:4].round(6)}")
            
        # 統計をファイルに保存
        if self.cache_stats:
            self._save_stats()
            
    def load_stats(self, split: str = "all") -> None:
        """
        保存済み統計をロード
        
        Args:
            split: "all" (通常), "train" (train専用統計)
        """
        
        if split == "train":
            stats_file = self.stats_train_file
        else:
            stats_file = self.stats_file
            
        if not stats_file.exists():
            raise FileNotFoundError(f"統計ファイルが見つかりません: {stats_file}")
            
        with open(stats_file, 'r') as f:
            data = json.load(f)
            
        # 新形式（メタデータ付き）か旧形式かを判定
        if 'timeframes' in data:
            self.stats = data['timeframes']
        else:
            self.stats = data  # 旧形式
            
        print(f"📂 正規化統計ロード完了: {len(self.stats)}個のTF ({split})")
        
    def _save_stats(self) -> None:
        """統計をファイルに保存"""
        
        # メタデータ追加
        stats_with_meta = {
            'metadata': {
                'n_timeframes': len(self.timeframes),
                'n_features': self.n_features,
                'feature_names': self.feature_names,
                'normalization_method': 'zscore'
            },
            'timeframes': self.stats
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats_with_meta, f, indent=2)
            
        print(f"💾 正規化統計保存完了: {self.stats_file}")
        
    def save_stats(self, split: str = "all") -> None:
        """
        統計をファイルに保存（split指定可能）
        
        Args:
            split: "all" (通常), "train" (train専用統計)
        """
        if split == "train":
            # train専用統計を保存
            stats_with_meta = {
                'metadata': {
                    'n_timeframes': len(self.timeframes),
                    'n_features': self.n_features,
                    'feature_names': self.feature_names,
                    'normalization_method': 'zscore',
                    'split': 'train_only'
                },
                'timeframes': self.stats
            }
            
            with open(self.stats_train_file, 'w') as f:
                json.dump(stats_with_meta, f, indent=2)
                
            print(f"💾 train専用統計保存完了: {self.stats_train_file}")
        else:
            # 通常の統計保存
            self._save_stats()
        
    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        """
        特徴量を正規化
        
        Args:
            features: [n_tf, seq_len, n_features] 特徴量テンソル
            
        Returns:
            normalized: [n_tf, seq_len, n_features] 正規化済み特徴量
        """
        if not self.stats:
            raise ValueError("正規化統計が計算されていません。先にfit()またはload_stats()を実行してください。")
            
        normalized = features.clone()
        
        for i, tf_name in enumerate(self.timeframes):
            if tf_name not in self.stats:
                continue
                
            tf_stats = self.stats[tf_name]
            mean = torch.tensor(tf_stats['mean'], dtype=features.dtype, device=features.device)
            std = torch.tensor(tf_stats['std'], dtype=features.dtype, device=features.device)
            
            # z-score正規化: (x - mean) / std
            normalized[i] = (features[i] - mean) / std
            
        # NaN/Inf処理
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
        
    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        ターゲット（OHLC）を正規化
        
        Args:
            targets: [n_tf, seq_len, 4] OHLCターゲット
            
        Returns:
            normalized: [n_tf, seq_len, 4] 正規化済みターゲット
        """
        if not self.stats:
            raise ValueError("正規化統計が計算されていません。")
            
        normalized = targets.clone()
        
        for i, tf_name in enumerate(self.timeframes):
            if tf_name not in self.stats:
                continue
                
            tf_stats = self.stats[tf_name]
            # OHLC用の統計（最初の4特徴量）
            mean = torch.tensor(tf_stats['mean'][:4], dtype=targets.dtype, device=targets.device)
            std = torch.tensor(tf_stats['std'][:4], dtype=targets.dtype, device=targets.device)
            
            # z-score正規化
            normalized[i] = (targets[i] - mean) / std
            
        # NaN/Inf処理
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
        
    def normalize_single_tf(self, features: np.ndarray, tf_name: str) -> np.ndarray:
        """
        単一TFの特徴量を正規化（Model v2用）
        
        Args:
            features: [seq_len, n_features] 特徴量配列
            tf_name: タイムフレーム名
            
        Returns:
            normalized: [seq_len, n_features] 正規化済み特徴量
        """
        if not self.stats:
            raise ValueError("正規化統計が計算されていません。先にfit()またはload_stats()を実行してください。")
            
        if tf_name not in self.stats:
            print(f"⚠️ {tf_name}の統計が見つかりません。正規化せずに返します。")
            return features
            
        tf_stats = self.stats[tf_name]
        mean = np.array(tf_stats['mean'])
        std = np.array(tf_stats['std'])
        
        # z-score正規化: (x - mean) / std
        normalized = (features - mean) / std
        
        # NaN/Inf処理
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
        
    def normalize_targets_single_tf(self, targets: np.ndarray, tf_name: str) -> np.ndarray:
        """
        単一TFのターゲット（OHLC）を正規化（Model v2用）
        
        Args:
            targets: [seq_len, 4] OHLCターゲット配列
            tf_name: タイムフレーム名
            
        Returns:
            normalized: [seq_len, 4] 正規化済みターゲット
        """
        if not self.stats:
            raise ValueError("正規化統計が計算されていません。")
            
        if tf_name not in self.stats:
            print(f"⚠️ {tf_name}の統計が見つかりません。正規化せずに返します。")
            return targets
            
        tf_stats = self.stats[tf_name]
        # OHLC用の統計（最初の4特徴量）
        mean = np.array(tf_stats['mean'][:4])
        std = np.array(tf_stats['std'][:4])
        
        # z-score正規化
        normalized = (targets - mean) / std
        
        # NaN/Inf処理
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized
        
    def denormalize(self, normalized: torch.Tensor, tf_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        正規化済みデータを元のスケールに戻す
        
        Args:
            normalized: [n_tf, seq_len, n_features] 正規化済みテンソル
            tf_indices: TFインデックス（Noneの場合は全TF）
            
        Returns:
            denormalized: 元スケールに戻したテンソル
        """
        if not self.stats:
            raise ValueError("正規化統計が計算されていません。")
            
        denormalized = normalized.clone()
        
        tf_list = self.timeframes if tf_indices is None else [self.timeframes[i] for i in tf_indices]
        
        for i, tf_name in enumerate(tf_list):
            if tf_name not in self.stats:
                continue
                
            tf_stats = self.stats[tf_name]
            mean = torch.tensor(tf_stats['mean'], dtype=normalized.dtype, device=normalized.device)
            std = torch.tensor(tf_stats['std'], dtype=normalized.dtype, device=normalized.device)
            
            # 逆正規化: x * std + mean
            idx = i if tf_indices is None else tf_indices[i]
            denormalized[idx] = normalized[idx] * std + mean
            
        return denormalized
        
    def get_stats_summary(self) -> Dict:
        """統計サマリーを取得（デバッグ用）"""
        
        if not self.stats:
            return {"error": "統計が計算されていません"}
            
        summary = {}
        
        for tf_name, tf_stats in self.stats.items():
            mean = np.array(tf_stats['mean'])
            std = np.array(tf_stats['std'])
            
            summary[tf_name] = {
                'n_samples': tf_stats['n_samples'],
                'features': {}
            }
            
            for j, feature_name in enumerate(self.feature_names):
                summary[tf_name]['features'][feature_name] = {
                    'mean': float(mean[j]),
                    'std': float(std[j])
                }
                
        return summary