#!/usr/bin/env python3
"""
Stage 1 特徴量エンジニアリング
OHLC -> [open, high, low, close, Δclose, %body] 変換
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """特徴量エンジニアリングクラス"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: 設定辞書
        """
        self.config = config
        self.timeframes = config['data']['timeframes']
        self.n_features = config['data']['n_features']  # 6特徴量
        
        # 特徴量名
        self.feature_names = ['open', 'high', 'low', 'close', 'delta_close', 'body_ratio']
        
        print(f"🔧 FeatureEngineer初期化")
        print(f"   特徴量数: {self.n_features}")
        print(f"   特徴量: {self.feature_names}")
        
    def process_window(self, window_data: Dict[str, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ウィンドウデータを特徴量とターゲットに変換
        
        Args:
            window_data: {tf_name: DataFrame} マルチTFウィンドウ
            
        Returns:
            features: [n_tf, seq_len, n_features] 特徴量テンソル
            targets: [n_tf, seq_len, 4] ターゲット（OHLC）テンソル
        """
        n_tf = len(self.timeframes)
        
        # 各TFの長さを取得（M1以外は可変長の可能性）
        tf_lengths = {tf: len(window_data[tf]) for tf in self.timeframes}
        max_len = max(tf_lengths.values())
        
        # テンソル初期化
        features = torch.zeros(n_tf, max_len, self.n_features)
        targets = torch.zeros(n_tf, max_len, 4)  # OHLC
        
        for i, tf in enumerate(self.timeframes):
            df = window_data[tf]
            tf_len = len(df)
            
            if tf_len == 0:
                continue
                
            # OHLC抽出
            ohlc = df[['open', 'high', 'low', 'close']].values
            
            # 特徴量計算
            tf_features = self._calculate_features(df)
            
            # テンソルに格納（右端整列）
            start_idx = max_len - tf_len
            features[i, start_idx:, :] = torch.from_numpy(tf_features)
            targets[i, start_idx:, :] = torch.from_numpy(ohlc)
            
        return features, targets
        
    def _calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        単一TFのDataFrameから6特徴量を計算
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            features: [seq_len, 6] 特徴量配列
        """
        seq_len = len(df)
        features = np.zeros((seq_len, self.n_features))
        
        # 基本OHLC
        features[:, 0] = df['open'].values      # open
        features[:, 1] = df['high'].values      # high
        features[:, 2] = df['low'].values       # low
        features[:, 3] = df['close'].values     # close
        
        # Δclose: 終値の変化量
        close_values = df['close'].values
        delta_close = np.zeros_like(close_values)
        if seq_len > 1:
            delta_close[1:] = np.diff(close_values)
            # 最初の値は前のバーがないので0とする
            delta_close[0] = 0.0
        features[:, 4] = delta_close
        
        # %body: ローソク足実体の割合
        body_ratio = self._calculate_body_ratio(df)
        features[:, 5] = body_ratio
        
        # NaNや無限大の処理
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    def _calculate_body_ratio(self, df: pd.DataFrame) -> np.ndarray:
        """
        ローソク足実体の割合を計算
        %body = |close - open| / (high - low) * 100
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            body_ratio: 実体の割合 (0-100)
        """
        open_val = df['open'].values
        high_val = df['high'].values
        low_val = df['low'].values
        close_val = df['close'].values
        
        # 実体サイズ
        body_size = np.abs(close_val - open_val)
        
        # 全体レンジ（高値-安値）
        total_range = high_val - low_val
        
        # ゼロ除算回避
        body_ratio = np.where(
            total_range > 1e-8,  # 非常に小さな値を閾値とする
            (body_size / total_range) * 100.0,
            0.0
        )
        
        # 0-100の範囲にクリップ
        body_ratio = np.clip(body_ratio, 0.0, 100.0)
        
        return body_ratio
        
    def get_feature_stats(self, window_data: Dict[str, pd.DataFrame]) -> Dict:
        """特徴量の統計情報を取得（デバッグ用）"""
        
        features, targets = self.process_window(window_data)
        
        stats = {}
        for i, tf in enumerate(self.timeframes):
            tf_features = features[i]  # [seq_len, n_features]
            
            # 非ゼロ部分のみ（パディング除外）
            non_zero_mask = torch.any(tf_features != 0, dim=1)
            if non_zero_mask.sum() > 0:
                valid_features = tf_features[non_zero_mask]
                
                stats[tf] = {
                    'shape': valid_features.shape,
                    'mean': valid_features.mean(dim=0).tolist(),
                    'std': valid_features.std(dim=0).tolist(),
                    'min': valid_features.min(dim=0)[0].tolist(),
                    'max': valid_features.max(dim=0)[0].tolist()
                }
            else:
                stats[tf] = {'shape': (0, self.n_features), 'note': 'No valid data'}
                
        return stats