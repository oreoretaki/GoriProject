#!/usr/bin/env python3
"""
TF別ウィンドウサンプラー（リーク完全遮断版）
SingleTFWindowSampler + MultiTFWindowSampler（ラッパー）
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.ndimage import convolve1d
import time
import hashlib
from pathlib import Path
warnings.filterwarnings('ignore')

class SingleTFWindowSampler:
    """単一TF用ウィンドウサンプラー（リーク防止・TF固有gap適用）"""
    
    def __init__(
        self,
        tf_name: str,
        tf_data: pd.DataFrame,
        seq_len: int,
        split: str = "train",
        val_split: float = 0.2,
        min_coverage: float = 0.8,
        cache_dir: Optional[str] = None,
        val_gap_days: float = 1.0
    ):
        """
        Args:
            tf_name: タイムフレーム名 ('m1', 'm5', etc.)
            tf_data: 単一TFのDataFrame
            seq_len: このTFでのシーケンス長（M1基準から自動計算）
            split: "train" or "val"
            val_split: 検証データ割合
            min_coverage: 最小データカバレッジ
            cache_dir: キャッシュディレクトリ
            val_gap_days: 訓練と検証の間の時間的ギャップ（日数）
        """
        self.tf_name = tf_name
        self.tf_data = tf_data
        self.split = split
        self.val_split = val_split
        self.min_coverage = min_coverage
        self.cache_dir = Path(cache_dir) / "windows" if cache_dir else Path("./cache/windows")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.val_gap_days = val_gap_days
        
        # TF固有の設定
        self.step_map = {
            'm1': 1, 'm5': 5, 'm15': 15, 'm30': 30, 
            'h1': 60, 'h4': 240, 'd': 1440
        }
        self.tf_step_minutes = self.step_map.get(tf_name, 1)
        
        # M1基準seq_lenをこのTFに合わせて変換
        # M1=128なら、M5=128/5=25.6→26, H1=128/60=2.13→3
        m1_duration_minutes = seq_len * 1  # M1は1分間隔
        self.seq_len = max(1, int(m1_duration_minutes / self.tf_step_minutes))
        
        print(f"🔍 SingleTFWindowSampler({tf_name})")
        print(f"   データ期間: {tf_data.index[0]} - {tf_data.index[-1]}")
        print(f"   レコード数: {len(tf_data):,}")
        print(f"   シーケンス長: {self.seq_len} (M1={seq_len}基準)")
        print(f"   TF間隔: {self.tf_step_minutes}分")
        
        # 有効ウィンドウ検索
        self.valid_windows = self._find_valid_windows()
        
        # 訓練/検証分割（TF固有gap適用）
        self.split_windows = self._split_windows()
        
        print(f"   総ウィンドウ数: {len(self.valid_windows)}")
        print(f"   {split}ウィンドウ数: {len(self.split_windows)}")
        
    def _find_valid_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """単一TFでの有効ウィンドウを検索"""
        n_windows = len(self.tf_data) - self.seq_len + 1
        
        if n_windows <= 0:
            return []
            
        # キャッシュファイル名（TF固有）
        data_hash = hashlib.md5(str(self.tf_data.index[[0, -1]]).encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"windows_{data_hash}_{self.tf_name}.npy"
        
        if cache_file.exists():
            print(f"   📂 キャッシュから読み込み: {cache_file.name}")
            valid_indices = np.load(cache_file)
        else:
            print(f"   🔍 有効ウィンドウ検索中: {n_windows:,} 候補")
            start_time = time.time()
            
            # 単純に連続データの存在をチェック
            valid_indices = []
            for i in range(n_windows):
                start_idx = i
                end_idx = i + self.seq_len
                
                # データ存在チェック（NaN以外）
                window_data = self.tf_data.iloc[start_idx:end_idx]
                valid_ratio = (~window_data.isna()).all(axis=1).mean()
                
                if valid_ratio >= self.min_coverage:
                    start_ts = self.tf_data.index[start_idx]
                    end_ts = self.tf_data.index[end_idx - 1]
                    valid_indices.append((start_ts, end_ts))
            
            # キャッシュ保存
            np.save(cache_file, valid_indices)
            print(f"   💾 キャッシュ保存: {cache_file.name}")
            print(f"   ⚡ 処理時間: {time.time() - start_time:.2f}秒")
        
        return valid_indices
    
    def _split_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """TF固有gap適用での訓練/検証分割"""
        n_total = len(self.valid_windows)
        n_val = int(n_total * self.val_split)
        
        if n_val == 0:
            return self.valid_windows if self.split == "train" else []
        
        # TF固有のgap計算
        val_gap_minutes = int(self.val_gap_days * 24 * 60)
        tf_gap_windows = int(val_gap_minutes / self.tf_step_minutes)
        
        print(f"   🕐 TF固有ギャップ: {self.val_gap_days}日 = {val_gap_minutes}分")
        print(f"   📊 {self.tf_name} gap窓数: {tf_gap_windows} ({self.tf_step_minutes}分間隔)")
        
        if self.split == "train":
            # 訓練: 最後の (n_val + tf_gap_windows) を除外
            return self.valid_windows[:-(n_val + tf_gap_windows)]
        else:  # val
            # 検証: 最後の n_val のみ使用
            val_windows = self.valid_windows[-n_val:]
            
            if val_windows:
                first_val_ts = val_windows[0][0]
                print(f"   [DBG] {self.tf_name} 検証開始: {first_val_ts}")
                
                # ギャップ検証
                if n_val + tf_gap_windows < len(self.valid_windows):
                    last_train_window = self.valid_windows[-(n_val + tf_gap_windows) - 1]
                    last_train_ts = last_train_window[1]
                    gap_actual = (first_val_ts - last_train_ts).total_seconds() / 86400
                    print(f"   [DBG] {self.tf_name} 実際ギャップ: {gap_actual:.1f}日")
            
            return val_windows
    
    def __len__(self) -> int:
        return len(self.split_windows)
    
    def __getitem__(self, idx: int) -> pd.DataFrame:
        """指定インデックスのウィンドウデータを取得"""
        if idx >= len(self.split_windows):
            raise IndexError(f"Index {idx} out of range for {len(self.split_windows)} windows")
        
        start_time, end_time = self.split_windows[idx]
        
        # 時間範囲でデータ取得
        window_data = self.tf_data.loc[start_time:end_time]
        
        # 右端整列（最新のseq_len分を取得）
        if len(window_data) > self.seq_len:
            window_data = window_data.tail(self.seq_len)
        elif len(window_data) < self.seq_len:
            # パディング（前方をNaNで埋める）
            padding_needed = self.seq_len - len(window_data)
            padding_index = pd.date_range(
                end=window_data.index[0] - pd.Timedelta(minutes=self.tf_step_minutes),
                periods=padding_needed,
                freq=f"{self.tf_step_minutes}min"
            )
            padding_df = pd.DataFrame(
                np.nan, 
                index=padding_index,
                columns=window_data.columns
            )
            window_data = pd.concat([padding_df, window_data])
        
        return window_data

class MultiTFWindowSampler:
    """マルチTF同期ウィンドウサンプラー（ラッパー版・リーク完全遮断）"""
    
    def __init__(
        self,
        tf_data: Dict[str, pd.DataFrame],
        seq_len: int,
        split: str = "train",
        val_split: float = 0.2,
        min_coverage: float = 0.8,
        cache_dir: Optional[str] = None,
        val_gap_days: float = 1.0
    ):
        """
        Args:
            tf_data: {tf_name: DataFrame} 形式のTFデータ
            seq_len: M1でのシーケンス長
            split: "train" or "val"
            val_split: 検証データ割合
            min_coverage: 最小データカバレッジ（全TFでデータが存在する割合）
            cache_dir: キャッシュディレクトリ
            val_gap_days: 訓練と検証の間の時間的ギャップ（日数）
        """
        self.tf_data = tf_data
        self.seq_len = seq_len
        self.split = split
        self.val_split = val_split
        self.min_coverage = min_coverage
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.val_gap_days = val_gap_days
        
        # タイムフレーム名リスト
        self.timeframes = list(tf_data.keys())
        
        print(f"🔄 MultiTFWindowSampler初期化 ({split}) - ラッパー版")
        print(f"   TF数: {len(self.timeframes)}")
        print(f"   TF: {self.timeframes}")
        
        # 各TFに対してSingleTFWindowSamplerを作成
        self.tf_samplers = {}
        sample_counts = []
        valid_timeframes = []
        
        for tf_name, tf_df in tf_data.items():
            sampler = SingleTFWindowSampler(
                tf_name=tf_name,
                tf_data=tf_df,
                seq_len=seq_len,
                split=split,
                val_split=val_split,
                min_coverage=min_coverage,
                cache_dir=str(self.cache_dir),
                val_gap_days=val_gap_days
            )
            
            # 🔥 有効なサンプラーのみ保持（IndexError回避）
            if len(sampler) > 0:
                self.tf_samplers[tf_name] = sampler
                sample_counts.append(len(sampler))
                valid_timeframes.append(tf_name)
                print(f"   ✅ {tf_name}: {len(sampler):,} windows")
            else:
                print(f"   ❌ {tf_name}: サンプル数0 - 除外")
        
        # 有効なTFリストを更新
        self.timeframes = valid_timeframes
        
        # 最小サンプル数を安全に計算
        self.min_samples = min(sample_counts) if sample_counts else 0
        
        if self.min_samples == 0:
            raise ValueError("全TFでサンプル数0 - データローダーを作成できません")
        
        print(f"📊 MultiTFWindowSampler統計:")
        print(f"   有効TF数: {len(self.timeframes)}")
        print(f"   有効TF: {self.timeframes}")
        print(f"   最小サンプル数: {self.min_samples:,}")
    
    def __len__(self) -> int:
        """最小サンプル数を返す（全TFで同期）"""
        return self.min_samples
    
    def __getitem__(self, idx: int) -> Dict[str, pd.DataFrame]:
        """全TFの同期ウィンドウデータを取得"""
        if idx >= self.min_samples:
            raise IndexError(f"Index {idx} out of range for {self.min_samples} synchronized windows")
        
        # 各TFからウィンドウデータを取得
        tf_windows = {}
        for tf_name in self.timeframes:
            sampler = self.tf_samplers[tf_name]
            tf_windows[tf_name] = sampler[idx]
        
        return tf_windows
