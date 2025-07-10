#!/usr/bin/env python3
"""
マルチTFウィンドウサンプラー
同じカレンダーウィンドウを全TFからスライス・右端整列
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

class MultiTFWindowSampler:
    """マルチTF同期ウィンドウサンプラー"""
    
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
        
        # M1をベースとして使用
        self.base_tf = 'm1'
        if self.base_tf not in tf_data:
            raise ValueError(f"ベースTF '{self.base_tf}' がデータに存在しません")
            
        # TFごとのステップ間隔（分）
        self.step_map = {
            'm1': 1,
            'm5': 5, 
            'm15': 15,
            'm30': 30,
            'h1': 60,
            'h4': 240
        }
            
        print(f"🔄 MultiTFWindowSampler初期化 ({split})")
        
        # キャッシュ機能
        self.cache_dir.mkdir(exist_ok=True)
        cache_hash = self._compute_cache_hash()
        cache_file = self.cache_dir / f"windows_{cache_hash}.npy"
        
        # 有効ウィンドウ範囲を計算（キャッシュまたはベクトル化版）
        start_time = time.time()
        if cache_file.exists():
            print(f"   📂 キャッシュからウィンドウ読み込み: {cache_file.name}")
            valid_indices = np.load(cache_file)
            self.valid_windows = self._indices_to_windows(valid_indices)
        else:
            print(f"   🔄 新規ウィンドウ計算中...")
            self.valid_windows = self._find_valid_windows()
            # キャッシュ保存
            valid_indices = self._windows_to_indices(self.valid_windows)
            np.save(cache_file, valid_indices)
            print(f"   💾 ウィンドウキャッシュ保存: {cache_file.name}")
        
        elapsed_time = time.time() - start_time
        print(f"   ⚡ ウィンドウ処理時間: {elapsed_time:.2f}秒")
        
        # 訓練/検証分割
        self.split_windows = self._split_windows()
        
        print(f"   総ウィンドウ数: {len(self.valid_windows)}")
        print(f"   {split}ウィンドウ数: {len(self.split_windows)}")
        
    def _find_valid_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """全TFでデータが十分存在する有効ウィンドウを発見（ベクトル化版）"""
        
        base_df = self.tf_data[self.base_tf]
        n_windows = len(base_df) - self.seq_len + 1
        
        if n_windows <= 0:
            return []
            
        print(f"   ベクトル化ウィンドウ探索開始: {n_windows:,} 候補")
        
        # 各TFのカバレッジをベクトル計算
        all_valid = np.ones(n_windows, dtype=bool)
        
        for tf_name, df in self.tf_data.items():
            # このTFの有効データマスクを作成
            tf_valid = self._compute_tf_coverage_vectorized(
                base_df, df, tf_name
            )
            all_valid &= tf_valid
            
            valid_count = np.sum(tf_valid)
            print(f"     {tf_name}: {valid_count:,}/{n_windows:,} ウィンドウが有効")
        
        # 有効なウィンドウのインデックスを取得
        valid_indices = np.where(all_valid)[0]
        
        # タイムスタンプペアに変換
        valid_windows = []
        for idx in valid_indices:
            start_time = base_df.index[idx]
            end_time = base_df.index[idx + self.seq_len - 1]
            valid_windows.append((start_time, end_time))
            
        final_count = len(valid_windows)
        print(f"   ベクトル化探索完了: {final_count:,} 有効ウィンドウ")
        
        # 性能統計
        if n_windows > 0:
            efficiency = (final_count / n_windows) * 100
            print(f"   データ効率: {efficiency:.1f}% ({final_count:,}/{n_windows:,})")
        
        return valid_windows
        
    def _compute_tf_coverage_vectorized(self, base_df: pd.DataFrame, tf_df: pd.DataFrame, tf_name: str) -> np.ndarray:
        """特定TFのカバレッジをベクトル計算"""
        
        n_windows = len(base_df) - self.seq_len + 1
        
        if tf_name == self.base_tf:
            # M1は全ウィンドウで有効（既にサイズ確認済み）
            return np.ones(n_windows, dtype=bool)
            
        # このTFに必要な最小データ点数を計算
        window_duration_min = self.seq_len - 1  # M1分単位での期間
        tf_intervals = {
            'm1': 1, 'm5': 5, 'm15': 15, 'm30': 30, 
            'h1': 60, 'h4': 240, 'd': 1440
        }
        interval = tf_intervals[tf_name]
        expected_len = max(1, int(window_duration_min / interval) + 1)
        min_required = int(expected_len * self.min_coverage)
        
        # ベクトル化で全ウィンドウのデータ数を一括計算
        try:
            # タイムゾーン統一 - 両方をtz-naiveに変換
            base_index = base_df.index
            tf_index = tf_df.index
            
            # tz-awareの場合はUTCに変換してからtz-naiveに
            if hasattr(base_index, 'tz') and base_index.tz is not None:
                base_index = base_index.tz_convert('UTC').tz_localize(None)
            if hasattr(tf_index, 'tz') and tf_index.tz is not None:
                tf_index = tf_index.tz_convert('UTC').tz_localize(None)
            
            # ウィンドウ開始時刻配列を作成
            start_times = base_index[:n_windows].values
            end_times = base_index[self.seq_len-1:self.seq_len-1+n_windows].values
            
            # ベクトル化された検索でデータ数を計算
            start_indices = tf_index.searchsorted(start_times, side='left')
            end_indices = tf_index.searchsorted(end_times, side='right')
            
            # 各ウィンドウのデータ数を計算
            data_counts = end_indices - start_indices
            
            # 闾値以上のウィンドウを特定
            valid_mask = data_counts >= min_required
            
        except Exception as e:
            # フォールバック: バッチ処理で計算
            print(f"     フォールバックモードで{tf_name}を処理: {str(e)}")
            valid_mask = np.zeros(n_windows, dtype=bool)
            
            batch_size = 10000
            for batch_start in range(0, n_windows, batch_size):
                batch_end = min(batch_start + batch_size, n_windows)
                
                for i in range(batch_start, batch_end):
                    start_time = base_df.index[i]
                    end_time = base_df.index[i + self.seq_len - 1]
                    
                    # タイムゾーン統一
                    if hasattr(start_time, 'tz') and start_time.tz is not None:
                        start_time = start_time.tz_convert('UTC').tz_localize(None)
                    if hasattr(end_time, 'tz') and end_time.tz is not None:
                        end_time = end_time.tz_convert('UTC').tz_localize(None)
                    
                    # tf_indexも統一
                    tf_index_safe = tf_df.index
                    if hasattr(tf_index_safe, 'tz') and tf_index_safe.tz is not None:
                        tf_index_safe = tf_index_safe.tz_convert('UTC').tz_localize(None)
                    
                    start_idx = tf_index_safe.searchsorted(start_time, side='left')
                    end_idx = tf_index_safe.searchsorted(end_time, side='right')
                    
                    data_count = end_idx - start_idx
                    valid_mask[i] = data_count >= min_required
                    
        return valid_mask
    
    def _check_coverage(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> bool:
        """指定期間での全TFデータカバレッジをチェック（旧実装・レガシー用）"""
        
        for tf_name, df in self.tf_data.items():
            # この期間のデータを取得
            window_data = df.loc[start_time:end_time]
            
            if tf_name == self.base_tf:
                # M1は正確に seq_len 必要
                expected_len = self.seq_len
            else:
                # 他のTFは期間に応じた期待長を計算
                expected_len = self._calculate_expected_length(tf_name, start_time, end_time)
                
            # カバレッジチェック
            if len(window_data) < expected_len * self.min_coverage:
                return False
                
        return True
        
    def _calculate_expected_length(self, tf_name: str, start_time: pd.Timestamp, end_time: pd.Timestamp) -> int:
        """TFと期間に基づく期待データ長を計算"""
        
        # 期間の長さ（分）
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # TFごとの間隔（分）
        tf_intervals = {
            'm1': 1,
            'm5': 5,
            'm15': 15,
            'm30': 30,
            'h1': 60,
            'h4': 240,
            'd': 1440
        }
        
        interval = tf_intervals[tf_name]
        expected_len = int(duration_minutes / interval) + 1
        
        return max(1, expected_len)  # 最低1データポイント
        
    def _split_windows(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """訓練/検証分割（時間的ギャップ付き）"""
        
        n_total = len(self.valid_windows)
        n_val = int(n_total * self.val_split)
        
        # val_gap_days を分単位に変換
        val_gap_minutes = int(self.val_gap_days * 24 * 60)
        
        # ベースTF（M1）の固定間隔を使用してギャップ計算
        base_step_minutes = self.step_map[self.base_tf]  # M1 = 1分
        gap_windows = int(val_gap_minutes / base_step_minutes)
        
        if n_val == 0:
            return self.valid_windows if self.split == "train" else []
            
        # ギャップを考慮した分割（修正版）
        if self.split == "train":
            print(f"   🕐 時間的ギャップ: {self.val_gap_days}日 = {val_gap_minutes}分 = {gap_windows}窓 (ベース間隔={base_step_minutes}分)")
            
            # 訓練: 最後の (n_val + gap_windows) を除外
            return self.valid_windows[:-(n_val + gap_windows)]
        else:  # val
            # 検証: 最後の n_val のみ使用（gapの後から）
            val_windows = self.valid_windows[-n_val:]
            
            # デバッグ出力：検証データの最初のタイムスタンプを表示
            if val_windows:
                first_val_ts = val_windows[0][0]  # (start_time, end_time)のstart_time
                print(f"   [DBG] 検証データ開始時刻: {first_val_ts}")
                print(f"   [DBG] 計算されたギャップ窓数: {gap_windows} (ベース間隔={base_step_minutes}分)")
                
                # 訓練データの最後のタイムスタンプも表示
                if n_val + gap_windows < len(self.valid_windows):
                    last_train_window = self.valid_windows[-(n_val + gap_windows) - 1]
                    last_train_ts = last_train_window[1]  # end_time
                    gap_actual = (first_val_ts - last_train_ts).total_seconds() / 86400  # 日数
                    print(f"   [DBG] 訓練データ終了時刻: {last_train_ts}")
                    print(f"   [DBG] 実際のギャップ: {gap_actual:.1f}日")
                else:
                    print(f"   [DBG] 警告: ギャップ計算で範囲外アクセス (n_val={n_val}, gap_windows={gap_windows}, total={len(self.valid_windows)})")
                    
            return val_windows
            
    def __len__(self) -> int:
        """サンプル数"""
        return len(self.split_windows)
        
    def __getitem__(self, idx: int) -> Dict[str, pd.DataFrame]:
        """
        指定インデックスのマルチTFウィンドウを取得
        
        Returns:
            Dict[tf_name, DataFrame]: 各TFのウィンドウデータ
        """
        if idx >= len(self.split_windows):
            raise IndexError(f"Index {idx} out of range (max: {len(self.split_windows)-1})")
            
        start_time, end_time = self.split_windows[idx]
        
        window_data = {}
        
        for tf_name, df in self.tf_data.items():
            # 期間データを取得
            tf_window = df.loc[start_time:end_time].copy()
            
            if tf_name == self.base_tf:
                # M1は正確に seq_len にリサンプル（必要に応じて）
                if len(tf_window) != self.seq_len:
                    # 時間インデックスで補間してseq_len長にする
                    tf_window = self._resample_to_length(tf_window, self.seq_len, start_time, end_time)
                    
            window_data[tf_name] = tf_window
            
        return window_data
        
    def _resample_to_length(
        self, 
        df: pd.DataFrame, 
        target_len: int, 
        start_time: pd.Timestamp, 
        end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """DataFrameを指定長にリサンプル"""
        
        # 等間隔時間インデックス作成
        time_index = pd.date_range(start=start_time, end=end_time, periods=target_len)
        
        # OHLCデータの適切なリサンプリング
        resampled = df.reindex(time_index, method='nearest')
        
        # 欠損値を前方埋め
        resampled = resampled.fillna(method='ffill').fillna(method='bfill')
        
        return resampled
        
    def get_sample_window_info(self, idx: int = 0) -> Dict:
        """サンプルウィンドウの情報を取得（デバッグ用）"""
        
        if len(self.split_windows) == 0:
            return {"error": "No valid windows"}
            
        start_time, end_time = self.split_windows[idx]
        window_data = self[idx]
        
        info = {
            "window_index": idx,
            "start_time": start_time,
            "end_time": end_time,
            "duration_hours": (end_time - start_time).total_seconds() / 3600,
            "tf_lengths": {tf: len(data) for tf, data in window_data.items()}
        }
        
        return info
    
    def _compute_cache_hash(self) -> str:
        """キャッシュ用ハッシュ計算"""
        # データの特性に基づくハッシュ生成
        base_df = self.tf_data[self.base_tf]
        hash_data = f"{len(base_df)}_{self.seq_len}_{self.min_coverage}"
        
        # データの開始・終了時刻も含める
        hash_data += f"_{base_df.index[0]}_{base_df.index[-1]}"
        
        return hashlib.md5(hash_data.encode()).hexdigest()[:8]
    
    def _windows_to_indices(self, windows: List[Tuple]) -> np.ndarray:
        """ウィンドウリストをインデックス配列に変換"""
        base_df = self.tf_data[self.base_tf]
        indices = []
        
        for start_time, end_time in windows:
            # 開始時刻のインデックスを取得
            start_idx = base_df.index.get_loc(start_time)
            indices.append(start_idx)
            
        return np.array(indices, dtype=np.int32)
    
    def _indices_to_windows(self, indices: np.ndarray) -> List[Tuple]:
        """インデックス配列をウィンドウリストに変換"""
        base_df = self.tf_data[self.base_tf]
        windows = []
        
        for idx in indices:
            start_time = base_df.index[idx]
            end_time = base_df.index[idx + self.seq_len - 1]
            windows.append((start_time, end_time))
            
        return windows