#!/usr/bin/env python3
"""
TF生成ロジックのユニットテスト
CI/CDで高速実行可能なテストケース
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

class TestTFGeneration:
    """TF生成ロジックのテストクラス"""
    
    def test_data_files_exist(self):
        """必要なデータファイルの存在確認"""
        data_dir = Path('data/derived')
        
        required_files = [
            'safe_ohlc_m5.parquet',
            'safe_ohlc_m15.parquet', 
            'safe_ohlc_h1.parquet',
            'safe_ohlc_d.parquet'
        ]
        
        for file_name in required_files:
            assert (data_dir / file_name).exists(), f"Missing: {file_name}"
    
    def test_volume_exclusion(self):
        """Volumeカラムの完全除外確認"""
        data_dir = Path('data/derived')
        
        for tf in ['m5', 'm15', 'h1', 'd']:
            df = pd.read_parquet(data_dir / f'safe_ohlc_{tf}.parquet')
            assert 'volume' not in df.columns, f"Volume found in {tf}"
            
    def test_ohlc_consistency(self):
        """OHLC論理整合性確認"""
        data_dir = Path('data/derived')
        
        for tf in ['m5', 'm15', 'h1']:
            df = pd.read_parquet(data_dir / f'safe_ohlc_{tf}.parquet')
            
            # OHLC論理チェック
            valid_ohlc = (
                (df['low'] <= df['high']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            )
            
            assert valid_ohlc.all(), f"OHLC violation in {tf}"
    
    def test_timestamp_ordering(self):
        """タイムスタンプソート確認"""
        data_dir = Path('data/derived')
        
        for tf in ['m5', 'h1']:
            df = pd.read_parquet(data_dir / f'safe_ohlc_{tf}.parquet')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            assert df['timestamp'].is_monotonic_increasing, f"Timestamp not sorted in {tf}"
    
    def test_data_range(self):
        """価格データ範囲確認（USD/JPY想定）"""
        data_dir = Path('data/derived')
        
        for tf in ['m5', 'h1']:
            df = pd.read_parquet(data_dir / f'safe_ohlc_{tf}.parquet')
            
            # USD/JPYの妥当な範囲
            for col in ['open', 'high', 'low', 'close']:
                assert df[col].min() >= 50, f"{tf} {col} too low"
                assert df[col].max() <= 200, f"{tf} {col} too high"
    
    def test_no_null_values(self):
        """NULL値なし確認"""
        data_dir = Path('data/derived')
        
        for tf in ['m5', 'h1']:
            df = pd.read_parquet(data_dir / f'safe_ohlc_{tf}.parquet')
            
            assert not df[['open', 'high', 'low', 'close']].isnull().any().any(), f"NULL values in {tf}"

if __name__ == "__main__":
    # 個別実行用
    import sys
    test_instance = TestTFGeneration()
    
    try:
        test_instance.test_data_files_exist()
        test_instance.test_volume_exclusion()
        test_instance.test_ohlc_consistency()
        test_instance.test_timestamp_ordering()
        test_instance.test_data_range()
        test_instance.test_no_null_values()
        
        print("✅ All tests passed!")
        sys.exit(0)
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
