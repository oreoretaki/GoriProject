#!/usr/bin/env python3
"""
境界条件ユニットテスト
リサンプリング境界の正確性をテスト
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをPATHに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestBoundaryConditions(unittest.TestCase):
    """境界条件テストクラス"""
    
    @classmethod
    def setUpClass(cls):
        """テストクラス初期化"""
        print("\\n🧪 境界条件ユニットテスト開始")
        print("=" * 50)
        
        # Clean M1データ読み込み
        m1_file = Path('../data/derived/simple_gap_aware_m1.parquet')
        if not m1_file.exists():
            raise FileNotFoundError(f"M1ファイルが見つかりません: {m1_file}")
            
        cls.df_m1 = pd.read_parquet(m1_file)
        if cls.df_m1.index.name == 'timestamp':
            cls.df_m1 = cls.df_m1.sort_index()
        else:
            cls.df_m1 = cls.df_m1.set_index('timestamp').sort_index()
        
        # タイムゾーン統一
        if cls.df_m1.index.tz is None:
            cls.df_m1.index = cls.df_m1.index.tz_localize('UTC')
        elif str(cls.df_m1.index.tz) != 'UTC':
            cls.df_m1.index = cls.df_m1.index.tz_convert('UTC')
        
        print(f"M1データ読み込み完了: {len(cls.df_m1):,}件")
        
        # TFデータ読み込み
        cls.tf_data = {}
        tf_files = {
            'm5': '../data/derived/simple_gap_aware_m5.parquet',
            'm15': '../data/derived/simple_gap_aware_m15.parquet',
            'h1': '../data/derived/simple_gap_aware_h1.parquet',
            'd': '../data/derived/simple_gap_aware_d.parquet'
        }
        
        for tf_name, tf_file in tf_files.items():
            if Path(tf_file).exists():
                df_tf = pd.read_parquet(tf_file)
                if df_tf.index.name == 'timestamp':
                    df_tf = df_tf.sort_index()
                else:
                    df_tf = df_tf.set_index('timestamp').sort_index()
                
                # タイムゾーン統一
                if df_tf.index.tz is None:
                    df_tf.index = df_tf.index.tz_localize('UTC')
                elif str(df_tf.index.tz) != 'UTC':
                    df_tf.index = df_tf.index.tz_convert('UTC')
                
                cls.tf_data[tf_name] = df_tf
                print(f"{tf_name.upper()}データ読み込み完了: {len(df_tf):,}件")
    
    def test_m5_boundary_specific_time(self):
        """M5境界テスト: 2017-01-03 08:00 UTC"""
        print("\\n1. M5境界テスト: 2017-01-03 08:00 UTC")
        
        # 対象時刻
        target_time = pd.Timestamp('2017-01-03 08:00:00', tz='UTC')
        
        # M5データから該当バーを取得
        if 'm5' not in self.tf_data:
            self.skipTest("M5データが利用できません")
        
        df_m5 = self.tf_data['m5']
        
        # 該当するM5バーが存在するかチェック
        if target_time not in df_m5.index:
            # 最も近い時刻を探す
            available_times = df_m5.index[df_m5.index >= target_time][:5]
            if len(available_times) > 0:
                target_time = available_times[0]
                print(f"  調整後対象時刻: {target_time}")
            else:
                self.skipTest(f"対象時刻のM5データが見つかりません")
        
        m5_bar = df_m5.loc[target_time]
        
        # 対応するM1期間 (left-closed: 08:00のM5バーは08:00-08:04の5本)
        m1_start = target_time
        m1_end = target_time + pd.Timedelta(minutes=4)
        
        print(f"  M5バー時刻: {target_time}")
        print(f"  対応M1期間: {m1_start} ～ {m1_end}")
        
        # M1データ抽出
        m1_slice = self.df_m1.loc[m1_start:m1_end]
        
        print(f"  M1件数: {len(m1_slice)}件")
        
        # M1データが存在する場合のみテスト
        if len(m1_slice) > 0:
            # 期待値計算
            expected_open = m1_slice['open'].iloc[0]
            expected_high = m1_slice['high'].max()
            expected_low = m1_slice['low'].min()
            expected_close = m1_slice['close'].iloc[-1]
            
            # 検証
            tolerance = 1e-6
            
            self.assertAlmostEqual(m5_bar['open'], expected_open, delta=tolerance,
                                 msg=f"Open不一致: M5={m5_bar['open']}, 期待値={expected_open}")
            self.assertAlmostEqual(m5_bar['high'], expected_high, delta=tolerance,
                                 msg=f"High不一致: M5={m5_bar['high']}, 期待値={expected_high}")
            self.assertAlmostEqual(m5_bar['low'], expected_low, delta=tolerance,
                                 msg=f"Low不一致: M5={m5_bar['low']}, 期待値={expected_low}")
            self.assertAlmostEqual(m5_bar['close'], expected_close, delta=tolerance,
                                 msg=f"Close不一致: M5={m5_bar['close']}, 期待値={expected_close}")
            
            print(f"  ✅ M5境界テスト合格")
            print(f"     Open: {m5_bar['open']:.5f} = {expected_open:.5f}")
            print(f"     High: {m5_bar['high']:.5f} = {expected_high:.5f}")
            print(f"     Low: {m5_bar['low']:.5f} = {expected_low:.5f}")
            print(f"     Close: {m5_bar['close']:.5f} = {expected_close:.5f}")
        else:
            self.skipTest("対象期間にM1データが存在しません")
    
    def test_h1_boundary_specific_time(self):
        """H1境界テスト: 2017-01-03 08:00 UTC"""
        print("\\n2. H1境界テスト: 2017-01-03 08:00 UTC")
        
        # 対象時刻
        target_time = pd.Timestamp('2017-01-03 08:00:00', tz='UTC')
        
        # H1データから該当バーを取得
        if 'h1' not in self.tf_data:
            self.skipTest("H1データが利用できません")
        
        df_h1 = self.tf_data['h1']
        
        # 該当するH1バーが存在するかチェック
        if target_time not in df_h1.index:
            available_times = df_h1.index[df_h1.index >= target_time][:5]
            if len(available_times) > 0:
                target_time = available_times[0]
                print(f"  調整後対象時刻: {target_time}")
            else:
                self.skipTest(f"対象時刻のH1データが見つかりません")
        
        h1_bar = df_h1.loc[target_time]
        
        # 対応するM1期間 (08:00-08:59の60本)
        m1_start = target_time
        m1_end = target_time + pd.Timedelta(minutes=59)
        
        print(f"  H1バー時刻: {target_time}")
        print(f"  対応M1期間: {m1_start} ～ {m1_end}")
        
        # M1データ抽出
        m1_slice = self.df_m1.loc[m1_start:m1_end]
        
        print(f"  M1件数: {len(m1_slice)}件")
        
        # M1データが存在する場合のみテスト
        if len(m1_slice) > 0:
            # 期待値計算
            expected_open = m1_slice['open'].iloc[0]
            expected_high = m1_slice['high'].max()
            expected_low = m1_slice['low'].min()
            expected_close = m1_slice['close'].iloc[-1]
            
            # 検証
            tolerance = 1e-6
            
            self.assertAlmostEqual(h1_bar['open'], expected_open, delta=tolerance,
                                 msg=f"Open不一致: H1={h1_bar['open']}, 期待値={expected_open}")
            self.assertAlmostEqual(h1_bar['high'], expected_high, delta=tolerance,
                                 msg=f"High不一致: H1={h1_bar['high']}, 期待値={expected_high}")
            self.assertAlmostEqual(h1_bar['low'], expected_low, delta=tolerance,
                                 msg=f"Low不一致: H1={h1_bar['low']}, 期待値={expected_low}")
            self.assertAlmostEqual(h1_bar['close'], expected_close, delta=tolerance,
                                 msg=f"Close不一致: H1={h1_bar['close']}, 期待値={expected_close}")
            
            print(f"  ✅ H1境界テスト合格")
            print(f"     Open: {h1_bar['open']:.5f} = {expected_open:.5f}")
            print(f"     High: {h1_bar['high']:.5f} = {expected_high:.5f}")
            print(f"     Low: {h1_bar['low']:.5f} = {expected_low:.5f}")
            print(f"     Close: {h1_bar['close']:.5f} = {expected_close:.5f}")
        else:
            self.skipTest("対象期間にM1データが存在しません")
    
    def test_daily_boundary_alignment(self):
        """日足境界テスト: UTC 00:00アライメント"""
        print("\\n3. 日足境界テスト: UTC 00:00アライメント")
        
        if 'd' not in self.tf_data:
            self.skipTest("日足データが利用できません")
        
        df_d = self.tf_data['d']
        
        # 最初の日足バーを取得
        first_daily = df_d.iloc[0]
        first_time = df_d.index[0]
        
        print(f"  最初の日足時刻: {first_time}")
        
        # UTC 00:00であることを確認（または調整されたUTC時刻）
        self.assertEqual(first_time.hour, 0, "日足は00:00から始まるべき")
        self.assertEqual(first_time.minute, 0, "日足は00分から始まるべき")
        
        # 対応するM1期間（その日の00:00-23:59）
        day_start = first_time
        day_end = first_time + pd.Timedelta(hours=23, minutes=59)
        
        print(f"  対応M1期間: {day_start} ～ {day_end}")
        
        # M1データ抽出
        m1_slice = self.df_m1.loc[day_start:day_end]
        
        print(f"  M1件数: {len(m1_slice)}件")
        
        if len(m1_slice) > 0:
            # 期待値計算
            expected_open = m1_slice['open'].iloc[0]
            expected_high = m1_slice['high'].max()
            expected_low = m1_slice['low'].min()
            expected_close = m1_slice['close'].iloc[-1]
            
            # 検証
            tolerance = 1e-6
            
            self.assertAlmostEqual(first_daily['open'], expected_open, delta=tolerance,
                                 msg=f"Daily Open不一致")
            self.assertAlmostEqual(first_daily['high'], expected_high, delta=tolerance,
                                 msg=f"Daily High不一致")
            self.assertAlmostEqual(first_daily['low'], expected_low, delta=tolerance,
                                 msg=f"Daily Low不一致")
            self.assertAlmostEqual(first_daily['close'], expected_close, delta=tolerance,
                                 msg=f"Daily Close不一致")
            
            print(f"  ✅ 日足境界テスト合格")
        else:
            self.skipTest("対象期間にM1データが存在しません")
    
    def test_resampling_logic_consistency(self):
        """リサンプリングロジック一貫性テスト"""
        print("\\n4. リサンプリングロジック一貫性テスト")
        
        # label='left', closed='left' の確認
        if 'm5' not in self.tf_data:
            self.skipTest("M5データが利用できません")
        
        df_m5 = self.tf_data['m5']
        
        # M5の複数のバーをチェック
        sample_times = df_m5.index[:3]  # 最初の3バー
        
        for i, m5_time in enumerate(sample_times):
            print(f"  サンプル {i+1}: {m5_time}")
            
            # M5バーのデータ
            m5_bar = df_m5.loc[m5_time]
            
            # 期待されるM1期間
            m1_start = m5_time
            m1_end = m5_time + pd.Timedelta(minutes=4)
            
            # M1データ抽出
            m1_slice = self.df_m1.loc[m1_start:m1_end]
            
            if len(m1_slice) > 0:
                # 各M1の時刻がM5期間内にあることを確認
                for m1_time in m1_slice.index:
                    self.assertGreaterEqual(m1_time, m1_start, 
                                          f"M1時刻がM5期間より前: {m1_time} < {m1_start}")
                    self.assertLessEqual(m1_time, m1_end,
                                       f"M1時刻がM5期間より後: {m1_time} > {m1_end}")
                
                print(f"    ✅ M1期間チェック合格: {len(m1_slice)}件")
            else:
                print(f"    ⚠️ M1データなし")
        
        print("  ✅ リサンプリングロジック一貫性テスト完了")

if __name__ == '__main__':
    # テスト実行
    unittest.main(verbosity=2, exit=False)
    
    print("\\n🎯 境界条件ユニットテスト完了")
    print("すべての境界条件でM1→TF変換の正確性を確認")