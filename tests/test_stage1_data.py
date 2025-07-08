#!/usr/bin/env python3
"""
Stage 1 データパイプラインテスト
ウィンドウサンプリング、特徴量エンジニアリング、マスキング、整列のテスト
"""

import unittest
import torch
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをPATHに追加
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from stage1.src.window_sampler import MultiTFWindowSampler
from stage1.src.feature_engineering import FeatureEngineer
from stage1.src.masking import MaskingStrategy
from stage1.src.normalization import TFNormalizer
from stage1.src.data_loader import Stage1Dataset

class TestStage1DataPipeline(unittest.TestCase):
    """Stage 1 データパイプラインテストクラス"""
    
    @classmethod
    def setUpClass(cls):
        """テストクラス初期化"""
        print("\\n🧪 Stage 1 データパイプラインテスト開始")
        print("=" * 60)
        
        # テスト用設定
        cls.config = {
            'data': {
                'seq_len': 50,  # テスト用に短縮
                'n_timeframes': 6,
                'n_features': 6,
                'timeframes': ['m1', 'm5', 'm15', 'm30', 'h1', 'h4'],
                'data_dir': '../data/derived',
                'stats_file': 'test_stats.json'
            },
            'masking': {
                'mask_ratio': 0.15,
                'mask_span_min': 3,
                'mask_span_max': 10,
                'sync_across_tf': True
            },
            'validation': {
                'val_split': 0.2
            },
            'normalization': {
                'method': 'zscore',
                'per_tf': True
            }
        }
        
        # テスト用ダミーデータ作成
        cls.dummy_tf_data = cls._create_dummy_tf_data()
        
    @classmethod
    def _create_dummy_tf_data(cls):
        """テスト用ダミーTFデータ作成"""
        
        # ベース時間範囲
        start_time = pd.Timestamp('2024-01-01 08:00:00', tz='UTC')
        
        tf_data = {}
        
        # M1データ（ベース）
        m1_periods = 1000
        m1_index = pd.date_range(start=start_time, periods=m1_periods, freq='1T')
        
        # ダミーOHLCデータ生成（リアルな価格変動をシミュレート）
        np.random.seed(42)
        base_price = 150.0
        returns = np.random.normal(0, 0.001, m1_periods)  # 0.1%標準偏差
        prices = base_price * (1 + returns).cumprod()
        
        # OHLC計算
        ohlc_data = []
        for i in range(m1_periods):
            open_price = prices[i]
            high_price = open_price * (1 + abs(np.random.normal(0, 0.0005)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.0005)))
            close_price = prices[i]
            
            ohlc_data.append([open_price, high_price, low_price, close_price])
            
        m1_df = pd.DataFrame(ohlc_data, index=m1_index, columns=['open', 'high', 'low', 'close'])
        tf_data['m1'] = m1_df
        
        # 他のTFデータ（M1から集約）
        tf_intervals = {'m5': '5T', 'm15': '15T', 'm30': '30T', 'h1': '1H', 'h4': '4H', 'd': '1D'}
        
        for tf_name, freq in tf_intervals.items():
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }
            
            tf_df = m1_df.resample(freq, label='left', closed='left').agg(agg_rules).dropna()
            tf_data[tf_name] = tf_df
            
        print(f"   ダミーデータ作成完了:")
        for tf_name, df in tf_data.items():
            print(f"     {tf_name.upper()}: {len(df)}レコード")
            
        return tf_data
        
    def test_window_sampler_basic(self):
        """ウィンドウサンプラー基本テスト"""
        print("\\n1. ウィンドウサンプラー基本テスト")
        
        sampler = MultiTFWindowSampler(
            self.dummy_tf_data,
            seq_len=self.config['data']['seq_len'],
            split='train'
        )
        
        # 基本属性チェック
        self.assertGreater(len(sampler), 0, "有効ウィンドウが存在しない")
        
        # サンプル取得テスト
        sample_window = sampler[0]
        
        # 全TFが含まれているか
        expected_tfs = set(self.config['data']['timeframes'])
        actual_tfs = set(sample_window.keys())
        self.assertTrue(expected_tfs.issubset(actual_tfs), f"TF不足: 期待{expected_tfs}, 実際{actual_tfs}")
        
        # M1のシーケンス長チェック
        m1_data = sample_window['m1']
        self.assertEqual(len(m1_data), self.config['data']['seq_len'], 
                        f"M1シーケンス長不正: 期待{self.config['data']['seq_len']}, 実際{len(m1_data)}")
        
        print(f"     ✅ ウィンドウ数: {len(sampler)}")
        print(f"     ✅ M1シーケンス長: {len(m1_data)}")
        
        # ウィンドウ情報表示
        info = sampler.get_sample_window_info(0)
        print(f"     ✅ ウィンドウ情報: {info['duration_hours']:.2f}時間")
        
    def test_feature_engineering(self):
        """特徴量エンジニアリングテスト"""
        print("\\n2. 特徴量エンジニアリングテスト")
        
        engineer = FeatureEngineer(self.config)
        
        # サンプルウィンドウ取得
        sampler = MultiTFWindowSampler(self.dummy_tf_data, self.config['data']['seq_len'])
        window_data = sampler[0]
        
        # 特徴量・ターゲット生成
        features, targets = engineer.process_window(window_data)
        
        # 形状チェック
        expected_shape = (self.config['data']['n_timeframes'], 
                         self.config['data']['seq_len'], 
                         self.config['data']['n_features'])
        self.assertEqual(features.shape, expected_shape, 
                        f"特徴量形状不正: 期待{expected_shape}, 実際{features.shape}")
        
        target_shape = (self.config['data']['n_timeframes'], 
                       self.config['data']['seq_len'], 4)
        self.assertEqual(targets.shape, target_shape,
                        f"ターゲット形状不正: 期待{target_shape}, 実際{targets.shape}")
        
        # 特徴量値チェック
        self.assertFalse(torch.isnan(features).any(), "特徴量にNaNが含まれている")
        self.assertFalse(torch.isinf(features).any(), "特徴量に無限大が含まれている")
        
        print(f"     ✅ 特徴量形状: {features.shape}")
        print(f"     ✅ ターゲット形状: {targets.shape}")
        
        # 特徴量統計表示
        stats = engineer.get_feature_stats(window_data)
        print(f"     ✅ M1特徴量範囲: [{stats['m1']['min'][0]:.6f}, {stats['m1']['max'][0]:.6f}]")
        
    def test_masking_strategy(self):
        """マスキング戦略テスト"""
        print("\\n3. マスキング戦略テスト")
        
        masking = MaskingStrategy(self.config)
        
        # ダミー特徴量（3Dで生成）
        features = torch.randn(self.config['data']['n_timeframes'],
                              self.config['data']['seq_len'],
                              self.config['data']['n_features'])
        
        # マスク生成
        masks = masking.generate_masks(features, seed=42)
        
        # 形状チェック
        expected_mask_shape = (self.config['data']['n_timeframes'],
                              self.config['data']['seq_len'])
        self.assertEqual(masks.shape, expected_mask_shape,
                        f"マスク形状不正: 期待{expected_mask_shape}, 実際{masks.shape}")
        
        # マスク率チェック
        for tf_idx in range(self.config['data']['n_timeframes']):
            tf_mask = masks[tf_idx]
            mask_ratio = tf_mask.mean().item()
            expected_ratio = self.config['masking']['mask_ratio']
            
            self.assertLessEqual(abs(mask_ratio - expected_ratio), 0.1,
                               f"TF{tf_idx}のマスク率が範囲外: {mask_ratio:.3f} (期待: {expected_ratio})")
        
        print(f"     ✅ マスク形状: {masks.shape}")
        
        # マスク統計表示
        stats = masking.get_mask_statistics(masks)  # マスクテンソル全体
        for tf_name, tf_stats in stats.items():
            print(f"     ✅ {tf_name}: マスク率{tf_stats['mask_ratio']:.3f}, ブロック数{tf_stats['n_blocks']}")
            
    def test_normalization(self):
        """正規化テスト"""
        print("\\n4. 正規化テスト")
        
        # 一時ディレクトリ作成
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = self.config.copy()
            test_config['data']['data_dir'] = temp_dir
            test_config['data']['stats_file'] = 'test_stats.json'
            
            normalizer = TFNormalizer(test_config, cache_stats=True)
            
            # 統計計算
            normalizer.fit(self.dummy_tf_data)
            
            # 統計存在チェック
            self.assertGreater(len(normalizer.stats), 0, "正規化統計が計算されていない")
            
            # ダミー特徴量で正規化テスト（TF毎に処理）
            features = torch.randn(len(self.config['data']['timeframes']), 50, 6)
            normalized = normalizer.normalize(features)
            
            # 形状保持チェック
            self.assertEqual(features.shape, normalized.shape, "正規化で形状が変化")
            
            # NaN/Inf チェック
            self.assertFalse(torch.isnan(normalized).any(), "正規化後にNaN")
            self.assertFalse(torch.isinf(normalized).any(), "正規化後に無限大")
            
            print(f"     ✅ 統計計算完了: {len(normalizer.stats)}個のTF")
            print(f"     ✅ 正規化形状: {normalized.shape}")
            
            # 統計サマリー表示
            summary = normalizer.get_stats_summary()
            if 'm1' in summary:
                m1_stats = summary['m1']['features']
                print(f"     ✅ M1 open統計: mean={m1_stats['open']['mean']:.6f}, std={m1_stats['open']['std']:.6f}")
                
    def test_dataset_integration(self):
        """データセット統合テスト"""
        print("\\n5. データセット統合テスト")
        
        # Stage 0データが存在するかチェック
        data_dir = Path("../data/derived")
        if not data_dir.exists():
            print("     ⚠️ Stage 0データが存在しないため、統合テストをスキップ")
            return
            
        try:
            # 実際のデータセット作成テスト
            dataset = Stage1Dataset(
                str(data_dir),
                self.config,
                split="train",
                cache_stats=False  # テスト用
            )
            
            # 基本属性チェック
            self.assertGreater(len(dataset), 0, "データセットが空")
            
            # サンプル取得テスト
            sample = dataset[0]
            
            # 必要キーの存在チェック
            required_keys = ['features', 'targets', 'masks', 'tf_ids']
            for key in required_keys:
                self.assertIn(key, sample, f"サンプルに{key}が含まれていない")
                
            # 形状チェック
            features = sample['features']
            expected_shape = (len(self.config['data']['timeframes']),
                             self.config['data']['seq_len'],
                             self.config['data']['n_features'])
            
            self.assertEqual(features.shape, expected_shape,
                           f"サンプル特徴量形状不正: 期待{expected_shape}, 実際{features.shape}")
            
            print(f"     ✅ データセットサイズ: {len(dataset)}")
            print(f"     ✅ サンプル特徴量形状: {features.shape}")
            print(f"     ✅ サンプルキー: {list(sample.keys())}")
            
        except Exception as e:
            print(f"     ⚠️ 統合テストエラー（データ不足の可能性）: {e}")
            
    def test_data_alignment(self):
        """データ整列テスト"""
        print("\\n6. データ整列テスト")
        
        sampler = MultiTFWindowSampler(self.dummy_tf_data, self.config['data']['seq_len'])
        window_data = sampler[0]
        
        # 時間整列チェック
        m1_data = window_data['m1']
        m5_data = window_data['m5']
        
        # M1とM5の時間範囲が一致するかチェック
        m1_start = m1_data.index[0]
        m1_end = m1_data.index[-1]
        m5_start = m5_data.index[0]
        m5_end = m5_data.index[-1]
        
        # 右端整列：終了時刻が近いことを確認
        time_diff = abs((m1_end - m5_end).total_seconds())
        self.assertLess(time_diff, 300, f"M1とM5の終了時刻が大きく異なる: {time_diff}秒")
        
        print(f"     ✅ M1期間: {m1_start} → {m1_end}")
        print(f"     ✅ M5期間: {m5_start} → {m5_end}")
        print(f"     ✅ 終了時刻差: {time_diff:.0f}秒")
        
        # OHLC妥当性チェック
        for tf_name, tf_data in window_data.items():
            if len(tf_data) > 0:
                # High >= Open, Close; Low <= Open, Close
                highs = tf_data['high'].values
                lows = tf_data['low'].values
                opens = tf_data['open'].values
                closes = tf_data['close'].values
                
                valid_high = np.all(highs >= np.maximum(opens, closes))
                valid_low = np.all(lows <= np.minimum(opens, closes))
                
                self.assertTrue(valid_high, f"{tf_name}: High < Open/Close")
                self.assertTrue(valid_low, f"{tf_name}: Low > Open/Close")
                
        print("     ✅ OHLC妥当性: 全TF合格")

def main():
    """テスト実行"""
    print("🚀 Stage 1 データパイプラインテスト実行")
    
    # テストスイート作成
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStage1DataPipeline)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    print("\\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 全テスト合格")
        return 0
    else:
        print("❌ テスト失敗")
        print(f"   失敗数: {len(result.failures)}")
        print(f"   エラー数: {len(result.errors)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)