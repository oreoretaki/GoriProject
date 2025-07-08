#!/usr/bin/env python3
"""
全件TF整合性検証システム
M1→全TFの完全ベクトル化チェック
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

class FullConsistencyValidator:
    def __init__(self):
        self.tf_files = {
            'm5': 'data/derived/simple_gap_aware_m5.parquet',
            'm15': 'data/derived/simple_gap_aware_m15.parquet', 
            'm30': 'data/derived/simple_gap_aware_m30.parquet',
            'h1': 'data/derived/simple_gap_aware_h1.parquet',
            'h4': 'data/derived/simple_gap_aware_h4.parquet',
            'd': 'data/derived/simple_gap_aware_d.parquet'
        }
        
    def load_m1_source(self):
        """M1ソースデータ読み込み"""
        print("M1ソースデータ読み込み...")
        
        conn = sqlite3.connect('data/oanda_historical.db')
        query = """
        SELECT timestamp, open, high, low, close 
        FROM candles_m1 
        WHERE instrument = 'USD_JPY'
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        print(f"  M1データ: {len(df):,}件")
        return df
        
    def vectorized_consistency_check(self, df_m1, tf_name):
        """ベクトル化された全件整合性チェック"""
        print(f"\\n{tf_name.upper()}全件整合性チェック...")
        
        tf_file = self.tf_files[tf_name]
        
        if not Path(tf_file).exists():
            print(f"  ⚠️ ファイルなし: {tf_file}")
            return {'error': 'file_not_found'}
            
        # TFデータ読み込み
        df_tf = pd.read_parquet(tf_file)
        df_tf['timestamp'] = pd.to_datetime(df_tf['timestamp'])
        df_tf = df_tf.set_index('timestamp').sort_index()
        
        print(f"  TFデータ: {len(df_tf):,}件")
        
        if len(df_tf) == 0:
            return {'error': 'empty_data'}
        
        # 期間設定
        if tf_name == 'd':
            freq_minutes = 1440
            resample_rule = '1D'
        elif tf_name == 'h4':
            freq_minutes = 240
            resample_rule = '4H'
        elif tf_name == 'h1':
            freq_minutes = 60
            resample_rule = '1H'
        else:
            freq_minutes = int(tf_name[1:])
            resample_rule = f'{freq_minutes}T'
        
        # M1から期待値を再計算（ベクトル化）
        print(f"  M1→{tf_name.upper()}期待値計算中...")
        
        expected_tf = (df_m1
                      .resample(resample_rule, label='left', closed='left')
                      .agg({
                          'open': 'first',
                          'high': 'max', 
                          'low': 'min',
                          'close': 'last'
                      })
                      .dropna())
        
        print(f"  期待値: {len(expected_tf):,}件")
        
        # 共通期間での比較
        common_index = df_tf.index.intersection(expected_tf.index)
        print(f"  共通期間: {len(common_index):,}件")
        
        if len(common_index) == 0:
            return {'error': 'no_common_periods'}
        
        # 実際値と期待値の抽出
        actual = df_tf.loc[common_index]
        expected = expected_tf.loc[common_index]
        
        # ベクトル化された一致チェック
        tolerance = 1e-6
        
        open_matches = np.abs(actual['open'] - expected['open']) < tolerance
        high_matches = np.abs(actual['high'] - expected['high']) < tolerance
        low_matches = np.abs(actual['low'] - expected['low']) < tolerance
        close_matches = np.abs(actual['close'] - expected['close']) < tolerance
        
        # 全OHLC一致
        all_ohlc_matches = open_matches & high_matches & low_matches & close_matches
        
        # 統計計算
        results = {
            'total_periods': len(common_index),
            'open_matches': int(open_matches.sum()),
            'high_matches': int(high_matches.sum()),
            'low_matches': int(low_matches.sum()),
            'close_matches': int(close_matches.sum()),
            'all_ohlc_matches': int(all_ohlc_matches.sum()),
            'consistency_rates': {
                'open': float(open_matches.mean()),
                'high': float(high_matches.mean()),
                'low': float(low_matches.mean()),
                'close': float(close_matches.mean()),
                'overall': float(all_ohlc_matches.mean())
            }
        }
        
        # 不一致サンプル
        mismatch_mask = ~all_ohlc_matches
        if mismatch_mask.any():
            mismatch_timestamps = actual[mismatch_mask].index[:5]
            results['mismatch_sample'] = [str(ts) for ts in mismatch_timestamps]
        else:
            results['mismatch_sample'] = []
        
        print(f"  Open一致: {results['consistency_rates']['open']:.1%}")
        print(f"  High一致: {results['consistency_rates']['high']:.1%}")
        print(f"  Low一致: {results['consistency_rates']['low']:.1%}")
        print(f"  Close一致: {results['consistency_rates']['close']:.1%}")
        print(f"  総合一致: {results['consistency_rates']['overall']:.1%}")
        
        if results['consistency_rates']['overall'] < 0.99:
            print(f"  ⚠️ 整合性基準未達成 (<99%)")
            
        return results
        
    def generate_data_hashes(self):
        """データハッシュ生成"""
        print("\\nデータハッシュ生成中...")
        
        hashes = {}
        
        for tf_name, file_path in self.tf_files.items():
            if Path(file_path).exists():
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    file_size = Path(file_path).stat().st_size
                    
                hashes[f'{tf_name}_hash'] = file_hash
                hashes[f'{tf_name}_size'] = file_size
                
                print(f"  {tf_name.upper()}: {file_hash[:8]}... ({file_size:,} bytes)")
        
        return hashes
        
    def run_full_validation(self):
        """完全検証実行"""
        print("🔍 Stage 0 完全整合性検証")
        print("=" * 50)
        
        # M1データ読み込み
        df_m1 = self.load_m1_source()
        
        # 各TFの整合性チェック
        validation_results = {}
        
        for tf_name in self.tf_files.keys():
            result = self.vectorized_consistency_check(df_m1, tf_name)
            validation_results[tf_name] = result
        
        # データハッシュ生成
        data_hashes = self.generate_data_hashes()
        
        # 総合レポート作成
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'validation_version': '2.0.0',
            'tf_consistency_results': validation_results,
            'data_integrity_hashes': data_hashes,
            'stage0_ready': self.assess_stage0_readiness(validation_results)
        }
        
        # レポート保存
        report_file = Path('data/derived/full_consistency_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n📋 完全整合性レポート保存: {report_file}")
        
        return report
        
    def assess_stage0_readiness(self, validation_results):
        """Stage 0準備状況評価"""
        print("\\n🎯 Stage 0準備状況評価...")
        
        ready_criteria = {
            'all_tf_generated': True,
            'consistency_above_99pct': True,
            'no_critical_errors': True
        }
        
        for tf_name, result in validation_results.items():
            if 'error' in result:
                ready_criteria['all_tf_generated'] = False
                ready_criteria['no_critical_errors'] = False
                print(f"  ❌ {tf_name.upper()}: エラー - {result['error']}")
            elif result.get('consistency_rates', {}).get('overall', 0) < 0.99:
                ready_criteria['consistency_above_99pct'] = False
                print(f"  ⚠️ {tf_name.upper()}: 整合性不足 - {result['consistency_rates']['overall']:.1%}")
            else:
                print(f"  ✅ {tf_name.upper()}: OK - {result['consistency_rates']['overall']:.1%}")
        
        stage0_ready = all(ready_criteria.values())
        
        print(f"\\n🚀 Stage 0ステータス: {'✅ READY' if stage0_ready else '❌ NOT READY'}")
        
        return {
            'ready': stage0_ready,
            'criteria': ready_criteria,
            'assessment_time': pd.Timestamp.now().isoformat()
        }

def main():
    validator = FullConsistencyValidator()
    report = validator.run_full_validation()
    
    # サマリー表示
    print("\\n📊 検証結果サマリー:")
    for tf_name, result in report['tf_consistency_results'].items():
        if 'consistency_rates' in result:
            overall = result['consistency_rates']['overall']
            print(f"  {tf_name.upper()}: {overall:.1%} 整合性")
        else:
            print(f"  {tf_name.upper()}: エラー")

if __name__ == "__main__":
    main()