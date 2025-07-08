#!/usr/bin/env python3
"""
Stage 0 ETL本体: M1→全TF生成スクリプト
唯一のデータ生成ソース - gap補完ポリシー・TZ処理を含む
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class Stage0TFGenerator:
    """Stage 0 TF生成器"""
    
    def __init__(self, db_path='../data/oanda_historical.db', output_dir='../data/derived'):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # TF設定: label='left', closed='left' でリサンプリング
        # 例: M5の08:00バー = 08:00-08:04の5本M1を集約
        self.tf_configs = {
            'm5': {'rule': '5T', 'description': '5分足'},
            'm15': {'rule': '15T', 'description': '15分足'},
            'm30': {'rule': '30T', 'description': '30分足'},
            'h1': {'rule': '1H', 'description': '1時間足'},
            'h4': {'rule': '4H', 'description': '4時間足'},
            'd': {'rule': '1D', 'description': '日足'}
        }
    
    def load_m1_from_sqlite(self):
        """SQLiteからM1データ読み込み（Volume完全除外）"""
        print("1. M1データ読み込み（SQLite → Memory）...")
        
        # Volume除外でOHLC限定読み込み
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT timestamp, open, high, low, close 
        FROM candles_m1 
        WHERE instrument = 'USD_JPY'
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # タイムスタンプ処理: naive → UTC変換
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # タイムゾーン統一: UTC固定（金融データの標準）
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        elif str(df.index.tz) != 'UTC':
            df.index = df.index.tz_convert('UTC')
        
        # gap_flag追加: 現在は全てFalse（実データのみ）
        # 将来的なギャップ補完対応のための予約フィールド
        df['gap_flag'] = False
        
        print(f"   レコード数: {len(df):,}件")
        print(f"   期間: {df.index.min()} ～ {df.index.max()}")
        print(f"   タイムゾーン: {df.index.tz}")
        print(f"   gap_flag: {df['gap_flag'].sum()}件（全て実データ）")
        
        return df
    
    def save_clean_m1(self, df_m1):
        """Clean M1 Parquet保存（唯一のソース真実）"""
        print("\\n2. Clean M1保存（唯一のソース真実）...")
        
        output_file = self.output_dir / 'simple_gap_aware_m1.parquet'
        
        # インデックス保持でParquet保存（Snappy圧縮）
        df_m1.to_parquet(output_file, compression='snappy', index=True)
        
        file_size = output_file.stat().st_size
        print(f"   保存完了: {output_file}")
        print(f"   ファイルサイズ: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"   圧縮形式: Parquet + Snappy")
        
        return output_file
    
    def generate_tf_from_m1(self, df_m1, tf_name, rule):
        """M1からTF生成（Gap-aware処理）"""
        
        # ギャップ耐性リサンプリング
        # label='left': 08:00バーは08:00-08:04を表す
        # closed='left': 開始時刻を含み、終了時刻を含まない
        agg_rules = {
            'open': 'first',   # 期間開始価格
            'high': 'max',     # 期間最高価格  
            'low': 'min',      # 期間最安価格
            'close': 'last'    # 期間終了価格
            # gap_flagは除外: TFレベルではギャップ情報を保持しない
        }
        
        # リサンプリング実行
        df_tf = (df_m1
                .drop(columns=['gap_flag'])  # gap_flagを除外
                .resample(rule, label='left', closed='left')
                .agg(agg_rules)
                .dropna())  # 不完全なバーを除去
        
        # OHLC論理チェック: データ品質保証
        invalid_ohlc = (
            (df_tf['low'] > df_tf['high']) |      # Low > High
            (df_tf['high'] < df_tf['open']) |     # High < Open  
            (df_tf['high'] < df_tf['close']) |    # High < Close
            (df_tf['low'] > df_tf['open']) |      # Low > Open
            (df_tf['low'] > df_tf['close'])       # Low > Close
        )
        
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            print(f"   ⚠️ OHLC論理違反: {invalid_count}件検出")
            # 違反レコードを除去（データ品質保護）
            df_tf = df_tf[~invalid_ohlc]
        
        return df_tf
    
    def save_tf_parquet(self, df_tf, tf_name):
        """TFデータをParquet保存"""
        output_file = self.output_dir / f'simple_gap_aware_{tf_name}.parquet'
        
        # インデックスをリセットしてParquet保存
        df_output = df_tf.reset_index()
        df_output.to_parquet(output_file, compression='snappy', index=False)
        
        file_size = output_file.stat().st_size
        print(f"   {tf_name.upper()}: {len(df_tf):,}件 → {output_file.name} ({file_size/1024/1024:.1f}MB)")
        
        return output_file
    
    def generate_all_timeframes(self):
        """全TF生成メインフロー"""
        print("🔧 Stage 0 ETL開始: M1→全TF生成")
        print("=" * 60)
        
        # 1. M1データ読み込み
        df_m1 = self.load_m1_from_sqlite()
        
        # 2. Clean M1保存
        m1_file = self.save_clean_m1(df_m1)
        
        # 3. 各TF生成
        print("\\n3. 各TF生成（M1→上位TF）...")
        
        generated_files = [str(m1_file)]
        
        for tf_name, config in self.tf_configs.items():
            rule = config['rule']
            description = config['description']
            
            # TF生成
            df_tf = self.generate_tf_from_m1(df_m1, tf_name, rule)
            
            # 保存
            tf_file = self.save_tf_parquet(df_tf, tf_name)
            generated_files.append(str(tf_file))
        
        # 4. 生成サマリー
        print("\\n4. 生成完了サマリー:")
        summary = {
            'generation_timestamp': pd.Timestamp.now().isoformat(),
            'source_database': str(self.db_path),
            'output_directory': str(self.output_dir),
            'generated_files': generated_files,
            'tf_configs': self.tf_configs,
            'volume_policy': 'EXCLUDED_COMPLETELY',
            'gap_policy': 'SIMPLE_DROPNA',
            'timezone': 'UTC_FIXED',
            'resampling_params': {
                'label': 'left',
                'closed': 'left',
                'agg_functions': {
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last'
                }
            }
        }
        
        summary_file = self.output_dir / 'tf_generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   📋 サマリー保存: {summary_file}")
        print(f"   📁 出力ディレクトリ: {self.output_dir}")
        print(f"   📊 生成ファイル数: {len(generated_files)}")
        
        print("\\n✅ Stage 0 ETL完了")
        print("次のステップ: python3 scripts/run_validate.py で整合性検証")
        
        return summary

def main():
    """メイン実行"""
    generator = Stage0TFGenerator()
    summary = generator.generate_all_timeframes()
    return summary

if __name__ == "__main__":
    main()