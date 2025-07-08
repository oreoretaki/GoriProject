#!/usr/bin/env python3
"""
データ再現性検証スクリプト
"""

import json
import hashlib
import pandas as pd
from pathlib import Path

# 現在のデータハッシュ
EXPECTED_HASHES = {
  "source_m1_sample": "fa9b835e31765c0b2a9a8ef6522b42d1",
  "safe_ohlc_m5": "5a96704f088ca0a633c6119e3c789214",
  "safe_ohlc_m5_size": 10595335,
  "safe_ohlc_m15": "3ede9c80fbb55ea7c4e9463346effd46",
  "safe_ohlc_m15_size": 4201529,
  "safe_ohlc_h1": "2803b36170165db9f507d96388e9917f",
  "safe_ohlc_h1_size": 1273973,
  "safe_ohlc_d": "46d7bc2d94554255e26ebf62e54dbe4a",
  "safe_ohlc_d_size": 82054
}

def verify_data_integrity():
    """データ整合性の検証"""
    
    current_hashes = {}
    
    # 生成Parquetファイルのハッシュ再計算
    data_dir = Path('data/derived')
    
    for tf in ['m5', 'm15', 'h1', 'd']:
        parquet_file = data_dir / f'safe_ohlc_{tf}.parquet'
        
        if parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            file_hash = hashlib.md5(df.to_string().encode()).hexdigest()
            current_hashes[f'safe_ohlc_{tf}'] = file_hash
            
    # 比較
    mismatches = []
    for key, expected in EXPECTED_HASHES.items():
        if key.endswith('_size'):
            continue  # ファイルサイズは除外
            
        current = current_hashes.get(key)
        if current != expected:
            mismatches.append({
                'file': key,
                'expected': expected,
                'current': current
            })
    
    return len(mismatches) == 0, mismatches

if __name__ == "__main__":
    success, mismatches = verify_data_integrity()
    
    if success:
        print("✅ データ整合性検証PASS")
    else:
        print("❌ データ整合性検証FAIL")
        for mismatch in mismatches:
            print(f"  {mismatch['file']}: ハッシュ不一致")
