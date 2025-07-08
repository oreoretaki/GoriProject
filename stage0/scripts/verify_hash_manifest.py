#!/usr/bin/env python3
"""
ハッシュマニフェスト検証スクリプト
CI/CDで使用
"""

import hashlib
import json
import sys
from pathlib import Path

def calculate_md5(file_path):
    """ファイルのMD5ハッシュ計算"""
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def verify_manifest():
    """マニフェスト検証"""
    manifest_file = Path('reports/hash_manifest.json')
    
    if not manifest_file.exists():
        print("❌ hash_manifest.json が見つかりません")
        return False
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    print("🔍 データファイル整合性検証開始")
    print("=" * 40)
    
    all_passed = True
    
    for file_path, expected in manifest['files'].items():
        path_obj = Path(file_path)
        
        if not path_obj.exists():
            print(f"❌ {file_path}: ファイルが見つかりません")
            all_passed = False
            continue
        
        # ハッシュ計算
        actual_hash = calculate_md5(path_obj)
        expected_hash = expected['md5']
        
        if actual_hash == expected_hash:
            print(f"✅ {file_path}: OK ({actual_hash[:8]}...)")
        else:
            print(f"❌ {file_path}: ハッシュ不一致")
            print(f"   期待値: {expected_hash}")
            print(f"   実際値: {actual_hash}")
            all_passed = False
    
    if all_passed:
        print("\\n🎯 全ファイル検証合格")
        print("データ整合性: ✅ OK")
        return True
    else:
        print("\\n🚨 ファイル整合性エラーが検出されました")
        return False

if __name__ == "__main__":
    success = verify_manifest()
    sys.exit(0 if success else 1)