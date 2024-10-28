# check_input_data.py
import os
import json
from PIL import Image
from typing import Dict, List
import mimetypes

def check_input_files(input_folder: str) -> Dict:
    """入力フォルダー内のファイルをチェック"""
    print(f"\n=== Checking Input Files in {input_folder} ===")
    
    files = os.listdir(input_folder)
    file_groups = {}
    
    # ファイルをグループ化
    for file in files:
        base_name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1].lower()
        
        if base_name not in file_groups:
            file_groups[base_name] = {
                'text': None,
                'image': None,
                'json': None,
                'issues': []
            }
        
        full_path = os.path.join(input_folder, file)
        
        # ファイルタイプの判定
        if ext == '.txt':
            file_groups[base_name]['text'] = full_path
        elif ext in ['.jpg', '.jpeg', '.png']:
            file_groups[base_name]['image'] = full_path
        elif ext == '.json':
            file_groups[base_name]['json'] = full_path

    return analyze_file_groups(file_groups)

def analyze_file_groups(file_groups: Dict) -> Dict:
    """各ファイルグループの詳細分析"""
    for base_name, group in file_groups.items():
        print(f"\nAnalyzing group: {base_name}")
        
        # ... [前のコード部分は同じ] ...
        
        # JSONファイルの確認を修正
        if group['json']:
            try:
                with open(group['json'], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                print(f"JSON file:")
                print(f"- Size: {os.path.getsize(group['json'])} bytes")
                if isinstance(json_data, list):
                    print(f"- Type: Array with {len(json_data)} items")
                    for i, item in enumerate(json_data):
                        print(f"- Item {i+1} keys: {list(item.keys())}")
                elif isinstance(json_data, dict):
                    print(f"- Keys: {list(json_data.keys())}")
                if not json_data:
                    group['issues'].append("Empty JSON file")
            except Exception as e:
                group['issues'].append(f"JSON file error: {str(e)}")
        else:
            print("No JSON file found")
        
        # 問題があれば表示
        if group['issues']:
            print("\nIssues found:")
            for issue in group['issues']:
                print(f"- {issue}")

    return file_groups

def summarize_analysis(file_groups: Dict):
    """分析結果のサマリーを表示"""
    print("\n=== Analysis Summary ===")
    
    total_groups = len(file_groups)
    groups_with_issues = sum(1 for group in file_groups.values() if group['issues'])
    
    print(f"\nTotal groups: {total_groups}")
    print(f"Groups with issues: {groups_with_issues}")
    
    if groups_with_issues > 0:
        print("\nGroups requiring attention:")
        for base_name, group in file_groups.items():
            if group['issues']:
                print(f"\n{base_name}:")
                for issue in group['issues']:
                    print(f"- {issue}")

def main():
    input_folder = "./data/chapter01"  # 入力フォルダーのパス
    
    if not os.path.exists(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    
    file_groups = check_input_files(input_folder)
    summarize_analysis(file_groups)

if __name__ == "__main__":
    main()