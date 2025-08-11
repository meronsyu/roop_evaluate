import os
import json
from collections import defaultdict
from datasets import load_dataset

def analyze_judged_results(base_dir="judged/50"):
    """
    指定されたディレクトリ内のJSONファイルを分析し、問題IDごとの正解数を集計します。
    
    Args:
        base_dir (str): JSONファイルが保存されているディレクトリのパス。
    """
    id_results = defaultdict(lambda: {"total": 0, "correct": 0})
    
    # 指定されたディレクトリ内のすべてのファイルをリストアップ
    try:
        file_list = [f for f in os.listdir(base_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Error: Directory '{base_dir}' not found.")
        return

    if not file_list:
        print(f"No JSON files found in '{base_dir}'.")
        return

    print(f"Analyzing {len(file_list)} JSON files...")

    for filename in file_list:
        filepath = os.path.join(base_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # dataがリスト形式であることを前提に修正
                for item in data:
                    # itemがNoneではないかチェック
                    if item is not None:
                        q_id = item[0]
                        result = item[1]

                        # resultが辞書であることを前提にNoneチェックを追加
                        if result is not None and "judge_response" in result:
                            is_correct = result["judge_response"]["correct"] == "yes"
                            
                            id_results[q_id]["total"] += 1
                            if is_correct:
                                id_results[q_id]["correct"] += 1
        except json.JSONDecodeError:
            print(f"Warning: Skipping file '{filename}' due to JSON decode error.")
        except (KeyError, IndexError) as e:
            print(f"Warning: Skipping an entry in '{filename}' due to a structural error: {e}")

    # 集計結果の出力
    print("\n--- Summary of Correct Answers per Question ID ---")
    for q_id, counts in id_results.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"問題ID: {q_id} | 正解数: {correct}/{total} ({accuracy:.2f}%)")

def analyze_judged_results(base_dir="judged/50"):
    """
    指定されたディレクトリ内のJSONファイルを分析し、問題IDごとの正解数とカテゴリを集計します。
    
    Args:
        base_dir (str): JSONファイルが保存されているディレクトリのパス。
    """
    id_results = defaultdict(lambda: {"total": 0, "correct": 0})
    
    # 元のデータセットからIDとカテゴリの対応付けを作成
    print("Loading original dataset for category mapping...")
    try:
        dataset = load_dataset("cais/hle", split="test")
        category_map = {item["id"]: item["category"] for item in dataset}
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 指定されたディレクトリ内のすべてのファイルをリストアップ
    try:
        file_list = [f for f in os.listdir(base_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Error: Directory '{base_dir}' not found.")
        return

    if not file_list:
        print(f"No JSON files found in '{base_dir}'.")
        return

    print(f"Analyzing {len(file_list)} JSON files...")

    for filename in file_list:
        filepath = os.path.join(base_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                # dataがリスト形式であることを前提に修正
                for item in data:
                    # itemがNoneではないかチェック
                    if item is not None and len(item) == 2:
                        q_id = item[0]
                        result = item[1]

                        # resultが辞書であることを前提にNoneチェックを追加
                        if result is not None and "judge_response" in result:
                            is_correct = result["judge_response"]["correct"] == "yes"
                            
                            id_results[q_id]["total"] += 1
                            if is_correct:
                                id_results[q_id]["correct"] += 1
        except json.JSONDecodeError:
            print(f"Warning: Skipping file '{filename}' due to JSON decode error.")
        except (KeyError, IndexError) as e:
            print(f"Warning: Skipping an entry in '{filename}' due to a structural error: {e}")

    # 集計結果の出力
    print("\n--- Summary of Correct Answers per Question ID ---")
    for q_id, counts in id_results.items():
        total = counts["total"]
        correct = counts["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # カテゴリ情報を取得
        category = category_map.get(q_id, "Unknown")
        
        print(f"問題ID: {q_id} | カテゴリ: {category} | 正解数: {correct}/{total} ({accuracy:.2f}%)")

if __name__ == "__main__":
    analyze_judged_results()