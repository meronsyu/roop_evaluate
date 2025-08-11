import random
from collections import defaultdict
from datasets import load_dataset

def get_questions_by_category():
    """
    データセットを読み込み、カテゴリごとに問題をグループ化して返します。
    """
    # データセットを読み込み、画像のない問題にフィルタリング
    dataset = load_dataset("cais/hle", split="test")
    dataset = dataset.filter(lambda item: item['image'] == "")

    # 辞書に新しいカテゴリのキーが追加されると、自動的に空のリストが値として割り当てられる
    questions_by_category = defaultdict(list)
    
    # 対応するカテゴリのリストにIDを追加
    for item in dataset:
        questions_by_category[item["category"]].append(item["id"])
    
    return questions_by_category

def get_random_questions(questions_by_category):
    """
    指定された問題数に基づいて、カテゴリごとにランダムに問題IDとカテゴリのペアを取得します。
    """
    # カテゴリごとに取得する問題数を定義
    category_counts = {
        "Math": 18,
        "Physics": 5,
        "Biology/Medicine": 6,
        "Humanities/Social Science": 5,
        "Computer Science/AI": 5,
        "Engineering": 2,
        "Chemistry": 4,
        "Other": 5,
    }
    
    selected_questions = []
    selected_questions_id = []
    
    # キーと値のペアを一度に取得
    for category, count in category_counts.items():
        # カテゴリ内に十分な問題があることを前提としてランダムサンプリング
        if category in questions_by_category:
            random_ids = random.sample(questions_by_category[category], count)
            # 各IDをカテゴリとペアにしてリストに追加
            for q_id in random_ids:
                selected_questions.append({"id": q_id, "category": category})
            for q_id in random_ids:
                selected_questions_id.append(q_id)
        else:
            print(f"Warning: Category '{category}' not found in the dataset.")
            
    # 合計数が50問になることを確認
    if len(selected_questions) != 50:
        print(f"Error: The total number of selected questions is {len(selected_questions)}, not 50.")
    
    return selected_questions, selected_questions_id

def count_categories(selected_questions):
    """
    IDとカテゴリのペアのリストから、カテゴリごとの問題数をカウントします。
    """
    category_counts = defaultdict(int)
    for q in selected_questions:
        category_counts[q["category"]] += 1
    return category_counts


if __name__ == "__main__":
    try:
        # ステップ1: カテゴリごとに問題をグループ化
        questions_by_category = get_questions_by_category()
        
        # ステップ2: カテゴリ分布に基づいて問題をランダム取得
        # ここでIDだけでなく、カテゴリ情報も取得するように変更
        selected_questions, selected_questions_ids  = get_random_questions(questions_by_category)
        
        # 結果の出力
        print("Selected 50 question IDs based on category distribution:")
        for q in selected_questions:
            print(q)
        print(f"Total selected questions: {len(selected_questions)}")
        print(selected_questions_ids)

        # 新しい関数の呼び出し（修正されたリストを渡す）
        final_counts = count_categories(selected_questions)
        
        # 結果の出力
        print("\nFinal category distribution from selected questions:")
        for category, count in final_counts.items():
            print(f"- {category}: {count} questions")


    except Exception as e:
        print(f"An error occurred: {e}")
