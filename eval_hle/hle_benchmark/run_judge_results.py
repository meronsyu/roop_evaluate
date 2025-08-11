# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import json
import copy
import math
import argparse
import asyncio
import numpy as np
from typing import Literal
from pydantic import BaseModel
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
from datetime import datetime
from hle_benchmark._configs import Config

# 非同期処理
client = AsyncOpenAI(
    timeout=300.0,
    max_retries=1,
    api_key=<APIKEY>
)


# 回答、reasoning、cofidenceの妥当性を図る
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""

# pydanticによる型設定
class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] # 100% reliability


# LLMが作成した駄文から評価結果をjudgeモデルを用いて出力    
async def extract_answer(args, question, correct_answer, response):
    prompt = JUDGE_PROMPT.format(question=question, correct_answer=correct_answer, response=response)
    try:
        response = await client.beta.chat.completions.parse(
                model=args.judge,
                max_completion_tokens=4096, # overkill for judge
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format=ExtractedAnswer, 
            ) 
        content = response.choices[0].message.parsed
        return { 
            "correct_answer": correct_answer,
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence
        }
    except Exception as e: # very, very rare
        print("Error:", e)
        return None

# 評価済みでない場合、評価を行う         
async def add_judge_response(args, question, predictions):
    unique_id = question["id"]
    
    # 予測データにキーが存在しない場合は、処理をスキップ
    if unique_id not in predictions:
        # この警告メッセージは、どの問題がスキップされたか知る上で役立ちます
        print(f"Warning: No prediction found for question ID '{unique_id}'. Skipping.")
        return None, None
        
    prediction = copy.deepcopy(predictions[unique_id]) # not in-place
    question_text = question["question"]
    correct_answer = question["answer"]

    if "judge_response" in prediction: # already judged
        return unique_id, prediction
    
    response = prediction["response"]
    
    # judgeモデルに回答評価
    content = await extract_answer(args, question_text, correct_answer, response)

    if content is not None:
        prediction["judge_response"] = content # local in-place
        return unique_id, prediction
    else:
        return None, None

# openaiに一括で投げる並列処理
# 並列で何個投げるのか
async def judge_all_responses(args, questions, predictions):
    async def bound_func(question):
        async with semaphore:
            content = await add_judge_response(args, question, predictions)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results

# エラーの計算方法を決定するパラメータについて、
# エラーの計算方法
## '2' (デフォルト): L2ノルム（二乗平均平方根誤差）を使用。大きな誤差に敏感
## '1': L1ノルム（平均絶対誤差）を使用。
## 'infty' / 'infinity' / 'max': 最大誤差を使用
# 各「ビン」（区間）の目標となるサンプルサイズ。自信度をソートし、このサイズでビンに分割。
# source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py

# todo：binごとのキャリブレーションを求めたい。
# 自信度ごとのキャリブレーションの値がわかる
def calib_err(confidence, correct, p='2', beta=50): 
    # 
    if len(confidence) < beta:
        return 0.0
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr

# 
def dump_metrics(args, predictions, total_questions, all_questions):
    correct = []
    confidence = []
    correct_by_category = {
        "Math": [], # 41%
        "Physics": [], # 9%
        "Biology/Medicine": [], # 11%
        "Humanities/Social Science": [], # 9%
        "Computer Science/AI": [], # 10%
        "Engineering": [], # 4%
        "Chemistry": [], # 7%
        "Other": [], # 9%
    }
    for k, v in predictions.items():
        data = next(filter(lambda x: x["id"] == k, all_questions))
        if "judge_response" in v:
            judge_response = v["judge_response"]
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
            correct_by_category[data["category"]].append("yes" in judge_response["correct"])
        else:
            print(f"Missing judge response for {k}, you should rerun the judge")

    correct = np.array(correct)
    confidence = np.array(confidence) / 100
    n = len(correct)

    # sometimes model collapses on same questions
    if len(correct) != total_questions:
        print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")


    accuracy = round(100 * sum(correct) / n, 2)
    # Wald estimator, 95% confidence interval
    confidence_half_width = round(1.96 * math.sqrt(accuracy * (100 - accuracy) / n), 2)
    calibration_error = 100 * round(calib_err(confidence, correct, p='2', beta=100), 2)

    print("*** Metrics ***")
    print(f"Accuracy: {accuracy}% +/- {confidence_half_width}% | n = {n}")
    print(f"Calibration Error: {calibration_error}")

    print("\n*** Accuracy Per Category ***") # ここから追加
    for category, results_list in correct_by_category.items():
        num_correct = sum(results_list) # Trueの数を数える
        num_total = len(results_list)   # そのカテゴリの総質問数

        if num_total > 0:
            print(f"- {category}: {num_correct} / {num_total} ({round(num_correct / num_total * 100, 2)}%)")
        else:
            print(f"- {category}: No questions evaluated in this category.")
    print("*****************************") # ここまで追加

    results = []
    for k,v in predictions.items():
        data = next(filter(lambda x: x["id"] == k, all_questions))
        results.append({
            "id": k,
            "category": data["category"],
            "question": data["question"],
            "user_prompt": "", # TODO
            "answer_type": data["answer_type"],
            "prediction": v["judge_response"]["model_answer"],
            "gold": v["judge_response"]["correct_answer"],
            "correct": 1 if v["judge_response"]["correct"] == "yes" else 0,
            "judgement": v["judge_response"]["reasoning"],
        })

    summary = {
        "model_name": predictions[k]["model"],
        "overall_accuracy": accuracy,
        "accuracy_per_category": {
            "Math": sum(correct_by_category["Math"]) / len(correct_by_category["Math"]) * 100 if len(correct_by_category["Math"]) else None,
            "Physics": sum(correct_by_category["Physics"]) / len(correct_by_category["Physics"]) * 100 if len(correct_by_category["Physics"]) else None,
            "Biology/Medicine": sum(correct_by_category["Biology/Medicine"]) / len(correct_by_category["Biology/Medicine"]) * 100 if len(correct_by_category["Biology/Medicine"]) else None,
            "Humanities/Social Science": sum(correct_by_category["Humanities/Social Science"]) / len(correct_by_category["Humanities/Social Science"]) * 100 if len(correct_by_category["Humanities/Social Science"]) else None,
            "Computer Science/AI": sum(correct_by_category["Computer Science/AI"]) / len(correct_by_category["Computer Science/AI"]) * 100 if len(correct_by_category["Computer Science/AI"]) else None,
            "Engineering": sum(correct_by_category["Engineering"]) / len(correct_by_category["Engineering"]) * 100 if len(correct_by_category["Engineering"]) else None,
            "Chemistry": sum(correct_by_category["Chemistry"]) / len(correct_by_category["Chemistry"]) * 100 if len(correct_by_category["Chemistry"]) else None,
            "Other": sum(correct_by_category["Other"]) / len(correct_by_category["Other"]) * 100 if len(correct_by_category["Other"]) else None,
        },
        "num_questions": n,
        "timestamp": datetime.now().isoformat(),
    }

    output_folder_name = "leaderboard/" +datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(output_folder_name, exist_ok=True)
    with open(output_folder_name + "/results.jsonl", "w") as f:
        f.writelines([json.dumps(l) + "\n" for l in results])
    with open(output_folder_name + "/summary.json", "w") as f:
        json.dump(summary, f, indent=4) 


def dump_metrics_by_confidence_and_category(args, predictions, total_questions, all_questions):
    """
    Calculates metrics by confidence bins and categories and prints the results to the console.
    """
    confidence_bins = {
        "0-20%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "20-40%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "40-60%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "60-80%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
        "80-100%": {"categories": {}, "total_in_bin": 0, "correct_in_bin": 0},
    }

    categories = {
        "Math": [], # 41%
        "Physics": [], # 9%
        "Biology/Medicine": [], # 11%
        "Humanities/Social Science": [], # 9%
        "Computer Science/AI": [], # 10%
        "Engineering": [], # 4%
        "Chemistry": [], # 7%
        "Other": [], # 9%
    }

    for bin_name in confidence_bins:
        for category in categories:
            confidence_bins[bin_name]["categories"][category] = {"num_samples": 0, "num_correct": 0}

    # 集計処理
    for q_id, pred in predictions.items():
        if "judge_response" in pred:
            judge_response = pred["judge_response"]
            confidence_score = judge_response["confidence"]
            is_correct = ("yes" in judge_response["correct"])

            # 予測結果に対応する質問データを取得
            question_data = next((q for q in all_questions if q["id"] == q_id), None)
            if not question_data:
                continue
            category = question_data["category"]

            bin_name = None
            if 0 <= confidence_score < 20:
                bin_name = "0-20%"
            elif 20 <= confidence_score < 40:
                bin_name = "20-40%"
            elif 40 <= confidence_score < 60:
                bin_name = "40-60%"
            elif 60 <= confidence_score < 80:
                bin_name = "60-80%"
            elif 80 <= confidence_score <= 100:
                bin_name = "80-100%"

            if bin_name:
                confidence_bins[bin_name]["categories"][category]["num_samples"] += 1
                confidence_bins[bin_name]["total_in_bin"] += 1
                if is_correct:
                    confidence_bins[bin_name]["categories"][category]["num_correct"] += 1
                    confidence_bins[bin_name]["correct_in_bin"] += 1

    # コンソール出力
    for bin_name, bin_data in confidence_bins.items():
        total_samples = bin_data["total_in_bin"]

        if total_samples > 0:
            print(f"\n*** Accuracy for Confidence {bin_name} ***")
            for category in categories:
                cat_data = bin_data["categories"][category]
                num_correct = cat_data["num_correct"]
                num_total = cat_data["num_samples"]
                accuracy = round(num_correct / num_total * 100, 2) if num_total > 0 else 0.0
                
                print(f"- {category}: {num_correct} / {num_total} ({accuracy}%)")
            print("*************************************")
        else:
            print(f"\n*** No predictions in Confidence Bin {bin_name} ***")
            print("*************************************")

def dump_metrics_by_raw_subject(args, predictions, all_questions):
    """
    Calculates and prints accuracy per raw subject based on judged predictions.
    """
    correct_by_raw_subject = {}

    # raw_subjectのリストを動的に生成
    raw_subjects = sorted(list(set(q["raw_subject"] for q in all_questions if "raw_subject" in q)))

    for subject in raw_subjects:
        correct_by_raw_subject[subject] = {"num_samples": 0, "num_correct": 0}

    for q_id, pred in predictions.items():
        if "judge_response" in pred:
            judge_response = pred["judge_response"]
            is_correct = ("yes" in judge_response["correct"])
            
            question_data = next((q for q in all_questions if q["id"] == q_id), None)
            if not question_data:
                continue
            
            subject = question_data.get("raw_subject")
            if subject and subject in correct_by_raw_subject:
                correct_by_raw_subject[subject]["num_samples"] += 1
                if is_correct:
                    correct_by_raw_subject[subject]["num_correct"] += 1

    # サンプル数が多いもの事にソート
    raw_subjects = sorted(
        correct_by_raw_subject.keys(),
        key=lambda s: correct_by_raw_subject[s]["num_samples"],
        reverse=True
    )
    print("\n*** Accuracy Per Raw Subject ***")
    for subject in raw_subjects:
        results_data = correct_by_raw_subject[subject]
        num_correct = results_data["num_correct"]
        num_total = results_data["num_samples"]

        if num_total > 0:
            accuracy = round(num_correct / num_total * 100, 2)
            print(f"- {subject}: {num_correct} / {num_total} ({accuracy}%)")
        else:
            print(f"- {subject}: No questions evaluated in this raw subject.")
    print("*****************************")

def main(args: Config, num_runs: int):
    assert args.num_workers > 1, "num_workers must be 2 or greater"

    output_filepath = f"judged/50/judged_hle_{os.path.basename(args.model)}_50_{num_runs}.json"   
    dataset = load_dataset(args.dataset, split="test")
    
    # 画像フィルタ
    dataset = dataset.filter(lambda item: item['image'] == "")

    # フィルタリングしたいidのリスト
    selected_ids_list = ['66ef5e39a57b7ef1047d2f58', '6730a9be58ef965949f1faa4', '674818a0d2dbfbb2e99bd257', '66ffaae1068d942d32104650', '6776ffa855710e45669a4481', '671b16741c8a4560f96a3a66', '6726f1f73958e8253ed79aed', '6706c88503718618700edfbc', '66f25c95da5074d064015c54', '6742f485e9256150e88912f1', '66f57e187a2ac7b4fffe174e', '67079b3aeb66f588bd3789ec', '67225e2f53af61d9b29732c8', '676e9656e3e0846ee73dbf9d', '672c8e7a86c5d04249bd338e', '6719c58cd5ad96a75c350fa6', '66fa67dc29908114d8954b55', '6712f157cf58f1d947689708', '6713cedd6978edcd74f82863', '6725280ff2e932808735b2e8', '6734956467d2904eebed3a09', '6721767ddb8105efc71a7d1b', '67455f379dbdcf3802abd8f6', '67372744600c9c0daa5d8f3f', '6723ba00339572beba8f91b2', '67672352c393c4ff629cb820', '673af092fa64168165769f1e', '67216d4134cd9a7f210d4100', '67335820c7d8c66591e6dfc7', '672e6368958c5d0efe9d037d', '67015a7f6a2b21f149f3aaba', '6742fe5ea2e78a79e46bb220', '6725145d97743d26179494d6', '6736f62aafbcf9397103d22f', '671963d90f87e9920aff9d11', '671fb4ddf2a13c812068cdd7', '67009ad56c339d61ecccb85c', '67249d57d91756473725533a', '671bf785fc1ad15079783d89', '66fc49ef5067ff35e6302b7f', '676226f6fbdba9bd68127327', '6730f3c006cd33fe46ca2dfe', '673681def5487e4de6e78e1e', '6725255ee6807ea2c8372c3c', '66ff0a666dc4b2ab76a19db9', '67095af56022f900e8e76028', '675b1c7bc75183431b382945', '679ea13cf03a19732cbbe68f', '6724955b1dc5c59953c463ec', '675c41c7fbd66ff2e12f23c0']
    # リストをセットに変換することで、検索が高速になります
    selected_ids_set = set(selected_ids_list)

    # ラムダ関数を使ってデータセットをフィルタリング
    dataset = dataset.filter(lambda data: data['id'] in selected_ids_set)

    # dataset = dataset.select(range(2))
    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    all_questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    

    total_questions = len(all_questions)

    # output_filepath のディレクトリ部分を取得
    output_dir = os.path.dirname(output_filepath)
    # ディレクトリが存在しない場合に作成
    os.makedirs(output_dir, exist_ok=True) 
    # todo:今回カテゴリごとにばらばらに予測を行ったから貼り付ける必要あり。
    with open(f"predictions/50_eval/hle_{os.path.basename(args.model)}_50_{num_runs}.json", "r") as f:
        predictions = json.load(f)

     # Qwen3 will be called for unjudged responses
    results_list = asyncio.run(judge_all_responses(args, all_questions, predictions))

    # judged_predictionsをここで初期化
    judged_predictions = {}

    # cache judge output
    with open(output_filepath, "w") as f:
        json.dump(results_list, f, indent=4)

    # リストを辞書に変換
    for unique_id, updated_prediction in results_list:
        if unique_id is not None:
            judged_predictions[unique_id] = updated_prediction
    
    dump_metrics(args, judged_predictions, total_questions, all_questions)

    dump_metrics_by_confidence_and_category(args, judged_predictions, total_questions, all_questions)

    dump_metrics_by_raw_subject(args, judged_predictions, all_questions)


if __name__ == "__main__":
    # コマンドライン引数をパースする
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run the evaluation.")
    args = parser.parse_args()

    # Configオブジェクトをハードコーディングした値で作成
    config_obj = Config(
        dataset="cais/hle",
        provider="vllm",
        base_url="http://localhost:8000/v1",
        model="/home/Competition2025/P12/P12U007/model/Qwen3-32B",
        max_completion_tokens=100000,
        reasoning=True,
        num_workers=2500,
        max_samples=2500,
        judge="o3-mini-2025-01-31"
    )

    # main関数を呼び出す際に、作成したオブジェクトとnum_runsを渡す
    main(config_obj, args.num_runs)