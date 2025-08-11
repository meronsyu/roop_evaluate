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
from dataclasses import dataclass
import os
import json
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from hle_benchmark._configs import Config
import datetime # datetimeモジュールをインポート
import argparse


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def format_message(args, question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(args, question):
    messages = format_message(args, question)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            # temperature=args.temperature if "o1" not in args.model else None,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        ) 
        content = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            return None
        tokens = json.loads(response.usage.model_dump_json())
        if args.reasoning:
            tokens["reasoning_content"] = response.choices[0].message.reasoning_content
    except Exception as e:
        print("Error:", e)
        return None
    
    if content is None: # failed
        return None
    
    return question["id"], content, tokens


# 回答が10問ごとに保存
async def attempt_all(args, questions, predictions, output_filepath):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(args, question)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = []
        completed_count = 0
        
        # Process results as they complete and save every 10 items
        for result in tqdm_asyncio.as_completed(tasks):
            result_data = await result
            results.append(result_data)
            
            if result_data is not None:
                unique_id, response, usage = result_data
                predictions[unique_id] = {
                    "model": args.model,
                    "response": response,
                    "usage": usage
                }
                completed_count += 1
                print(f"Processed question ID: {unique_id}. Total completed: {completed_count}")
                
                # Save every 10 completed items
                if completed_count % 10 == 0:
                    with open(output_filepath, "w") as f:
                        json.dump(predictions, f, indent=4)
                    print(f"Saved progress: {completed_count} items completed")
    
    return results

def main(args: Config, num_runs: int): 
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )

    assert args.num_workers > 1, "num_workers must be 2 or greater"
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
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]


    # ファイル名にモデル名とタイムスタンプを組み込む
    # 例: predictions/hle_qwen3-32b_20250731_023309.json
    # 複数の実行で衝突が起きないようにするため
    output_filepath = f"predictions/50_eval/hle_{os.path.basename(args.model)}_50_{num_runs}.json"

    # もしディレクトリがない場合は作成
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = asyncio.run(attempt_all(args, questions, predictions, output_filepath))

    # Final save to ensure all data is persisted
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Final save completed: {len([r for r in results if r is not None])} total items processed")

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
