import subprocess
import os
import sys

def run_hle_script_multiple_times(script_path, num_runs: int):
    """
    指定されたshスクリプトを複数回実行します。
    現在の実行回数を引数としてスクリプトに渡します。

    Args:
        script_path (str): 実行するshスクリプトのパス。
        num_runs (int): スクリプトを実行する回数。
    """
    if not os.path.exists(script_path):
        print(f"Error: Script file not found at {script_path}")
        return

    print(f"Running script {script_path} for {num_runs} times.")

    for i in range(num_runs):
        run_number = i + 1
        print(f"--- Starting run {run_number}/{num_runs} ---")
        
        try:
            # ここを修正: スクリプトパスと引数のリストを渡す
            # 実行回数を文字列に変換して引数として追加
            # 修正後
            # Pythonスクリプトの親ディレクトリをPYTHONPATHに追加する
            script_dir = os.path.dirname(script_path)
            parent_dir = os.path.dirname(script_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = parent_dir + ":" + env.get("PYTHONPATH", "")

            command = [sys.executable, script_path, '--num_runs', str(run_number)]
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)

            
            print(f"Script output for run {run_number}:")
            print(result.stdout)
            print(f"--- Run {run_number} completed successfully. ---")
        except subprocess.CalledProcessError as e:
            print(f"--- Error during run {run_number}: ---")
            print(f"Command failed with exit code {e.returncode}")
            print(f"Error output: {e.stderr}")
        except FileNotFoundError:
            print(f"Error: Command '{script_path}' not found.")
            return

    print("All runs finished.")

if __name__ == "__main__":
    script_to_run = "/home/Competition2025/P12/P12U007/50_eval_64_hle/eval_hle/hle_benchmark/vllm_predictions.py" 
    # script_to_run = "/home/Competition2025/P12/P12U007/50_eval_64_hle/eval_hle/hle_benchmark/run_judge_results.py"
    number_of_runs = 64
    run_hle_script_multiple_times(script_to_run, number_of_runs)
