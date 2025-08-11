# roop_evaluate
50題同じ問題を64回解くことを実行するためのコードです。
分布に従って50問ランダムに問題を取得し、取得した問題文をN回繰り返し問題を解くコードです。
主に、hayashiさんの提案するAIMEタスクにおいてのpass1からcons64による性能向上がHLEタスクにおいて
も成り立つか調べるためのコードです。
現在は、学習用のHLEタスク50問を大体60回解き終えたので、解き終えた回答から正解のもののみを抽出し、データセットとしてSFT学習することで
どれだけ評価用のHLEタスク50問を解けるようになるのか判別するフェーズです。

## 注意
環境構築前提です。
eval_hle/hle_script.shのhuggingfaceのトークンを自分で埋めてください。
eval_hle/hle_benchmark/run_judge_results.pyの42行目のopenai tokenを自分で埋めてください。
パス関係を多少いじる必要があると思います。
Qwen3 32Bを使います。8gpuの方が早く動きます。bashでgpu内に入って動かす想定です。

## 評価の実行方法
### 実行前
GPU数を設定
```python
bash ../shareP12/scancel_hatakeyama.sh gpu84 gpu85 gpu86 && srun --job-name=evaluate --partition P12 --nodes=1 --nodelist osk-gpu[86] --gpus-per-node=4 --ntasks=16 --time=12:00:00 --pty bash -i
```
```python
conda activate llmbench
```

eval_hleのhle_script.shを自分が評価したいモデル、それに適したGPU数GPU数に変更
```python
#--- GPU 監視 -------------------------------------------------------
nvidia-smi -i 0,1,2,3,4 -l 3 > nvidia-smi.log &
pid_nvsmi=$!

#--- vLLM 起動（2GPU）----------------------------------------------
# tensor-parallel-sizeについてはmulti headsを割り切れる数に指定する必要あり
vllm serve Qwen/QwQ-32B \
  --tensor-parallel-size 4 \
  --reasoning-parser deepseek_r1 \
```

### 追加コードの位置
hle_benchmarkの中に
select_random50.py # 分布に沿って50問のidを取得
eval_hleの中に
roop_hle_prediction.py # n回繰り返し実行するためのコード #vllm_predictions.pyの繰り返しとrun_judge_results.pyの繰り返し
### 実行時
モデルが立ち上がるまで待機。nohup.outの中身を見ながら待つ。
```python
nohup ./hle_script.sh > vllm.log 2>&1 &
```
分布に従った50問を取得
```python
python3 select_random50.py 
```
実行結果からリストをコピペし、vllm_predictions.pyのリストを書き換え。
run_judge_results.pyのパスをvllm_prediction.pyの自分のパスに、Configも直書きしているので自分のものに書き換え。
うまく実行できると、指定した回数分vllm_predictions.pyが実行される

eval_hle/confのconfig.yamlを自分が評価したいにモデルに変更
```python
OPENAI_API_KEY=xxx python judge.py
```


## 注意点
慣れるまではエラーが出ると思うので、shellスクリプトを実行するときは&を消して、ログを確認するのをお勧めします。
gpu数は、モデルごとにmulti head数と語彙数でassertエラーが起きる可能性があるので、基本２の倍数がお勧めです。
https://www.notion.so/239e14b94af5807f88d0df0189e3cc98 にmulti head数の見方が載っています。

chmod +x hle_prediction.sh
chmod +x hle_script.sh
を忘れずに
## Contributors

```
```

## 学習用の数字
['66ef5e39a57b7ef1047d2f58', '6730a9be58ef965949f1faa4', '674818a0d2dbfbb2e99bd257', '66ffaae1068d942d32104650', '6776ffa855710e45669a4481', '671b16741c8a4560f96a3a66', '6726f1f73958e8253ed79aed', '6706c88503718618700edfbc', '66f25c95da5074d064015c54', '6742f485e9256150e88912f1', '66f57e187a2ac7b4fffe174e', '67079b3aeb66f588bd3789ec', '67225e2f53af61d9b29732c8', '676e9656e3e0846ee73dbf9d', '672c8e7a86c5d04249bd338e', '6719c58cd5ad96a75c350fa6', '66fa67dc29908114d8954b55', '6712f157cf58f1d947689708', '6713cedd6978edcd74f82863', '6725280ff2e932808735b2e8', '6734956467d2904eebed3a09', '6721767ddb8105efc71a7d1b', '67455f379dbdcf3802abd8f6', '67372744600c9c0daa5d8f3f', '6723ba00339572beba8f91b2', '67672352c393c4ff629cb820', '673af092fa64168165769f1e', '67216d4134cd9a7f210d4100', '67335820c7d8c66591e6dfc7', '672e6368958c5d0efe9d037d', '67015a7f6a2b21f149f3aaba', '6742fe5ea2e78a79e46bb220', '6725145d97743d26179494d6', '6736f62aafbcf9397103d22f', '671963d90f87e9920aff9d11', '671fb4ddf2a13c812068cdd7', '67009ad56c339d61ecccb85c', '67249d57d91756473725533a', '671bf785fc1ad15079783d89', '66fc49ef5067ff35e6302b7f', '676226f6fbdba9bd68127327', '6730f3c006cd33fe46ca2dfe', '673681def5487e4de6e78e1e', '6725255ee6807ea2c8372c3c', '66ff0a666dc4b2ab76a19db9', '67095af56022f900e8e76028', '675b1c7bc75183431b382945', '679ea13cf03a19732cbbe68f', '6724955b1dc5c59953c463ec', '675c41c7fbd66ff2e12f23c0']

## 評価用の数字
['6700aa576c5c0e0d48330ad1', '6720c7d9831d6450ba886ff0', '672262d988e407d7eb07744d', '674365a2b5d4e34a242899c5', '6706c88503718618700edfbc', '6747da64aa7f6182ef02efae', '6736d46462d883a670c09b5d', '66ec5671713826aca26a9dba', '6737c6ff3b2291333fbb48a4', '6702c6251454b2a5a3a4b0a9', '6740dcfff2fec88c6301d048', '672604e44ee1765ace1c8a40', '67350ad443f1d86ec88ce396', '672db3a88bc9e04cb09bf8f7', '6721176ed9397a0c64dba826', '6739374caf97ceab210adc21', '67955d8d28dc0cc835a5c3c1', '6721a42269600ecb23a3e623', '673704af1c2083e9eaa6d732', '671ec6d8a695a5847b48c39a', '6725e42052e181595c8bf328', '671ff43951f8a38cb737b3d4', '66b827b9b64deaedfbb997a2', '671abddb40e08981d8fd796a', '66ef3de8b8a1ba6e0ba23498', '66f1b5cec75c3ece07124859', '673ae99a3022b9e17f89d1b6', '6775b1ab76b69969709e80a7', '671ae258d2ea402783696a9f', '67363709b1a97400f39cdc71', '672fb1872357e1d1acd486dc', '67055b15222ba55ab0b88431', '6742fe5ea2e78a79e46bb220', '66e9b2899cf8fcf41599246f', '672be113f0b152f5e0c9e635', '66eb35e9e3d7202c68806272', '66e8d3ed713a83e8aeddc2f5', '671f941e919b83ce036e80d9', '671929c60fa7bca6462f63a3', '6722728827542064f9b14815', '66fc49ef5067ff35e6302b7f', '670e76dbfb3b4f650fe86113', '672c033ff576aed47449d75f', '6700a5f67899925724a8fa81', '66ecddac93c4bf17acb6c106', '668828540a642802bdfeadfc', '673612eaff612f83700ec41f', '670a2a076973040cfd5994a7', '67127dbf0d05bc73fc008e02', '6722a65a27a5569d0c5f5c0f']