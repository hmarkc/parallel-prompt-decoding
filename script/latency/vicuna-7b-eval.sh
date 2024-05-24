echo "Vicuna 7B Evaluation"
echo "---"
echo "Vicuna Baseline"
python3 get_throughput_results.py data/mt_bench/experiments/vicuna-7b-baseline1.jsonl data/mt_bench/experiments/vicuna-7b-baseline2.jsonl data/mt_bench/experiments/vicuna-7b-baseline3.jsonl
echo "---"
echo "Prompt Decoding"
python3 get_throughput_results.py data/mt_bench/experiments/vicuna-7b-faster1.jsonl data/mt_bench/experiments/vicuna-7b-faster2.jsonl data/mt_bench/experiments/vicuna-7b-faster3.jsonl