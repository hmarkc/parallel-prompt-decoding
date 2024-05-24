echo "Vicuna 13b Evaluation"
echo "---"
echo "Vicuna Baseline"
python3 get_throughput_results.py data/mt_bench/experiments/vicuna-13b-baseline1.jsonl data/mt_bench/experiments/vicuna-13b-baseline2.jsonl data/mt_bench/experiments/vicuna-13b-baseline3.jsonl
echo "---"
echo "Prompt Decoding"
python3 get_throughput_results.py data/mt_bench/experiments/vicuna-13b-faster1.jsonl data/mt_bench/experiments/vicuna-13b-faster2.jsonl data/mt_bench/experiments/vicuna-13b-faster3.jsonl