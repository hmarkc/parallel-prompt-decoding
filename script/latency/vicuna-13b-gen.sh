# MT bench, temperature sampling
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline1 --answer-file data/mt_bench/experiments/vicuna-13b-baseline1.jsonl --bench-name mt_bench
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline2 --answer-file data/mt_bench/experiments/vicuna-13b-baseline2.jsonl --bench-name mt_bench
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline3 --answer-file data/mt_bench/experiments/vicuna-13b-baseline3.jsonl --bench-name mt_bench
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster1 --answer-file data/mt_bench/experiments/vicuna-13b-faster1.jsonl --tree-length 60 --bench-name mt_bench
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster2 --answer-file data/mt_bench/experiments/vicuna-13b-faster2.jsonl --tree-length 60 --bench-name mt_bench
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster3 --answer-file data/mt_bench/experiments/vicuna-13b-faster3.jsonl --tree-length 60 --bench-name mt_bench
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster4 --answer-file data/mt_bench/experiments/vicuna-13b-faster4.jsonl --tree-length 60 --bench-name mt_bench
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster5 --answer-file data/mt_bench/experiments/vicuna-13b-faster5.jsonl --tree-length 60 --bench-name mt_bench

# MT bench, greedy sampling
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline1 --answer-file data/mt_bench/experiments/vicuna-13b-baseline1-greedy.jsonl --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline2 --answer-file data/mt_bench/experiments/vicuna-13b-baseline2-greedy.jsonl --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline3 --answer-file data/mt_bench/experiments/vicuna-13b-baseline3-greedy.jsonl --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster1 --answer-file data/mt_bench/experiments/vicuna-13b-faster1-greedy.jsonl --tree-length 60 --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster2 --answer-file data/mt_bench/experiments/vicuna-13b-faster2-greedy.jsonl --tree-length 60 --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster3 --answer-file data/mt_bench/experiments/vicuna-13b-faster3-greedy.jsonl --tree-length 60 --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster4 --answer-file data/mt_bench/experiments/vicuna-13b-faster4-greedy.jsonl --tree-length 60 --bench-name mt_bench --temperature 0.0
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster5 --answer-file data/mt_bench/experiments/vicuna-13b-faster5-greedy.jsonl --tree-length 60 --bench-name mt_bench --temperature 0.0

# HumanEval
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline1 --answer-file data/humaneval/experiments/vicuna-13b-baseline1.jsonl --bench-name humaneval --max-new-token 512 
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline2 --answer-file data/humaneval/experiments/vicuna-13b-baseline2.jsonl --bench-name humaneval --max-new-token 512
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline3 --answer-file data/humaneval/experiments/vicuna-13b-baseline3.jsonl --bench-name humaneval --max-new-token 512
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster1 --answer-file data/humaneval/experiments/vicuna-13b-faster1.jsonl --tree-length 60 --bench-name humaneval --max-new-token 512
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster2 --answer-file data/humaneval/experiments/vicuna-13b-faster2.jsonl --tree-length 60 --bench-name humaneval --max-new-token 512
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster3 --answer-file data/humaneval/experiments/vicuna-13b-faster3.jsonl --tree-length 60 --bench-name humaneval --max-new-token 512
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster4 --answer-file data/humaneval/experiments/vicuna-13b-faster4.jsonl --tree-length 60 --bench-name humaneval --max-new-token 512
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster5 --answer-file data/humaneval/experiments/vicuna-13b-faster5.jsonl --tree-length 60 --bench-name humaneval --max-new-token 512

# GSM8K
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline1 --answer-file data/gsm8k/experiments/vicuna-13b-baseline1.jsonl --bench-name gsm8k
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline2 --answer-file data/gsm8k/experiments/vicuna-13b-baseline2.jsonl --bench-name gsm8k
python3 gen_model_answer_baseline.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-13b-baseline3 --answer-file data/gsm8k/experiments/vicuna-13b-baseline3.jsonl --bench-name gsm8k
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster1 --answer-file data/gsm8k/experiments/vicuna-13b-faster1.jsonl --tree-length 60 --bench-name gsm8k
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster2 --answer-file data/gsm8k/experiments/vicuna-13b-faster2.jsonl --tree-length 60 --bench-name gsm8k
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster3 --answer-file data/gsm8k/experiments/vicuna-13b-faster3.jsonl --tree-length 60 --bench-name gsm8k
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster4 --answer-file data/gsm8k/experiments/vicuna-13b-faster4.jsonl --tree-length 60 --bench-name gsm8k
python3 gen_model_answer_prompt_decoding.py --model-path hmarkc/ppd-vicuna-13b-v1.3 --model-id vicuna-faster5 --answer-file data/gsm8k/experiments/vicuna-13b-faster5.jsonl --tree-length 60 --bench-name gsm8k
