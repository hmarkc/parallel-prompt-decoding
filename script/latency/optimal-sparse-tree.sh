#! /bin/bash
# use the alpaca eval dataset to find the optimal sparse tree
# Accept length for Vicuna 7b
python accept_length.py \
  --dir-path ./data/alpaca_eval/dynamic_sparse_tree_search/3-1-7b \
  --file-name dynamic_sparse_tree \
  --model-name hmarkc/ppd-vicuna-7b-v1.3 \
  --eval-file-name gen_model_answer_prompt_decoding.py \
  --run-baseline \
  --n 1 \
  --max-length 79 \
  --min-length 60 \
  --length-interval 9 \
  --choices "[5, 10, 20, 35, 60, 120, 200, 500]" \


# latency for Vicuna 7b
python3 tree_latency.py \
 --model-path hmarkc/ppd-vicuna-7b-v1.3 \
 --model-id vicuna_faster \
 --answer-file ./data/alpaca_eval/dynamic_sparse_tree_search/3-1-7b-21-04/tree_latency.jsonl \
 --bench-name alpaca_eval \
 --min-tree-length 60 \
 --max-tree-length 120 \
 --length-interval 3 \
 --max-new-token 1024

# Accept length for Vicuna 13b
python accept_length.py \
  --dir-path ./data/alpaca_eval/dynamic_sparse_tree_search/3-1-13b \
  --file-name dynamic_sparse_tree \
  --model-name hmarkc/ppd-vicuna-13b-v1.3\
  --eval-file-name gen_model_answer_prompt_decoding.py \
  --max-length 120 \
  --min-length 60 \
  --length-interval 3 \
  --n 1 \

# latency for Vicuna 13b
python3 tree_latency.py \
 --model-path hmarkc/ppd-vicuna-13b-v1.3 \
 --model-id vicuna_faster \
 --answer-file ./data/alpaca_eval/dynamic_sparse_tree_search/3-1-13b/tree_latency.jsonl \
 --bench-name alpaca_eval \
 --min-tree-length 60 \
 --max-tree-length 120 \
 --length-interval 3 \
 --max-new-token 1024

# Accept length for full sparse tree
# python accept_length.py \
#   --dir-path data/alpaca_eval/sparse_tree_search/3-1-7b/ \
#   --file-name full_sparse_tree \
#   --model-name hmarkc/ppd-vicuna-7b-v1.3 \
#   --eval-file-name gen_model_answer_full_sparse_tree.py \
#   --choices "[5, 10, 20, 35, 60, 120, 200, 500]" \
#   --n 1 \

# python accept_length.py \
  # --dir-path data/alpaca_eval/sparse_tree_search/3-1-13b/ \
  # --file-name sparse_tree \
  # --model-name hmarkc/ppd-vicuna-13b-v1.3 \
  # --eval-file-name gen_model_answer_full_sparse_tree.py \
  # --max-length 120 \
  # --min-length 60 \
  # --length-interval 3 \
  # --n 1 \

# Accept length for random sparse tree 
python accept_length.py \
  --dir-path data/alpaca_eval/random_tree_search/3-1-7b/ \
  --file-name random_sparse_tree \
  --model-name hmarkc/ppd-vicuna-7b-v1.3 \
  --eval-file-name gen_model_answer_random_sparse_tree.py \
  --choices "[5, 10, 20, 35, 60, 120, 200, 500]" \
  --n 1 \

# Accept length for MobileLLaMA
python accept_length.py \
  --dir-path ./data/alpaca_eval/dynamic_sparse_tree_search/MobileLLaMA \
  --file-name dynamic_sparse_tree \
  --model-name ../test/MobileLLaMA \
  --eval-file-name gen_model_answer_prompt_decoding.py \
  --n 1 \
  --max-length 79 \
  --min-length 60 \
  --length-interval 9 \
  --choices "[75, 105, 135, 165, 195, 225, 255, 285]" \


# latency for MobileLLaMA
python3 tree_latency.py \
 --model-path ../test/MobileLLaMA \
 --model-id MobileLLaMA \
 --answer-file ./data/alpaca_eval/dynamic_sparse_tree_search/MobileLLaMA/tree_latency.jsonl \
 --bench-name alpaca_eval \
 --min-tree-length 60 \
 --max-tree-length 285 \
 --length-interval 3 \
 --max-new-token 1024