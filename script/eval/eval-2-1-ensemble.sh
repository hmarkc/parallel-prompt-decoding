python3 prompt/evaluation/eval.py --model_path test/distillation/batch_4/prompt_vicuna-7b-v1.3_2_1_cl1024_ENSEMBLE_mean/\
                --model_name prompt_vicuna-7b-v1.3_2_1_cl1024_ENSEMBLE_mean_2048 \
                --data_path ./data/alpaca_eval.json \
                --save_dir ./log/eval/ \
                --steps 100
