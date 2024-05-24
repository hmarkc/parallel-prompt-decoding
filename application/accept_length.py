import os 
import json
import pandas as pd
import argparse
from tqdm import tqdm
import os 
import json
import pandas as pd
import argparse
from tqdm import tqdm

def get_throughput_results(input_file):
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]

    new_tokens = []
    wall_time = []
    throughputs = []
    for d in data:
        for choice in d["choices"]:
            new_tokens.extend(choice["new_tokens"])
            wall_time.extend(choice["wall_time"])
            for i in range(len(choice["new_tokens"])):
                throughputs.append(choice["new_tokens"][i] / choice["wall_time"][i])
    
    return sum(new_tokens) / sum(wall_time)
    # return sum(throughputs) / len(throughputs)

def main(model_name, dir_path, file_name, eval_file_name, max_length, min_length, length_interval, run_baseline, choices, n):
  max_throughput = 0
  # run baseline first 
  if run_baseline:
    for j in range(n):
      if os.system(f"python3 gen_model_answer_baseline.py --model-path ../test/{model_name} --model-id vicuna --answer-file {dir_path}/baseline_{j}.jsonl --bench-name alpaca_eval --max-new-token 20"):  
          raise ValueError("Failed to generate baseline")
  if choices is not None:
    r = eval(choices)
  else:
    r = range(min_length, max_length+1, length_interval)
  for i in tqdm(r):
    throughputs = []
    for j in range(n):
      # if file does not exist, generate it
      if not os.path.exists(f"{dir_path}/{file_name}{i}_{j}.jsonl"):
        # use alpaca dataset for evaluation
        if os.system(f"python3 {eval_file_name} --model-path ../test/{model_name} --model-id vicuna_faster --answer-file {dir_path}/{file_name}{i}_{j}.jsonl --tree-length {i} --bench-name alpaca_eval --max-new-token 20"): 
            raise ValueError("Failed to generate prompt decoding model")
    
      if os.path.exists(f"{dir_path}/{file_name}{i}_{j}.jsonl"):
        throughput = get_throughput_results(f"{dir_path}/{file_name}{i}_{j}.jsonl")
        throughputs.append(throughput)
    
    if len(throughputs) > 0:
      throughput = sum(throughputs) / len(throughputs)
      std = pd.Series(throughputs).std()
    
      if throughput > max_throughput:
        max_throughput = throughput
        best_sparse_tree = i
    
      print(f"Throughput for sparse tree {i}: {throughput:.3f} tokens/s", f"std: {std:.3f}")
      print(f"Best sparse tree: {best_sparse_tree}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--dir-path", type=str, default="data/mt_bench/dynamic_sparse_tree_search/3-1-7b", help="Path to the directory")
  parser.add_argument("--file-name", type=str, default="dynamic_sparse_tree", help="Name of the file")
  parser.add_argument("--model-name", type=str, default="vicuna-7b-3-1", help="Name of the model")
  parser.add_argument("--eval-file-name", type=str, default="gen_model_answer_prompt_decoding.py", help="Name of the evaluation file")
  parser.add_argument("--max-length", type=int, default=100, help="Max length of the sparse tree")
  parser.add_argument("--min-length", type=int, default=60, help="Min length of the sparse tree")
  parser.add_argument("--length-interval", type=int, default=1, help="Interval of the length of the sparse tree")
  parser.add_argument("--run-baseline", action="store_true", help="Run baseline first")
  parser.add_argument("--choices", type=str, default=None, help="Choices for the prompt decoding model")
  parser.add_argument("--n", type=int, default=1, help="Number of files to group")
  args = parser.parse_args()

  main(model_name=args.model_name, 
       dir_path=args.dir_path, 
       file_name=args.file_name, 
       eval_file_name=args.eval_file_name, 
       max_length=args.max_length, 
       min_length=args.min_length, 
       length_interval=args.length_interval,
       run_baseline=args.run_baseline,
       choices=args.choices,
        n=args.n)

  
  
  
  