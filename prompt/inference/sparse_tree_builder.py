
import numpy as np
import torch

import argparse

from pprint import pprint
from copy import deepcopy
import numpy as np
import concurrent.futures


def load_accuracy(file_name):
  eval_data = torch.load(file_name)
  accuracies = []
  for i, data in enumerate(eval_data):
      results= []
      for K in range(1, 11):
          results.append((data[:, :, :K].any(dim=-1).sum().float() / (data.shape[0] * data.shape[1])).cpu())
      print(f"{i+1}th accuracy - {', '.join(['Top '+str(i+1)+' : '+str(result.item()) for i, result in enumerate(results)])}")
      accuracy = [results[0]]
      for i in range(1, len(results)):
          accuracy.append(results[i] - results[i-1])
      accuracies.append(accuracy)
  accuracies = torch.tensor(accuracies)
  return accuracies


def expected_accuracy(es, vs, print_output=False):
    # only calculate depth of 3 
    e10, e20, e30 = es[0]
    e11, e21, e31 = es[1]
    e12, e22, e32 = es[2]
    e13, e23, e33 = es[3]
    v1, v2, v3 = vs
    a = np.array([[0, e11-1, e21, e31], 
                  [0, e12, e22-1, e32], 
                  [1, e13, e23, e33-1],
                  [1, 1, 1, 1]])
    output = np.linalg.solve(a, [0, 0, 0, 1])
    if print_output:
        print("Expected Probability:")
        pprint(output)
    return output[1]*v1 + output[2]*v2 + output[3]*v3


def find_all_candidates(parent, width, depth):
    if depth == 0:
        return []
    candidates = []
    for i in range(width):
        candidates.append(parent+[i])
        candidates.extend(find_all_candidates(parent+[i], width, depth-1))
    return candidates


def find_optimal_sparse_tree(accuracies, num_candidates, max_depth=None):
    # Generate all possible candidates of varying lengths
    if max_depth:
        accuracies = accuracies[:max_depth]
    candidates = find_all_candidates([], accuracies.shape[1], accuracies.shape[0])
    
    # Calculate cumulative accuracy for each candidate
    candidate_accuracies = []
    for candidate in candidates:
        cumulative_accuracy = 1.0
        for idx, top_i in enumerate(candidate):
            cumulative_accuracy *= accuracies[idx, top_i]
        candidate_accuracies.append((cumulative_accuracy, candidate))
    
    # Sort candidates by their cumulative accuracy in descending order and select top n
    top_candidates = sorted(candidate_accuracies, key=lambda x: x[0], reverse=True)[:num_candidates]
    
    # Extract just the candidate paths
    top_candidate_paths = [list(candidate) for _, candidate in top_candidates]
    top_candidate_accs = [round(acc.cpu().item(), 5) for acc, _ in top_candidates]
    
    return top_candidate_paths, top_candidate_accs


def find_optimal_extended_sparse_tree(accuracies, input_length_limit):
    # input_length_limit = num_candidates + sum(candidate_accuracy_n * num_special_tokens_n)
    # n is the depth of accuracies 
    n = accuracies.shape[0]
    # generate and store the optimal sparse tree for each num_candidates and each depth
    optimal_sparse_trees = {}
    optimal_sparse_tree_accuracies = {}
    for depth in range(1, n+1):
        candidates, accs = find_optimal_sparse_tree(accuracies, input_length_limit, depth)
        candidate_acc_pairs = list(zip(candidates, accs))
        for length in range(1, input_length_limit+1):
            ls = []
            for candidate, acc in candidate_acc_pairs[:length]: 
                sum_children_acc = sum([a for c, a in candidate_acc_pairs[:length] if c[:-1] == candidate])
                ls.append((candidate, acc - sum_children_acc))
            optimal_sparse_trees[(depth, length)] = ls
        for size in range(1, input_length_limit+1):
            optimal_sparse_tree_accuracies[(depth, size)] = sum(accs[:size])
    best_sparse_trees = None 
    best_expected_acc = -1 
    best_num_tree_nodes = None
    # only calculate depth of 3 for now
    for tree_node2 in range(input_length_limit//(n+1), input_length_limit//2 + 1): 
        for tree_node3 in range(input_length_limit//(n+1), input_length_limit//2 + 1):
            tree_nodes = {1: 10, 2: tree_node2, 3: tree_node3}
            sparse_trees, expected_acc = find_extended_sparse_tree_fixed_tree_nodes(tree_nodes, input_length_limit, optimal_sparse_tree_accuracies, deepcopy(optimal_sparse_trees), n)
            if expected_acc > best_expected_acc:
                best_sparse_trees = sparse_trees
                best_expected_acc = expected_acc
                best_num_tree_nodes = tree_nodes
    print("Input limit", input_length_limit, "Tree nodes:", best_num_tree_nodes, "Expected Acc:", best_expected_acc)
    return best_sparse_trees


def find_extended_sparse_tree_fixed_tree_nodes(tree_nodes, input_length_limit, optimal_sparse_tree_accuracies, candidate_acc_pairs, n):
    optimal_sparse_trees = {}
    for depth in range(1, n+1):
        num_nodes = n * len(candidate_acc_pairs[(depth, tree_nodes[depth])])
        optimal_sparse_trees[depth] = [[candidate, acc, n] for candidate, acc in candidate_acc_pairs[(depth, tree_nodes[depth])]]
        while num_nodes > input_length_limit - tree_nodes[depth]:
            min_accuracy_loss = float('-inf')
            min_index = 0
            for i, (_, candidate_acc, num_special_token) in enumerate(optimal_sparse_trees[depth]):
                if num_special_token == 1:
                    continue
                accuracy_loss = candidate_acc * (optimal_sparse_tree_accuracies[(num_special_token-1, tree_nodes[num_special_token-1])] - \
                        optimal_sparse_tree_accuracies[(num_special_token, tree_nodes[num_special_token])])
                if accuracy_loss > min_accuracy_loss:
                    min_accuracy_loss = accuracy_loss
                    min_index = i
            if min_accuracy_loss == float('inf'):
                break
            optimal_sparse_trees[depth][min_index][2] = optimal_sparse_trees[depth][min_index][2] - 1
            num_nodes -= 1
    
    es = []
    vs = []
    for depth in range(1, n+1):
        e_depth = [0] * (n+1) 
        for (_, candidate_acc, num_special_token) in optimal_sparse_trees[depth]:
            e_depth[num_special_token] += candidate_acc
        e_depth[0] = 1 - sum(e_depth[1:])
        es.append(e_depth)
        vs.append(optimal_sparse_tree_accuracies[(depth, tree_nodes[depth])])
    es = np.array(es).T.tolist()
    acc = expected_accuracy(es, vs)
    
    return optimal_sparse_trees, acc
    
    
def sparse_tree_info(best_sparse_trees):
  print("Best Sparse Trees:")
  pprint(best_sparse_trees)

  es = []
  vs = []
  for depth, best_sparse_tree in best_sparse_trees.items():
      print(f"Depth: {depth}")
      print("Number of tree nodes:", len(best_sparse_tree))
      print("Number of special tokens:", sum([num_special_token for _, _, num_special_token in best_sparse_tree]))
      acc_1 = sum([accuracy for _, accuracy, num_special_token in best_sparse_tree if num_special_token == 1])
      acc_2 = sum([accuracy for _, accuracy, num_special_token in best_sparse_tree if num_special_token == 2])
      acc_3 = sum([accuracy for _, accuracy, num_special_token in best_sparse_tree if num_special_token == 3])
      print("Probabilities to 1 special token:", acc_1)
      print("Probabilities to 2 special tokens:", acc_2)
      print("Probabilities to 3 special tokens:", acc_3)
      print("Probability to None", 1 - (acc_1 + acc_2 + acc_3))


def write_sparse_tree_to_file(file_name, min_input_length, max_input_length, accuracies):
  def task(i):
      # This is the task that will be executed by each thread.
      # It returns a tuple of (i, result) so we know which iteration it belongs to.
      return i, find_optimal_extended_sparse_tree(accuracies, i)

  # Prepare to collect the results
  results = []

  # Using ThreadPoolExecutor to execute tasks concurrently
  with concurrent.futures.ThreadPoolExecutor() as executor:
      # Map the task function to the range of values concurrently
      future_to_i = {executor.submit(task, i): i for i in range(min_input_length, max_input_length+1)}
      
      for future in concurrent.futures.as_completed(future_to_i):
          i = future_to_i[future]
          try:
              # Collect results as they are completed
              results.append(future.result())
          except Exception as exc:
              print(f'Generated an exception: {exc}')

  # Sorting results to ensure they are in order
  results.sort(key=lambda x: x[0])

  # Writing results to file sequentially
  with open(file_name, "w") as f:
      for i, result in results:
          f.write(f"# Dynamic Sparse Trees for input length limit of {i}\n")
          f.write(f"dynamic_sparse_trees_{i} = {result}\n")


def main(args):
  accuracies = load_accuracy(args.accuracies_file)
  write_sparse_tree_to_file(args.output_file, args.min_input_length, args.max_input_length, accuracies)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--accuracies-file", type=str, required=True, help="Path to the accuracies file.")
  parser.add_argument("--output-file", type=str, required=True, help="Path to the output file.")
  parser.add_argument("--min-input-length", type=int, required=True, help="Minimum input length limit.")
  parser.add_argument("--max-input-length", type=int, required=True, help="Maximum input length limit.")
  args = parser.parse_args()
  main(args)


