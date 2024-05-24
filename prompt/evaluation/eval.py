import os
import torch
import json
import numpy as np
from prompt.model.model import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# Once  
# 1st iteration: Once upon [a time]
# 2nd iteration: Once upon a [time there]
# 3rd iteration: Once upon a time [there was]
def get_accuracies(approximate_ids, logit):
    results = []
    _, _, num_special_tokens, _ = approximate_ids.shape
    for i in range(num_special_tokens):
        match = approximate_ids[:-1-i, :, i].eq(logit[1+i:, :, :1])
        results.append(match)
        # print(match.shape)
        # accuracy = match.any(dim=-1).sum().float() / (match.shape[0] * match.shape[1])
    # print(approximate_ids.shape, logit.shape)
    return results

def plot_accuracies(eval_data, save_path):
    plt.figure()
    for i, data in enumerate(eval_data):
        results= []
        for K in range(1, 11):
            results.append((data[:, :, :K].any(dim=-1).sum().float() / (data.shape[0] * data.shape[1])).cpu())
        plt.plot(results, label=f"{i}th prediction")
        print(f"{i+1}th accuracy - {', '.join(['Top '+str(i+1)+' : '+str(result.item()) for i, result in enumerate(results)])}")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks(range(10))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.legend()
    plt.show()
    plt.savefig(save_path)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = json.load(open(args.data_path))

    if args.eval_result_path:
        eval_data = torch.load(args.eval_result_path)
        plot_accuracies(eval_data, os.path.join(args.save_dir, args.model_name + "_accuracy.png"))
        return


    model = AutoPromptDecoder.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = model.tokenizer
    model = model.to(device)

    config = model.active_peft_config
    num_special_tokens = config.num_special_tokens
    virtual_tokens_per_special_token = config.virtual_tokens_per_special_token
    total_virtual_tokens = num_special_tokens * virtual_tokens_per_special_token
    # TODO: KV Cache
    results = None

    for sample in tqdm((data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        steps = args.steps
        logits_ids = []
        approximate_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).to(device)
            outputs = model(input_ids)
            logits = outputs.logits
            pred = torch.argmax(logits[:, -num_special_tokens-1, :], dim=-1)
            prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
            _, approximate_tokens = prompt_logits.topk(10, dim=-1)
            # print(pred.device, input_ids.device, model.device)
            preds = torch.cat((input_ids, pred.unsqueeze(0)), dim=-1)
            # print(f"Exact token: {tokenizer.batch_decode(pred)}, approximate tokens: {tokenizer.batch_decode(approximate_tokens.squeeze(0))}")
            logits_ids.append(preds[:, -1:].detach())
            approximate_ids.append(approximate_tokens.detach())
            for _ in range(steps):
                outputs= model(preds)
                logits = outputs.logits
                pred = torch.argmax(logits[:, -num_special_tokens-1, :], dim=-1)
                prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
                _, approximate_tokens = prompt_logits.topk(10, dim=-1)
                # print(f"Exact token: {tokenizer.batch_decode(pred)}, approximate tokens: {tokenizer.batch_decode(approximate_tokens.squeeze(0))}")
                preds = torch.cat((preds, pred.unsqueeze(0)), dim=-1)
                logits_ids.append(preds[:, -1:].detach())
                approximate_ids.append(approximate_tokens.detach())
            logits_ids = torch.stack(logits_ids, dim=0)
            approximate_ids = torch.stack(approximate_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(approximate_ids, logits_ids)
            else:
                # cat sub results
                cur_results = get_accuracies(approximate_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    save_path = os.path.join(args.save_dir, args.model_name + "_accuracy.pt")
    torch.save(results, save_path)
    plot_accuracies(results, os.path.join(args.save_dir, args.model_name + "_accuracy.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluator")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pre-trained model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the evaluation data in JSON format.")
    parser.add_argument("--save_dir", type=str, default="./",
                        help="Directory to save the results.")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of steps to run the model.")
    parser.add_argument("--eval_result_path", type=str, default=None, required=False,
                        help="Path to the evaluation result.")
    args = parser.parse_args()

    # If the save directory doesn't exist, create it
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)