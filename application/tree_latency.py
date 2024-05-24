"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time
import transformers

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

import sys

from prompt.model.kv_cache import initialize_past_key_values
from prompt.model.model import AutoPromptDecoder, PromptDecoder, PromptConfig
from prompt.model.modeling_llama_custom import LlamaForCausalLM as CustomLlamaForCausalLM
from human_eval.data import write_jsonl, read_problems
from datasets import load_dataset


def infer(input_ids, model, tokenizer, tree_length, max_steps = 512, temperature=0.7, posterior_threshold = 0.09, posterior_alpha = 0.3, sampling='greedy', max_new_token=1024):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    print('Generate buffers')
    model.generate_dynamic_buffers(tree_length)
    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    logits, prompt_logits = model.start_inference(input_ids, past_key_values, current_length_data)
    new_token = 0
    
    torch.cuda.synchronize()
    start = time.time()
    for idx in range(max_steps): 
        candidates, tree_candidates_embeds = model.generate_candidates(
            logits, 
            prompt_logits, 
            temperature, 
            posterior_threshold, 
            posterior_alpha, 
            sampling)
        logits, all_logits = model.tree_decoding(tree_candidates_embeds, past_key_values, input_ids)
        best_candidate, accept_length = model.evaluate_posterior(
            logits, 
            candidates, 
            temperature, 
            posterior_threshold, 
            posterior_alpha,
            sampling)
        input_ids, logits, prompt_logits, new_token = model.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                logits,
                all_logits,
                new_token,
                past_key_values_data,
                current_length_data,
        )

        # if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
        #     break
        # if new_token > max_new_token:
        #     break
    torch.cuda.synchronize()
    end = time.time()
    
    model.end_inference()
        
    return input_ids, new_token, idx, end - start


def infer_baseline(input_ids, model, tokenizer, choices, max_steps = 512, temperature=0.7, posterior_threshold = 0.09, posterior_alpha = 0.3, sampling='greedy', max_new_token=1024):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    print('Initialize past key values')
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    model.base_model.model.tree_mask = None
    model.base_model.model.vt_attention_mask = None
    model.base_model.model.prompt_token_indices = None
    outputs = model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
    new_token = 0
    
    torch.cuda.synchronize()
    start = time.time()
    
    for idx in range(max_steps): 
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = model.base_model(input_id, use_cache=True, past_key_values = past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)
        new_token += 1

        # if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
        #     break
        # if new_token > max_new_token:
        #     break
        
    torch.cuda.synchronize()
    end = time.time()
        
    return input_ids, new_token, idx, end - start


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    benchname,
    warmup,
    max_tree_length,
    min_tree_length,
    length_interval,
    temperature,
    posterior_threshold,
    posterior_alpha,
    sampling
):
    if benchname == 'mt_bench':
        questions = load_questions(question_file, question_begin, question_end)
    elif benchname == 'humaneval':
        questions = read_problems()
        questions = list(questions.values())[question_begin:question_end]
    elif benchname == 'alpaca_eval':
        questions = json.load(open(question_file))
    elif benchname == 'gsm8k':
        # only use the first 1000 questions from test set
        questions = load_dataset('gsm8k', 'main', streaming=False, split='test')['question'][:500] 
    else:
        raise ValueError("Unknown benchmark name")
        
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                benchname=benchname,
                warmup=warmup,
                max_tree_length=max_tree_length,
                min_tree_length=min_tree_length,
                length_interval=length_interval,
                temperature=temperature,
                posterior_threshold=posterior_threshold,
                posterior_alpha=posterior_alpha,
                sampling=sampling
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    benchname,
    warmup,
    max_tree_length,
    min_tree_length,
    length_interval,
    temperature,
    posterior_threshold,
    posterior_alpha,
    sampling
):
    model = AutoPromptDecoder.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        torch_dtype=torch.float16,
        # device_map="auto"
    )
    model.cuda()
    tokenizer = model.tokenizer
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)  
    
    tree_length = eval("dynamic_sparse_trees_" + str(max_tree_length))
    for i in range(warmup):
        torch.manual_seed(0)
        question = questions[i]
        if benchname == 'mt_bench':
            num_turns = len(question["turns"])
        elif benchname == 'humaneval' or benchname == 'alpaca_eval' or benchname == 'gsm8k':
            num_turns = 1
        conv = get_conversation_template(model_id)
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(num_turns):
            if benchname == 'mt_bench':
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            elif benchname == 'humaneval':
                qs = question["prompt"]
                prompt = qs
            elif benchname == 'alpaca_eval':
                conv = get_conversation_template(model_id)
                conv.messages = []
                conv.append_message(conv.roles[0], question["instruction"])
                conv.append_message(conv.roles[1], "")
                prompt = conv.get_prompt()
            elif benchname == 'gsm8k':
                qs = question
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
            input_ids = tokenizer([prompt]).input_ids

            # try:
            # torch.cuda.synchronize()
            # start_time = time.time()
            output_ids, new_token, idx, total_time = infer(
                torch.as_tensor(input_ids).cuda(),
                model,
                tokenizer,
                tree_length,
                temperature=0,
                posterior_threshold=posterior_threshold,
                posterior_alpha=posterior_alpha,
                sampling=sampling,
                max_new_token=max_new_token
            )
            # torch.cuda.synchronize()
            # total_time = time.time() - start_time
            if benchname == 'mt_bench':
                output_ids = output_ids[0][len(input_ids[0]) :]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
            
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()    
                conv.messages[-1][-1] = output
                
            elif benchname == 'humaneval' or benchname == 'alpaca_eval' or benchname == 'gsm8k':
                output = tokenizer.decode(
                    output_ids[0].tolist(),
                    spaces_between_special_tokens=False,
                )
            # except RuntimeError as e:
            #     print(e)
            #     print("ERROR question ID: ", question["question_id"])
            #     output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
    print('Warmup done', warmup, 'steps')
    
    r = [75, 105, 135, 165, 195, 225, 255, 285]
    # r = range(min_tree_length, max_tree_length, length_interval)
    for tl in tqdm(r):
        if tl > 0:
            tree_length = eval("dynamic_sparse_trees_" + str(tl))
        # warmup

        for i in range(1):
            question = questions[0]
            if benchname == 'mt_bench':
                question_id = question["question_id"]
                num_turns = len(question["turns"])
            elif benchname == 'humaneval':
                question_id = question["task_id"]
                num_turns = 1
            elif benchname == 'alpaca_eval' or benchname == 'gsm8k':
                question_id = i
                num_turns = 1
                
            if "category" in question and question["category"] in temperature_config:
                temperature = temperature_config[question["category"]]
            else:
                print(f"Unknown category, using default temperature 0.0")
                temperature = 0.0

            choices = []
            for i in range(num_choices):
                torch.manual_seed(0)
                conv = get_conversation_template(model_id)
                turns = []
                idxs = []
                new_tokens = []
                wall_time = []
                for j in range(num_turns):
                    if benchname == 'mt_bench':
                        qs = question["turns"][j]
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                    elif benchname == 'humaneval':
                        qs = question["prompt"]
                        prompt = qs
                    elif benchname == 'alpaca_eval':
                        conv.messages = []
                        conv.append_message(conv.roles[0], question["instruction"])
                        conv.append_message(conv.roles[1], "")
                        prompt = conv.get_prompt()
                    elif benchname == 'gsm8k':
                        qs = question
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                    prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Hi, could you share a tale about a charming llama that grows Medusa-like hair and starts its own coffee shop? ASSISTANT:"
                    input_ids = tokenizer([prompt]).input_ids

                    try:
                        # torch.cuda.synchronize()
                        # start_time = time.time()
                        if tl > 0:
                            output_ids, new_token, idx, total_time = infer(
                                torch.as_tensor(input_ids).cuda(),
                                model,
                                tokenizer,
                                tree_length,
                                temperature=temperature,
                                posterior_threshold=posterior_threshold,
                                posterior_alpha=posterior_alpha,
                                sampling=sampling,
                                max_new_token=max_new_token
                            )
                        else: 
                            output_ids, new_token, idx, total_time = infer_baseline(
                                torch.as_tensor(input_ids).cuda(),
                                model,
                                tokenizer,
                                dynamic_sparse_trees_60,
                                temperature=temperature,
                                posterior_threshold=posterior_threshold,
                                posterior_alpha=posterior_alpha,
                                sampling=sampling,
                                max_new_token=max_new_token
                            )
                        # torch.cuda.synchronize()
                        # total_time = time.time() - start_time
                        if benchname == 'mt_bench':
                            
                            # if model.config.is_encoder_decoder:
                            #     output_ids = output_ids[0]
                            # else:
                            output_ids = output_ids[0][len(input_ids[0]) :]

                            # be consistent with the template's stop_token_ids
                            if conv.stop_token_ids:
                                stop_token_ids_index = [
                                    i
                                    for i, id in enumerate(output_ids)
                                    if id in conv.stop_token_ids
                                ]
                                if len(stop_token_ids_index) > 0:
                                    output_ids = output_ids[: stop_token_ids_index[0]]

                            output = tokenizer.decode(
                                output_ids,
                                spaces_between_special_tokens=False,
                            )
                            if conv.stop_str and output.find(conv.stop_str) > 0:
                                output = output[: output.find(conv.stop_str)]
                            for special_token in tokenizer.special_tokens_map.values():
                                if isinstance(special_token, list):
                                    for special_tok in special_token:
                                        output = output.replace(special_tok, "")
                                else:
                                    output = output.replace(special_token, "")
                            output = output.strip()
                            
                            if conv.name == "xgen" and output.startswith("Assistant:"):
                                output = output.replace("Assistant:", "", 1).strip()
                            conv.messages[-1][-1] = output
                            
                        elif benchname == 'humaneval' or benchname == 'alpaca_eval' or benchname == 'gsm8k':
                            output = tokenizer.decode(
                                output_ids[0].tolist(),
                                spaces_between_special_tokens=False,
                            )

                    except RuntimeError as e:
                        print("ERROR question ID: ", question["question_id"])
                        print(e)
                        output = "ERROR"

                    turns.append(output)
                    idxs.append(int(idx))
                    new_tokens.append(int(new_token))
                    wall_time.append(total_time)
                # torch.cuda.empty_cache()
                choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

            # Dump answers
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "tree_length": tl,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["tree_length"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--warmup",
        type=int, 
        default=3,
        help="The number of warmup steps.",
    )
    parser.add_argument(
        "--max-tree-length",
        type=int,
        default=100,
        help="The choices for sampling.",
    )
    parser.add_argument(
        "--min-tree-length",
        type=int,
        default=60,
        help="The choices for sampling.",
    )
    parser.add_argument(
        "--length-interval",
        type=int,
        default=3,
        help="The choices for sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--posterior_threshold",
        type=float,
        default=0.09,
        help="The threshold for posterior sampling.",
    )
    parser.add_argument(
        "--posterior_alpha",
        type=float,
        default=0.3,
        help="The alpha for posterior sampling.",
    )
    parser.add_argument(
        "--sampling", 
        type=str,
        default='greedy',
        help="The sampling method for decoding."
    )

    args = parser.parse_args()

    if 'vicuna-13b' in args.model_path.lower():
        if '-2-' in args.model_path:
            from prompt.inference.dynamic_sparse_trees_2_13b import *
            print('Using 13b sparse trees')
        elif '-3-' in args.model_path:
            from prompt.inference.dynamic_sparse_trees_3_vicuna_13b import *
            print('Using 13b 3-1 sparse trees')
        else:
            from prompt.inference.dynamic_sparse_trees_3_vicuna_13b import *
            print('Using 13b 3-1 sparse trees, this is the default because the model path does not contain -2- or -3-')
        # args.tree_length = eval("dynamic_sparse_trees_" + args.tree_length)
    elif 'vicuna-7b' in args.model_path.lower():
        if '-2-' in args.model_path:
            from prompt.inference.dynamic_sparse_trees_2_7b import *
            print('Using 7b 2-1 sparse trees')
        elif '-3-' in args.model_path:
            from prompt.inference.dynamic_sparse_trees_3_vicuna_7b import *
            print('Using 7b 3-1 sparse trees')
        else:
            from prompt.inference.dynamic_sparse_trees_3_vicuna_7b import *
            print('Using 7b 3-1 sparse trees, this is the default because the model path does not contain -2- or -3-')
        # args.tree_length = eval("dynamic_sparse_trees_" + args.tree_length)
    elif 'mobilellama' in args.model_path.lower():
        from prompt.inference.dynamic_sparse_trees_3_MobileLLaMA import *
        print('Using MobileLLaMA 3-1 sparse trees')
        # args.tree_length = eval("dynamic_sparse_trees_" + args.tree_length)
    else:
        raise ValueError("Unknown model path")
    
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = None
    if args.bench_name == 'mt_bench':
        question_file = f"data/{args.bench_name}/question.jsonl"
    elif args.bench_name == 'alpaca_eval':
        question_file = f"data/{args.bench_name}/alpaca_eval.json"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file_name = args.model_id+"-temperature-"+str(args.temperature)+"-posterior_threshold-"+str(args.posterior_threshold)+"-posterior_alpha-"+str(args.posterior_alpha)+"-sampling-"+args.sampling
        answer_file = f"data/{args.bench_name}/model_answer/{answer_file_name}.jsonl"

    print(f"Output to {answer_file}")
    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        benchname=args.bench_name,
        warmup=args.warmup,
        max_tree_length=args.max_tree_length,
        min_tree_length=args.min_tree_length,
        length_interval=args.length_interval,
        temperature=args.temperature,
        posterior_threshold=args.posterior_threshold,
        posterior_alpha=args.posterior_alpha,
        sampling=args.sampling
    )

    reorg_answer_file(answer_file)
