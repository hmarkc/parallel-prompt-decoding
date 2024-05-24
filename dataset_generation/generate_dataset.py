from prompt.utils import *
import argparse
import math
from transformers import LlamaForCausalLM
from transformers import BitsAndBytesConfig

def generate_fine_tune_dataset(args):
  tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
        truncation=True
  )
  tokenizer.pad_token = tokenizer.unk_token
  data = get_finetune_dataset(tokenizer=tokenizer, data_path=args.data_path, size=args.size, offset=args.num_special_tokens+1)

  torch.save(data, f"{args.save_path}_{args.num_special_tokens}_finetune_{args.model_max_length}.pt")


def generate_self_distillation_dataset(args):
  # Set RoPE scaling factor
  config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
  orig_ctx_len = getattr(config, "max_position_embeddings", None)
  if orig_ctx_len and args.model_max_length > orig_ctx_len:
      scaling_factor = float(math.ceil(args.model_max_length / orig_ctx_len))
      config.rope_scaling = {"type": "linear", "factor": scaling_factor}
  config.use_cache = False

  config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
  
  quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
  )
  
  if config.model_type == "llama":
      model = LlamaForCausalLM.from_pretrained(
          args.model_name_or_path,
          config=config,
          low_cpu_mem_usage=True,
          quantization_config=quantization_config,
      )
  else:
      raise ValueError("Only support llama for now")
  data = get_self_distillation_dataset(model=model, data_path=args.data_path, num_special_tokens=args.num_special_tokens)

  model_name = args.model_name_or_path.split("/")[-1]
  torch.save(data, f"{args.save_path}_{args.num_special_tokens}_{model_name}_distillation__{args.model_max_length}.pt")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name_or_path", type=str, default="lmsys/vicuna-7b-v1.3")
  parser.add_argument("--model_max_length", type=int, default=2048)
  parser.add_argument("--save_path", type=str, default="data/ShareGPT_training_dataset")
  parser.add_argument("--data_path", type=str, default="data/ShareGPT_training_dataset_2.pt")
  parser.add_argument("--size", type=int, default=None)
  parser.add_argument("--num_special_tokens", type=int, default=2)
  parser.add_argument("--dataset_type", type=str, default="finetune", choices=["finetune", "distillation"])
  args = parser.parse_args()
  
  if args.dataset_type == "finetune":
    generate_fine_tune_dataset(args)
  elif args.dataset_type == "distillation":
    generate_self_distillation_dataset(args)
    