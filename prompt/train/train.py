from dataclasses import dataclass, field
import warnings
import math
import pathlib
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother

from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from prompt.utils import *
from prompt.model.model import PromptDecoder, PromptConfig, AutoPromptDecoder
from prompt.model.modeling_llama_custom import LlamaForCausalLM as CustomLlamaForCausalLM
from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict   

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class ParamEfficientFineTuner(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        num_special_tokens = self.model.model.active_peft_config.num_special_tokens
        if torch.any(inputs["input_ids"][:, -1] == self.tokenizer.eos_token_id):
            warnings.warn("Input ends with EOS token.")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits

        loss = 0
        # Calculate loss on the prompt tokens
        prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
        prompt_labels = labels[..., -num_special_tokens:].contiguous()
        prompt_labels = prompt_labels.to(logits.device)
        pdd_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        decay_coefficient = 0.8
        for i in range(num_special_tokens):
            pdd_loss += loss_fn(prompt_logits[:, i, :], prompt_labels[:, i]) * (decay_coefficient ** i)
        if num_special_tokens > 0:
            pdd_loss /= num_special_tokens
        
        # Calculate loss on the rest of the tokens
        # rest_logits = logits[:, :-num_special_tokens, :].contiguous()
        # rest_labels = labels[..., :-num_special_tokens].contiguous()
        # rest_labels = rest_labels.to(logits.device)
        # rest_loss = loss_fn(rest_logits.view(-1, rest_logits.size(-1)), rest_labels.view(-1))
        
        # Combine the two losses, with a coefficient 0.2, same as Medusa-2
        loss = 0.2 * pdd_loss 
            
        return (loss, outputs) if return_outputs else loss


class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        num_special_tokens = self.model.active_peft_config.num_special_tokens
        if torch.any(inputs["input_ids"][:, -1] == self.tokenizer.eos_token_id):
            warnings.warn("Input ends with EOS token.")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        logits = outputs.logits

        # Calculate loss on the prompt tokens
        prompt_logits = logits[:, -num_special_tokens:, :].contiguous()
        prompt_labels = labels.contiguous()
        prompt_labels = prompt_labels.to(logits.device)
        loss = 0
        decay_coefficient = 0.8
        for i in range(num_special_tokens):
            loss_i = F.kl_div(
                F.log_softmax(prompt_logits[:, i, :], dim=-1),
                F.softmax(prompt_labels[:, i, :], dim=-1),
                reduction='sum'
            ) / prompt_logits.shape[0]
            loss += loss_i * (decay_coefficient ** i)
        if num_special_tokens > 0:
            loss /= num_special_tokens
        return (loss, outputs) if return_outputs else loss
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="lmsys/vicuna-7b-v1.3")
    num_special_tokens: int = field(default=1)
    virtual_tokens_per_special_token: int = field(default=1)
    use_custom_lm_head: bool = field(default=False)
    use_prefix_tuning: bool = field(default=False)
    prefix_virtual_tokens: int = field(default=10)
    vt_attention_type: str = field(default="decoder")
    aggregation_type: str = field(default="mean")
    num_exits: int = field(default=1)
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the saved dataset."},
    )
    size: Optional[int] = field(
        default=None, metadata={"help": "Number of examples to use."}
    )
    use_chunked: bool = field(
        default=False, metadata={"help": "Whether to use chunked dataset."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str = field(default=None)
    optim: str = field(default="adamw_torch")
    trainer_type: str = field(default="param_efficient_fine_tuner", metadata={"help": "Trainer type: param_efficient_fine_tuner, distillation_trainer"})
    stage1_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the stage 1 model."},
    )
    lm_head_lr_multiplier: float = field(default=0.1)
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("load_in_4_bits", model_args.load_in_4bit)

    # Create model
    peft_config = PromptConfig(
        tokenizer_name_or_path=model_args.model_name_or_path,
        base_model_name_or_path=model_args.model_name_or_path,
        num_special_tokens=model_args.num_special_tokens,
        virtual_tokens_per_special_token=model_args.virtual_tokens_per_special_token,
        use_prefix_tuning=model_args.use_prefix_tuning,
        prefix_virtual_tokens=model_args.prefix_virtual_tokens,
        vt_attention_type=VTAttentionType.from_str(model_args.vt_attention_type),
        aggregation_type=AggregationType.from_str(model_args.aggregation_type),
        use_custom_lm_head=model_args.use_custom_lm_head,
        num_exits=model_args.num_exits,
    )
    if training_args.stage1_model_path:
        model = AutoPromptDecoder.from_pretrained(
            training_args.stage1_model_path,
            low_cpu_mem_usage=True,
            cache_dir=training_args.cache_dir,
            quantization_config=quantization_config if model_args.load_in_4bit else None,
            new_config=peft_config,
        )
    else:
        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.use_cache = False

        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )
        
        if config.model_type == "llama":
            base_model = CustomLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config if model_args.load_in_4bit else None,
                # load_in_4bit=model_args.load_in_4bit,
                # load_in_8bit=model_args.load_in_8bit,
            )
        else:
            raise ValueError("Only support llama for now")
        # lora_config = LoraConfig(
        #     r=0.5,
        #     lora_alpha=0.5,
        #     lora_dropout=0.075,
        #     task_type="CAUSAL_LM",
        # )
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            # target_modules=["q", "v"],
            lora_dropout=0.01,
        )
        # TODO: LORA + PPD doesn't work well, need modification of PPD
        for param in base_model.base_model.parameters():
            param.requires_grad = False
        prompt_model = PromptDecoder(base_model, peft_config)
        prompt_model.print_trainable_parameters()
        model  = get_peft_model(prompt_model, lora_config)
        # make prompt model trainable
        for n, param in prompt_model.named_parameters():
            if 'prompt_encoder' in n:
                print(n)
                param.requires_grad = True
    model.print_trainable_parameters()
    set_peft_model_state_dict(model, get_peft_model_state_dict(model))
    state_dict = model.model.get_peft_model_state_dict()
    params = [v.cpu().numpy() for _, v in state_dict.items()]
    keys = state_dict.keys()
    params_dict = zip(keys, params)
    from collections import OrderedDict
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.model.set_peft_model_state_dict(state_dict)


    # Output dir
    training_args.output_dir = f"{training_args.output_dir}/prompt_{model_args.model_name_or_path.split('/')[-1]}_{model_args.num_special_tokens}_{model_args.virtual_tokens_per_special_token}_cl{training_args.model_max_length}_{model_args.vt_attention_type.upper()}_{model_args.aggregation_type}{'_custom_lm_head' if model_args.use_custom_lm_head else ''}{'_prefix' + str(model_args.prefix_virtual_tokens) if model_args.use_prefix_tuning else ''}_exits{model_args.num_exits}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        truncation=True
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # Load data
    if data_args.use_chunked:
        data = ChunkDataset(data_args.dataset_path)
    else:
        data = torch.load(data_args.dataset_path)
        data.set_size(data_args.size)

    # Set up optimizer 
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (p.requires_grad and "lm_head" in n)
            ],
            "lr": training_args.learning_rate * training_args.lm_head_lr_multiplier,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (p.requires_grad and "prompt_encoder" in n)
            ],
            "lr": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
        },
    ]
    optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optim_cls(optimizer_grouped_parameters, **optim_kwargs)
    
    # Start trainner
    if training_args.trainer_type == "distillation_trainer":
        trainer = DistillationTrainer(
            model=model, tokenizer=tokenizer, args=training_args, train_dataset=data, eval_dataset=None, optimizers=(optimizer, None)
        )
    elif training_args.trainer_type == "param_efficient_fine_tuner":
        trainer = ParamEfficientFineTuner(
            model=model, tokenizer=tokenizer, args=training_args, train_dataset=data, eval_dataset=None, optimizers=(optimizer, None)
        )
    else: 
        raise ValueError(f"Trainer type {training_args.trainer_type} not supported.")

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.save_pretrained(training_args.output_dir)
    # Save ppd
    print(type(model.model))
    model.model.save_pretrained(f"{training_args.output_dir}/ppd")

if __name__ == "__main__":
    train()