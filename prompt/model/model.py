from peft import PeftModel, PeftType, PromptTuningConfig, PromptTuningInit, PrefixTuningConfig, PromptEncoderConfig, get_peft_model_state_dict, set_peft_model_state_dict
from prompt.model.modeling_llama_custom import LlamaForCausalLM as CustomLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Any, List, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field
from prompt.utils import *
from prompt.model.kv_cache import initialize_past_key_values
import torch 
import transformers
import json
import os

TOPK=10

@dataclass
class PromptConfig(PromptTuningConfig):
    """
    This class defines the configuration for the prompt decoding model.
    """
    num_special_tokens: int = 1

    virtual_tokens_per_special_token: int = 1

    use_cache: bool = True
    
    num_exits: int = 1
    
    use_custom_lm_head: bool = False

    use_prefix_tuning: bool = False

    prefix_virtual_tokens: Optional[int] = None

    vt_attention_type: VTAttentionType = VTAttentionType.DECODER
    
    aggregation_type: AggregationType = AggregationType.MEAN

    prompt_tuning_init: PromptTuningInit = field(
        default=PromptTuningInit.TEXT,
        metadata={"help": "How to initialize the prompt embedding"},
    )

    prompt_tuning_init_text: str = field(
        default=' '.join(["Next {} word".format(i+2) for i in range(num_special_tokens)]),
        metadata={"help": "The text to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"},
    )  

    def __post_init__(self):
        # have to do this as custom peft config type is not supported
        super().__post_init__()
        self.num_virtual_tokens = self.num_special_tokens * self.virtual_tokens_per_special_token


class PromptDecoder(PeftModel): 
  
    def __init__(self, model: torch.nn.Module, peft_config: PromptConfig, adapter_name: str='default') -> None:
        """
        This class defines the prompt decoding model.

        Args:
            model (torch.nn.Module): The base model to be used.
            peft_config (PromptConfig): The configuration for the prompt decoding.
            adapter_name (str, optional): The name of the adapter to be used. Defaults to 'default'. 
        """
        super().__init__(model, peft_config)
        # prevent peft wrapper to overwrite peft_config
        self.prompt_peft_config = {adapter_name: peft_config}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        self.base_model.model.prompt_token_indices = [-i for i in range(peft_config.num_virtual_tokens, 0, -1)]
        if peft_config.vt_attention_type == VTAttentionType.DECODER:
            # Every virtual token sees its predecessors
            vt_decoder_attention_mask = torch.zeros(peft_config.num_virtual_tokens, peft_config.num_virtual_tokens)
            for i in range(peft_config.num_virtual_tokens):
                vt_decoder_attention_mask[i, :i+1] = 1
            self.base_model.model.vt_attention_mask = vt_decoder_attention_mask
        elif peft_config.vt_attention_type == VTAttentionType.ENCODER:
            # Every virtual token sees all its neighbors (num_virtual_tokens_per_special_token)
            vt_encoder_attention_mask = torch.zeros(peft_config.num_virtual_tokens, peft_config.num_virtual_tokens)
            for i in range(peft_config.num_special_tokens):
                for j in range(peft_config.virtual_tokens_per_special_token):
                    vt_encoder_attention_mask[i*peft_config.virtual_tokens_per_special_token+j, 
                                              i*peft_config.virtual_tokens_per_special_token: (i+1)*peft_config.virtual_tokens_per_special_token] = 1
            self.base_model.model.vt_attention_mask = vt_encoder_attention_mask
        elif peft_config.vt_attention_type == VTAttentionType.ENSEMBLE:
            # Every virtual token sees 1 virtual token from previous special token, and it does not see its neighbors
            vt_ensemble_attention_mask = torch.zeros(peft_config.num_virtual_tokens, peft_config.num_virtual_tokens)
            for i in range(peft_config.num_special_tokens):
                for j in range(peft_config.virtual_tokens_per_special_token):
                    for k in range(i+1):
                        vt_ensemble_attention_mask[i*peft_config.virtual_tokens_per_special_token+j, 
                                                  k*peft_config.virtual_tokens_per_special_token+j] = 1
            self.base_model.model.vt_attention_mask = vt_ensemble_attention_mask
        else:
            raise ValueError("Invalid VT attention type")
        
        
        if peft_config.aggregation_type == AggregationType.MEAN:
            self.mean_aggregate = self.mean_aggregate
        elif peft_config.aggregation_type == AggregationType.WEIGHTED:
            self.mean_aggregate = self.weight_aggregate
            self.weighting_layers = torch.nn.ModuleList(
                [torch.nn.Linear(peft_config.virtual_tokens_per_special_token, 1, bias=False) for _ in range(peft_config.num_special_tokens)])
            # init to average 
            for layer in self.weighting_layers:
                layer.weight.data.fill_(1/peft_config.virtual_tokens_per_special_token)
            self.weighting_layers.to(self.device, dtype=self.base_model.lm_head.weight.dtype)
        elif peft_config.aggregation_type == AggregationType.ADAPTIVAE_WEIGHTED:
            # give a score to each virtual token based on the hidden states
            self.mean_aggregate = self.adaptive_weight_aggregate
            self.weighting_layers = torch.nn.ModuleList(
                [torch.nn.Linear(self.base_model.config.hidden_size, 1, bias=False) for _ in range(peft_config.num_special_tokens)])
            self.weighting_layers.to(self.device, dtype=self.base_model.lm_head.weight.dtype)
        else:
            raise ValueError("Invalid aggregation type")
        
        if peft_config.use_custom_lm_head:
            self.add_custom_lm_head()
        
        self.default_adapter_name = adapter_name
        if peft_config.use_prefix_tuning:
            self.add_prefix_tuning(peft_config)
        
        # if peft_config.num_exits > 1:
        #     self.exit_weights = torch.nn.Parameter(torch.ones(peft_config.num_exits) / peft_config.num_exits).to(self.device)


    # Overwrite the property to enable the use a PEFT wrapper 
    @property
    def active_peft_config(self):
        return self.prompt_peft_config[self.active_adapter]
    
    
    def get_peft_model_state_dict(self):
        prev_peft_config = self.peft_config
        self.peft_config = self.prompt_peft_config
        state_dict = get_peft_model_state_dict(self)
        self.peft_config = prev_peft_config
        return state_dict
    
    
    def set_peft_model_state_dict(self, state_dict):
        prev_peft_config = self.peft_config
        self.peft_config = self.prompt_peft_config
        set_peft_model_state_dict(self, state_dict)
        self.peft_config = prev_peft_config
    

    def add_custom_lm_head(self):
        self.custom_lm_head = torch.nn.Linear(self.base_model.config.hidden_size, self.base_model.config.vocab_size, bias=False)
        # Clone the weights and biases and wrap them in torch.nn.Parameter
        self.custom_lm_head.weight = torch.nn.Parameter(self.base_model.lm_head.weight.clone())
        self.custom_lm_head.to(self.device, dtype=self.base_model.lm_head.weight.dtype)

    
    def add_prefix_tuning(self, peft_config):
        self.prefix_adapter_name = "prefix"
        prefix_config = PrefixTuningConfig(task_type="CAUSAL_LM", 
                                           num_virtual_tokens=peft_config.prefix_virtual_tokens*peft_config.num_special_tokens)
        # PEFT does not support mixed prefix and prompt tuning, so have to do this hack
        self.peft_type = prefix_config.peft_type
        self.add_adapter(self.prefix_adapter_name, prefix_config)
        self.peft_type = peft_config.peft_type
        self.base_model.model.prefix_virtual_tokens = peft_config.prefix_virtual_tokens * peft_config.num_special_tokens
        # set prompt token attention to prefix token; the prompt tokens corresponding to different special tokens attend to different prefix tokens
        prefix_attention_mask = torch.zeros(peft_config.num_virtual_tokens, peft_config.prefix_virtual_tokens * peft_config.num_special_tokens)
        for i in range(peft_config.num_special_tokens):
            prefix_attention_mask[i*peft_config.virtual_tokens_per_special_token: (i+1)*peft_config.virtual_tokens_per_special_token, 
                                    i*peft_config.prefix_virtual_tokens: (i+1)*peft_config.prefix_virtual_tokens] = 1
        self.base_model.model.prefix_attention_mask = prefix_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        past_key_values=None,
        **kwargs,
        ):
        peft_config = self.active_peft_config
        total_virtual_tokens = peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token

        batch_size = input_ids.shape[0]
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You have to provide either input_ids or inputs_embeds")

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        if attention_mask is not None:
            # concat prompt attention mask
            suffix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((attention_mask, suffix_attention_mask), dim=1)

        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        if peft_config.use_prefix_tuning:
            self.set_adapter(self.prefix_adapter_name)
            past_key_values = self.get_prompt(batch_size)  
            if attention_mask is not None:
                prefix_attention_mask = torch.ones(batch_size, self.active_peft_config.num_virtual_tokens).to(attention_mask.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
            self.set_adapter(self.default_adapter_name)
            position_ids = torch.cat((position_ids, torch.repeat_interleave(torch.arange(
                input_ids.shape[1]+peft_config.prefix_virtual_tokens, 
                input_ids.shape[1]+peft_config.prefix_virtual_tokens+peft_config.num_special_tokens), 
                peft_config.virtual_tokens_per_special_token)))
        else:
            position_ids = torch.cat((position_ids, torch.repeat_interleave(torch.arange(
                input_ids.shape[1], 
                input_ids.shape[1]+peft_config.num_special_tokens), 
                peft_config.virtual_tokens_per_special_token)))

        # concat prompt labels
        if labels is not None:
            suffix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
            labels = torch.cat((labels, suffix_labels), dim=1)

        kwargs.update(
            {
                "attention_mask": attention_mask,
                # "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
        prompts = prompts.to(inputs_embeds.dtype)
        # append prompts to inputs_embeds
        inputs_embeds = torch.cat((inputs_embeds, prompts), dim=1)
        outputs = self.base_model.model(inputs_embeds=inputs_embeds, **kwargs)
        hidden_states = outputs[0]
        if peft_config.num_exits > 1:
            # set the hidden_states of special tokens to the average of last num_exit layers
            hidden_states_stack = torch.stack(outputs.hidden_states[-peft_config.num_exits:])
            hidden_states[:, -total_virtual_tokens:, :] = hidden_states_stack.mean(dim=0)[:, -total_virtual_tokens:, :]
        
        if peft_config.use_custom_lm_head:
            prompt_logits = self.custom_lm_head(hidden_states[:, -total_virtual_tokens:, :])
            logits = self.base_model.lm_head(hidden_states[:, 
                                                           :-total_virtual_tokens, 
                                                           :])
            logits = torch.cat((logits, prompt_logits), dim=1)
        else:
            logits = self.base_model.lm_head(hidden_states)
        logits = self.mean_aggregate(logits, hidden_states)
        return CausalLMOutputWithPast(logits=logits, 
                                      past_key_values=outputs.past_key_values, 
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)
    

    def mean_aggregate(self, logits, hidden_states):
        peft_config = self.active_peft_config
        # Reshape logits to separate virtual tokens for special tokens
        prompt_logits = logits[:, -peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token:, :].contiguous()
        if peft_config.virtual_tokens_per_special_token > 1:
            # mean aggregate the virtual tokens for each special token
            prompt_logits = prompt_logits.reshape(prompt_logits.shape[0], peft_config.num_special_tokens, peft_config.virtual_tokens_per_special_token, prompt_logits.shape[-1]).mean(dim=2)
        return torch.cat((logits[:, :-peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token, :], prompt_logits), dim=1)


    def weight_aggregate(self, logits, hidden_states):
        logits = logits.to(self.device)
        peft_config = self.active_peft_config
        # Reshape logits to separate virtual tokens for special tokens
        prompt_logits = logits[:, -peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token:, :].contiguous()
        batch_size, _, vocab_size = prompt_logits.shape
        if peft_config.virtual_tokens_per_special_token > 1:
            prompt_logits = prompt_logits.reshape(batch_size, peft_config.num_special_tokens, peft_config.virtual_tokens_per_special_token, vocab_size)
            aggregated_logits = []
            for i, layer in enumerate(self.weighting_layers):
                # Select the virtual tokens for the current special token
                special_token_logits = prompt_logits[:, i, :, :].permute(0, 2, 1).reshape(-1, peft_config.virtual_tokens_per_special_token)
                
                # Apply the corresponding linear layer to aggregate the virtual tokens
                # The output will be of shape (batch_size * vocab_size, 1), so we need to squeeze the last dimension
                weighted_logits = layer(special_token_logits).reshape(batch_size, vocab_size)
                
                # Collect the aggregated logits
                aggregated_logits.append(weighted_logits)

            # Stack the aggregated logits for all special tokens and reshape to match the expected output shape
            prompt_logits = torch.stack(aggregated_logits, dim=1)
        
        # Concatenate the weighted prompt logits back with the original logits
        return torch.cat((logits[:, :-peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token, :], prompt_logits), dim=1)


    def adaptive_weight_aggregate(self, logits, hidden_states):
        peft_config = self.active_peft_config
        # Reshape logits to separate virtual tokens for special tokens
        prompt_logits = logits[:, -peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token:, :].contiguous()
        special_token_hidden_states = hidden_states[:, -peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token:, :].contiguous()
        batch_size, _, vocab_size = prompt_logits.shape
        if peft_config.virtual_tokens_per_special_token > 1:
            prompt_logits = prompt_logits.reshape(batch_size, peft_config.num_special_tokens, peft_config.virtual_tokens_per_special_token, vocab_size)
            special_token_hidden_states = special_token_hidden_states.reshape(batch_size, peft_config.num_special_tokens, peft_config.virtual_tokens_per_special_token, special_token_hidden_states.shape[-1])
            aggregated_logits = []
            for i, layer in enumerate(self.weighting_layers):
                # caculate score based on hidden states
                special_token_hidden_state = special_token_hidden_states[:, i, :, :]
                special_token_hidden_state = special_token_hidden_state.reshape(-1, special_token_hidden_state.shape[-1])
                scores = layer(special_token_hidden_state).reshape(batch_size, peft_config.virtual_tokens_per_special_token)
                scores = torch.nn.functional.softmax(scores, dim=-1)
                weighted_logits = (prompt_logits[:, i, :, :].permute(0, 2, 1) * scores).sum(dim=-1)
                # Collect the aggregated logits
                aggregated_logits.append(weighted_logits)

            # Stack the aggregated logits for all special tokens and reshape to match the expected output shape
            prompt_logits = torch.stack(aggregated_logits, dim=1)
        
        # Concatenate the weighted prompt logits back with the original logits
        return torch.cat((logits[:, :-peft_config.num_special_tokens*peft_config.virtual_tokens_per_special_token, :], prompt_logits), dim=1)


    def end_inference(self):
        r"""
        Clear the inference buffers.
        """
        self.dynamic_inferece_buffers = None
        self.inference_buffers = None
        self.base_model.model.tree_mask = None
        self.base_model.model.vt_attention_mask = None


    def start_inference(self, input_ids, past_key_values=None, current_length_data=None, **kwargs):
        r"""
        Run the initial inference step for the prompt decoding model.
        """
        config = self.active_peft_config
        num_special_tokens = config.num_special_tokens
        self.base_model.model.tree_mask = None
        self.base_model.model.prompt_token_indices = None # only support 1 virtual token for inference
        input_embeds = self.word_embeddings(input_ids)
        prompts = self.get_prompt(1).to(input_embeds.dtype)
        input_embeds = torch.cat([input_embeds, prompts], dim=1)
        outputs = self.base_model(
            inputs_embeds=input_embeds, 
            past_key_values=past_key_values,
        )
        # logtis generated by the original model
        logits = outputs.logits[:, -num_special_tokens-1:-num_special_tokens, :]
        # logtis generated by the prompt tuning model
        prompt_logits = outputs.logits[:, -num_special_tokens:, :]
        # set past key values length 
        current_length_data.fill_(input_ids.shape[1])
        # use max depth buffers
        self.inference_buffers = self.dynamic_inferece_buffers[self.max_depth]
        self.base_model.model.tree_mask = self.inference_buffers["attn_mask"]
        return logits, prompt_logits
        

    def generate_dynamic_buffers(self, candidates):
        r"""
        Generate dynamic inference buffers for the prompt decoding model. If the max depth of the sparse
        tree is n, then there are n buffers generated for the model. The buffers are generated based on the
        candidates arguments. 
        
        Args:
            candidates (List[(List[Int], Float, Int)]): Candidate takes the form (path, acc, n) where path is the path
            to the tree node, acc is the probability of selecting this node, and n is the depth of the number of special
            tokens appended to this tree node.
        """
        dynamic_inferece_buffers = {}
        max_depth = 1
        for depth, candidate_lists in candidates.items():
            dynamic_inferece_buffers[depth] = self.generate_buffers(candidate_lists)
            max_depth = max(max_depth, depth)
        self.dynamic_inferece_buffers = dynamic_inferece_buffers
        self.inference_buffers = self.dynamic_inferece_buffers[max_depth]
        self.current_depth = max_depth
        self.max_depth = max_depth


    # adapted from Medusa: https://github.com/FasterDecoding/Medusa/blob/5e980538695096e7e372c1e27a6bcf142bfeab11/medusa/model/utils.py
    def generate_buffers(self, candidate_lists):
        r"""
        Generate buffers for the prompt decoding model.

        Args:
            candidate_lists (List[(List[Int], Float, Int)]): Candidate takes the form (path, acc, n) where path is the path
            to the tree node, acc is the probability of selecting this node, and n is the depth of the number of special
            tokens appended to this tree node.
            depth (int, optional): The depth of the tree. Defaults to 1.

        Returns:
            _type_: _description_
        """
        config = self.active_peft_config
        num_special_tokens = config.num_special_tokens
        candidate_lists = sorted([(path.copy(), acc, n) for path, acc, n in candidate_lists], key=lambda x: (len(x[0]), x[0]))
        original_paths = [path.copy() for (path, _, _) in candidate_lists]
        paths = [(path.copy(), n) for (path, _, n) in candidate_lists]
        # add special tokens to the paths
        if num_special_tokens > 0:
            paths += [(path + [-1], n) for (path, n) in paths] + [([-1], num_special_tokens)]
        for i in range(2, num_special_tokens+1):
            paths += [(path + [-i], n) for (path, n) in paths if path[-1]-1 == -i and i <= n]
        paths = sorted([path for path, _ in paths], key=lambda x: (len(x), x))
        # Sort the choices based on their lengths and then their values
        paths = sorted(paths, key=lambda x: (len(x), x))
        prompt_length = len(paths) + 1
        
        # Get indices for each special token
        special_token_indices = {}
        for i, path in enumerate([[]] + original_paths):
            candidate_special_token_indices = []
            for j in range(1, num_special_tokens+1):
                extended_path = path+[-n for n in range(1, j+1)]
                if extended_path in paths:
                    candidate_special_token_indices.append(paths.index(extended_path)+1)
            special_token_indices[i] = candidate_special_token_indices
        normal_token_indices = [0] + [i+1 for i in range(len(paths)) if paths[i][-1] >= 0]
        normal_token_indices = torch.tensor(normal_token_indices, dtype=torch.long)

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in paths:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth
        
        # Create the attention mask
        attn_mask = torch.eye(prompt_length, prompt_length)
        attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_path = paths[start + j]
                # retrieve ancestor position
                if len(cur_path) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_path) - 1):
                    ancestor_idx.append(paths.index(cur_path[:c+1]) + 1)
                attn_mask[start+j+1, ancestor_idx] = 1
            start += depth_counts[i]

        # Generate tree indices
        tree_indices = torch.zeros(prompt_length, dtype=torch.long)
        tree_indices[0] = 0
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_path = paths[start + j]
                if cur_path[-1] < 0:
                    tree_indices[start + j + 1] = -(num_special_tokens+cur_path[-1]+1)
                else:
                    tree_indices[start + j + 1] = cur_path[-1] + TOPK * i + 1
            start += depth_counts[i]
        
        # Generate position ids 
        position_ids = torch.arange(prompt_length, dtype=torch.long)
        start = 0 
        for i in range(len(depth_counts)):
            position_ids[start+1:start+depth_counts[i]+1] = i + 1 
            start += depth_counts[i]
        
        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(original_paths)):
            cur_choice = original_paths[-i-1]
            retrieve_indices = []
            if cur_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_choice)):
                    retrieve_indices.append(original_paths.index(cur_choice[:c+1]))
                    retrieve_paths.append(cur_choice[:c+1])
            retrieve_indices_nest.append(retrieve_indices)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)
 
        return {
            "attn_mask": attn_mask.unsqueeze(0).unsqueeze(0).to(self.device),
            "special_token_indices": special_token_indices,
            "normal_token_indices": normal_token_indices.to(self.device),
            "tree_indices": tree_indices.to(self.device),
            "position_ids": position_ids.to(self.device),
            "retrieve_indices": retrieve_indices.to(self.device),
        }
    

    # adapted from Medusa: https://github.com/FasterDecoding/Medusa/blob/5e980538695096e7e372c1e27a6bcf142bfeab11/medusa/model/utils.py
    def generate_candidates(self, logits, prompt_logits, temperature=0, posterior_threshold=0.3, posterior_alpha=0.09, sampling='greedy'):
        r"""
        This function generates candidates for the prompt decoding model.

        Args:
            logits (torch.Tensor): The logits of normal tokens generated by the model.
            prompt_logits (torch.Tensor): The logits of special tokens generated by the model.
            temperature (Float, optional): The temperature value for sampling. Defaults to 0.
            posterior_threshold (float, optional): The posterior threshold value. Defaults to 0.3.
            posterior_alpha (float, optional): The posterior alpha value. Defaults to 0.09.
            sampling (str, optional): The sampling method to be used. Defaults to 'greedy'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The cartesian candidates generated by the model and 
            the input embeddings used for the tree decoding.
        """
        tree_indices = self.inference_buffers["tree_indices"]
        normal_token_indices = self.inference_buffers["normal_token_indices"]
        if sampling == 'greedy' or temperature == 0:
            candidate_logit = torch.argmax(logits[:, -1]).unsqueeze(0)
        elif sampling == 'typical':
            candidate_logit = get_typical_one_token(logits[:, -1], temperature, posterior_threshold, posterior_alpha).squeeze(0)
        else: 
            raise NotImplementedError("Sampling method not implemented")

        prompt_candidate_logit = torch.topk(prompt_logits, TOPK, dim=-1).indices
        
        candidates = torch.cat([candidate_logit, prompt_candidate_logit.view(-1)], dim=-1)
        
        tree_candidates = candidates[tree_indices[normal_token_indices]]
        tree_candidates_ext = torch.cat([tree_candidates, torch.zeros(1, dtype=torch.long, device=tree_candidates.device)], dim=0)    
        cart_candidates = tree_candidates_ext[self.inference_buffers["retrieve_indices"]]
        
        candidates_embeds = self.word_embeddings(candidates)
        prompts = self.get_prompt(1).squeeze().to(candidates_embeds.dtype)
        tree_candidates_embeds = torch.cat([candidates_embeds, prompts], dim=0)[tree_indices]
        
        return cart_candidates, tree_candidates_embeds.unsqueeze(0)


    # adapted from Medusa: https://github.com/FasterDecoding/Medusa/blob/5e980538695096e7e372c1e27a6bcf142bfeab11/medusa/model/utils.py
    def tree_decoding(
        self, 
        tree_candidates_embeds,
        past_key_values,
        input_ids
    ): 
        r"""
        This function performs tree decoding (forward pass) for the prompt decoding model.

        Args:
            tree_candidates_embeds (torch.Tensor): The embeddings of the tree candidates.
            past_key_values : The past key values of the model.
            input_ids (torch.Tensor): The input ids of the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The logits of the normal tokens and all logtis generated by the model.
        """
        position_ids = input_ids.shape[1] + self.inference_buffers["position_ids"]
        
        outputs = self.base_model(
            inputs_embeds=tree_candidates_embeds, 
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        
        logits = outputs.logits[0, self.inference_buffers["normal_token_indices"]][self.inference_buffers["retrieve_indices"]]
        
        return logits, outputs.logits
    
    
    # adapted from Medusa: https://github.com/FasterDecoding/Medusa/blob/5e980538695096e7e372c1e27a6bcf142bfeab11/medusa/model/utils.py
    def evaluate_posterior(self, logits, candidates, temperature, posterior_threshold, posterior_alpha, sampling='greedy'):
        r"""
        This function evaluates the posterior probabilities of the candidates generated by the model.
        
        Args:
            logits (torch.Tensor): The logits of the normal tokens generated by the model.
            candidates (torch.Tensor): The candidates generated by the model.
            temperature (float): The temperature value for sampling.
            posterior_threshold (float): The posterior threshold value.
            posterior_alpha (float): The posterior alpha value.
            sampling (str, optional): The sampling method to be used. Defaults to 'greedy'.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The best candidate and the length of the accepted tokens.
        """ 
        # Greedy decoding based on temperature value
        if temperature == 0:
            # Find the tokens that match the maximum logits for each position in the sequence
            posterior_mask = (
                candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
            ).int()
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
            accept_length = candidates_accept_length.max()
            # Choose the best candidate
            if accept_length == 0:
                # Default to the first candidate if none are accepted
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            return best_candidate, accept_length
        
        if sampling == 'greedy':
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        if sampling == 'typical':
            # Calculate posterior probabilities and thresholds for candidate selection
            posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha)
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
                # Accept the best one according to likelihood
            return best_candidate, accept_length
        else:
            raise NotImplementedError("Sampling method not implemented")
        
    
    # adapted from Medusa: https://github.com/FasterDecoding/Medusa/blob/5e980538695096e7e372c1e27a6bcf142bfeab11/medusa/model/utils.py
    def update_inference_inputs(
        self,
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        logits,
        all_logits,
        new_token,
        past_key_values_data,
        current_length_data,
    ):
        r"""
        This function updates the inputs, KV cache based on the best candidate selected by the model.
        
        Args:
            input_ids (torch.Tensor): The input ids of the model.
            candidates (torch.Tensor): The candidates generated by the model.
            best_candidate (torch.Tensor): The best candidate selected by the model.
            accept_length (int): The length of the accepted tokens.
            logits (torch.Tensor): The logits of the normal tokens generated by the model.
            all_logits (torch.Tensor): The logits of all tokens generated by the model.
            new_token (int): The number of new tokens generated by the model.
            past_key_values_data (torch.Tensor): The past key values of the model.
            current_length_data (torch.Tensor): The current length of the model.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: The updated input ids, logits, prompt logits, 
            and the number of new tokens.
        """
        # Calculate the starting position for new tokens based on the previous input length
        prev_input_len = input_ids.shape[1]
        # Map the best candidate indices to the original indices in the sequence
        select_indices = (
           self.inference_buffers['retrieve_indices'][best_candidate, :accept_length + 1]
        )
        # Append the tokens from the best candidate to the input sequence
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
        )
        # Update the past key values based on the selected tokens
        # Source tensor that contains relevant past information based on the selected candidate
        tgt = past_key_values_data.index_select(-2, self.inference_buffers["normal_token_indices"] + prev_input_len)
        tgt = tgt.index_select(-2, select_indices)
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

        # Update the current length tensor (currently only support batch size is 1)
        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # Extract logits and speculative logits for the accepted tokens
        logits = logits[None, best_candidate, accept_length : accept_length + 1]
        candidate_index = self.inference_buffers['retrieve_indices'][best_candidate, accept_length : accept_length + 1]
        prompt_token_indices = self.inference_buffers['special_token_indices'][candidate_index.cpu().item()]
        prompt_logits = all_logits[:, prompt_token_indices]
        
        # dynamically update the inference buffers
        self.inference_buffers = self.dynamic_inferece_buffers[len(prompt_token_indices)]
        self.base_model.model.tree_mask = self.inference_buffers["attn_mask"]
        self.current_depth = len(prompt_token_indices)
        
        # Update the new token counter
        new_token += accept_length + 1

        return input_ids, logits, prompt_logits, new_token


    @torch.inference_mode()
    def ppd_generate(self, input_ids, max_steps=512, temperature=0., posterior_threshold=0.09, posterior_alpha=0.3, sampling='greedy'):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        tokenizer = self.tokenizer
        if not hasattr(self, "inference_buffers"):
            print('Generate buffers')
            self.generate_dynamic_buffers(get_dynamic_sparse_tree(self.active_peft_config.base_model_name_or_path))
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            print('Initialize past key values')
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        logits, prompt_logits = self.start_inference(input_ids, past_key_values, current_length_data)
        new_token = 0
        
        for idx in range(max_steps): 
            candidates, tree_candidates_embeds = self.generate_candidates(
                logits, 
                prompt_logits, 
                temperature, 
                posterior_threshold, 
                posterior_alpha, 
                sampling)
            logits, prompt_logits = self.tree_decoding(tree_candidates_embeds, past_key_values, input_ids)
            best_candidate, accept_length = self.evaluate_posterior(
                logits, 
                candidates, 
                temperature, 
                posterior_threshold, 
                posterior_alpha,
                sampling)
            input_ids, logits, prompt_logits, new_token = self.update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                logits,
                prompt_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
            
            yield input_ids

            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
    
    @torch.inference_mode()
    def naive_generate(self, input_ids, max_steps = 512, temperature=0.7, posterior_threshold = 0.09, posterior_alpha = 0.3, sampling='greedy', max_new_token=512):
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        tokenizer = self.tokenizer
        if not hasattr(self, "inference_buffers"):
            print('Generate buffers')
            self.generate_dynamic_buffers(get_dynamic_sparse_tree(self.active_peft_config.base_model_name_or_path))
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            print('Initialize past key values')
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        tree_mask = self.base_model.model.tree_mask
        vt_attention_mask = self.base_model.model.vt_attention_mask
        prompt_token_indices = self.base_model.model.prompt_token_indices
        self.base_model.model.tree_mask = None
        self.base_model.model.vt_attention_mask = None
        self.base_model.model.prompt_token_indices = None
        outputs = self.base_model(input_ids, past_key_values = past_key_values, use_cache=True)
        new_token = 0
        
        for idx in range(max_steps): 
            input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values = past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1
            
            yield input_ids

            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_token:
                break
        
        self.base_model.model.tree_mask = tree_mask
        self.base_model.model.vt_attention_mask = vt_attention_mask
        self.base_model.model.prompt_token_indices = prompt_token_indices


    def save_pretrained(self, save_directory: str, **kwargs):
        print("Saving model to", save_directory)
        prev_peft_config = self.peft_config
        self.peft_config = self.prompt_peft_config
        super().save_pretrained(save_directory, **kwargs)
        if hasattr(self, "weighting_layers"):
            torch.save(self.weighting_layers.state_dict(), os.path.join(save_directory, "weighting_layers.pt"))
        if hasattr(self, "custom_lm_head"):
            torch.save(self.custom_lm_head.state_dict(), os.path.join(save_directory, "custom_lm_head.pt"))
        self.peft_config = prev_peft_config
        # if hasattr(self, "exit_weights"):
        #     torch.save(self.exit_weights, os.path.join(save_directory, "exit_weights.pt"))



class AutoPromptDecoder:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        adapter_name: str = "default",
        is_trainable: bool = False,
        new_config: Optional[PromptConfig] = None,
        *args,
        **kwargs,
    ):
        config_path = os.path.join(pretrained_model_name_or_path, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r+") as f:
                config = json.load(f)
                if "peft_type" in config:
                    del config["peft_type"]
                f.seek(0)
                json.dump(config, f, indent=2, sort_keys=True)
                f.truncate()
        old_config = PromptConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        peft_config = old_config if new_config is None else new_config
        base_model_path = peft_config.base_model_name_or_path
        config = AutoConfig.from_pretrained(base_model_path)

        if config.model_type == "llama":
            base_model = CustomLlamaForCausalLM.from_pretrained(
                base_model_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama for now")

        model = PromptDecoder.from_pretrained(
            base_model,
            pretrained_model_name_or_path,
            is_trainable=is_trainable,
            config=peft_config,
            **kwargs,
        )

        if old_config.use_prefix_tuning:
            print("Loading prefix adapter")
            model.load_adapter(f"{pretrained_model_name_or_path}/prefix", "prefix")
        
        if old_config.aggregation_type == AggregationType.WEIGHTED or \
                old_config.aggregation_type == AggregationType.ADAPTIVAE_WEIGHTED:
            print("Loading weighting layers")
            model.weighting_layers.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "weighting_layers.pt")))
        
        if old_config.use_custom_lm_head:
            print("Loading custom lm head")
            model.custom_lm_head.load_state_dict(torch.load(os.path.join(pretrained_model_name_or_path, "custom_lm_head.pt")))
        
        if new_config and new_config.use_prefix_tuning:
            print("Adding prefix adapter")
            model.add_prefix_tuning(new_config)
        
        if new_config and new_config.use_custom_lm_head:
            # stage 2 training
            print("Adding custom lm head")
            model.add_custom_lm_head()

        return model
  