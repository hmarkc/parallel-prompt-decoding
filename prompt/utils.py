from dataclasses import dataclass, field
import json
import os
from enum import Enum
from typing import Dict, Optional, Sequence

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

import prompt.inference.dynamic_sparse_trees_3_vicuna_13b as dynamic_sparse_trees_3_vicuna_13b
import prompt.inference.dynamic_sparse_trees_3_vicuna_7b as dynamic_sparse_trees_3_vicuna_7b
import prompt.inference.dynamic_sparse_trees_3_MobileLLaMA as dynamic_sparse_trees_3_mobilellama


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class VTAttentionType(str, Enum):
    """Attention type for VicunaTuning
    """
    DECODER = "decoder"
    ENCODER = "encoder"
    ENSEMBLE = "ensemble"

    def __str__(self):
        return self.value
    
    @staticmethod
    def from_str(s):
        s = s.lower()
        if s == "decoder":
            return VTAttentionType.DECODER
        elif s == "encoder":
            return VTAttentionType.ENCODER
        elif s == "ensemble":
            return VTAttentionType.ENSEMBLE
        else:
            raise ValueError(f"Invalid attention type: {s}")


class AggregationType(str, Enum):
    """Aggregation type for VicunaTuning
    """
    MEAN = "mean"
    WEIGHTED = "weighted"
    ADAPTIVAE_WEIGHTED = "adaptive_weighted"
    
    def __str__(self):
        return self.value
    
    @staticmethod
    def from_str(s):
        s = s.lower()
        if s == "mean":
            return AggregationType.MEAN
        elif s == "weighted":
            return AggregationType.WEIGHTED
        elif s == "adaptive_weighted":
            return AggregationType.ADAPTIVAE_WEIGHTED
        else:
            raise ValueError(f"Invalid aggregation type: {s}")


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> Dict:
        """
        Preprocesses conversation data and tokenizes it for model input.

        Args:
            sources: A list of conversation sources.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            Dict: A dictionary containing tokenized inputs, labels, and attention mask.
        """
        conv = get_conversation_template("vicuna")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}, {j}, {role}, {conv.roles[j % 2]}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True
        ).input_ids
        targets = input_ids.clone()

        assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        # print("sep", sep)
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            # the number of preceding padding tokens
            cur_len = 1
            for p in target:
                if p == tokenizer.pad_token_id:
                    cur_len += 1
                else:
                    break
            target[:cur_len] = IGNORE_TOKEN_ID
            # target_imm = target.clone()
            # target_imm[target_imm == -100] = 0
            # print("target1", tokenizer.decode(target_imm))
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                # print(cur_len, cur_len + instruction_len)
                # target_imm = target.clone()
                # target_imm[target_imm == -100] = 0
                # print("target2", tokenizer.decode(target_imm))
                cur_len += turn_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID

        # a= (input_ids[0, :] != targets[0, :]).nonzero(as_tuple=False)
        # print("input_ids compare to targets", a)
        # print("targets compare to input_ids", a.shape)
        # print(targets[0, input_ids[0, :] != targets[0, :]])
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )



class FineTuningDataset(Dataset):
    """Dataset for fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, offset):
        super(FineTuningDataset, self).__init__()

        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)
        block_indices = find_last_positive_block(data_dict["labels"], IGNORE_TOKEN_ID, offset)
        input_ids, attention_mask, labels = randomly_truncate(data_dict["input_ids"], 
                                                               data_dict["attention_mask"], 
                                                               data_dict["labels"], 
                                                               block_indices, 
                                                               offset, 
                                                               tokenizer.pad_token_id,
                                                               IGNORE_TOKEN_ID)

        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    
    def set_size(self, size):
        if size is None:
            return
        self.input_ids = self.input_ids[:size]
        self.labels = self.labels[:size]
        self.attention_mask = self.attention_mask[:size]


def get_finetune_dataset(
    tokenizer: transformers.PreTrainedTokenizer, data_path, size: Optional[int] = None, offset=0
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """

    json_file = json.load(open(data_path, "r"))
    size = size or len(json_file)
    dataset = FineTuningDataset(json_file[:size], tokenizer=tokenizer, offset=offset)
    return dataset


def find_last_positive_block(A, ignored_id, n):
    """
    Find the start and end index of the last block of positive numbers of at least size n in each row of A.

    Args:
    - A (torch.Tensor): Input tensor of shape [N, L]
    - n (int): Minimum size of the block

    Returns:
    - torch.Tensor: Tensor of shape [N, 2] containing start and end indices of the last block of positive numbers of at least size n
    """
    N, L = A.shape
    indices = torch.full((N, 2), -1, dtype=torch.long)  # Initialize with -1

    for i in range(N):
        last_pos_end = -1
        block_size = 0

        for j in range(L-1, -1, -1):
            if A[i, j] != ignored_id:
                if last_pos_end == -1:
                    last_pos_end = j  # Mark the end of a positive block
                block_size += 1
            else:
                if last_pos_end != -1:
                    if block_size >= n:
                        indices[i, 0] = j + 1  # Start of the last positive block
                        indices[i, 1] = last_pos_end
                        break
                    else:
                        # Reset for next block search
                        last_pos_end = -1
                        block_size = 0
            if j == 0 and last_pos_end != -1 and block_size >= n:
                indices[i, 0] = 0
                indices[i, 1] = last_pos_end

    return indices


def randomly_truncate(input_ids, attention_mask, labels, positions, k, pad_token_id=0, IGNORE_TOKEN_ID=IGNORE_TOKEN_ID):
    N, L = input_ids.shape
    # Initialize the tensor that will hold the truncated sequences
    truncated_batch = torch.full_like(input_ids, pad_token_id)
    truncated_attention_mask = torch.full_like(attention_mask, 0)
    truncated_labels = torch.full_like(labels, IGNORE_TOKEN_ID)
    
    for i in range(N):
        start, end = positions[i]
        # The cut has to leave at least k elements truncated, so we adjust the end accordingly
        # Also, ensure the cut is at least at the start position or further to the right
        if start == -1 or end == -1:
            cut = L-k
        else:
            valid_end = max(start + 1, end - k + 1)
            # Randomly choose a cut point from start to the valid_end
            cut = torch.randint(start, valid_end, (1,)).item()
        # print(start, cut, L-cut)
        # Truncate the sequence and pad from the left
        try:
            truncated_batch[i, L-cut:] = input_ids[i, :cut]
            truncated_attention_mask[i, L-cut:] = attention_mask[i, :cut]
            truncated_labels[i, L-cut-k:] = labels[i, :cut+k]
        except:
            print(valid_end, cut, start, end)
            print(i, L-cut, cut, L, input_ids[i, :cut].shape, truncated_batch[i, L-cut:].shape)
            print(i, L-cut, cut, L, attention_mask[i, :cut].shape, truncated_attention_mask[i, L-cut:].shape)
            print(i, L-cut-k, cut+k, L, labels[i, :cut+k].shape, truncated_labels[i, L-cut-k:].shape)
            raise Exception("Error in truncation")

    return truncated_batch, truncated_attention_mask, truncated_labels


class DistillationDataset(Dataset):
    """Dataset for fine-tuning.

    Args:
        data (list): A list of data containing input_ids, labels, and attention_mask.
    """

    def __init__(self, data):
        super(DistillationDataset, self).__init__()
        self.input_ids = [d["input_ids"] for d in data]
        self.labels = [d["labels"] for d in data]
        self.attention_mask = [d["attention_mask"] for d in data]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
    
    def set_size(self, size):
        if size is None:
            return 
        self.input_ids = self.input_ids[:size]
        self.labels = self.labels[:size]
        self.attention_mask = self.attention_mask[:size]


def get_self_distillation_dataset(model, data_path, num_special_tokens):
    dataset = torch.load(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    data = []
    model.eval()
    # dataloader is faster but batched input need more memory
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_size, seq_length = input_ids.shape
        preds = input_ids.clone()
        batch_labels = []

        for j in range(num_special_tokens+1):
            with torch.inference_mode():
                outputs = model(input_ids=preds, attention_mask=attention_mask)
            logits = outputs.logits
            input_id = logits[:, -1:, :].argmax(-1)
            
            if j > 0:
                batch_labels.append(logits[:, -1, :])
                
            preds = torch.cat([preds, input_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1).to(attention_mask.device)], dim=1)
        
        labels = torch.stack(batch_labels, dim=1)
        for i in range(batch_size):
            data.append({"input_ids": preds[i, :-num_special_tokens-1], "labels": labels[i], "attention_mask": attention_mask[i, :-num_special_tokens-1]})
    return DistillationDataset(data)


def chunk_dataset(dataset_path, chunk_size, output_dir):
    dataset = torch.load(dataset_path)
    total_size = len(dataset)
    print(f"Total size: {total_size}")
    for i in tqdm(range(0, total_size, chunk_size)):
        chunk = dataset[i:i+chunk_size]
        torch.save(chunk, os.path.join(output_dir, f'dataset_chunk_{i//chunk_size}.pt'))


class ChunkDataset(Dataset):
    def __init__(self, chunk_dir):
        super(ChunkDataset, self).__init__()
        self.chunk_dir = chunk_dir
        # List all chunk files
        self.chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.startswith('dataset_chunk_')]
        self.chunk_files.sort(key=lambda x: (len(x), x))
        # Calculate offsets and total length
        self.lengths = [torch.load(f, map_location='cpu')['input_ids'].__len__() for f in self.chunk_files]
        self.cumulative_lengths = [sum(self.lengths[:i+1]) for i in range(len(self.lengths))]
    
    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which chunk contains the item
        chunk_idx = next(i for i, total in enumerate(self.cumulative_lengths) if total > idx)
        if chunk_idx > 0:
            idx -= self.cumulative_lengths[chunk_idx-1]  # Adjust index relative to the chunk
        
        # Load the chunk
        chunk = torch.load(self.chunk_files[chunk_idx], map_location='cpu')
        
        # Extract and return the item
        return dict(
            input_ids=chunk['input_ids'][idx],
            labels=chunk['labels'][idx],
            attention_mask=chunk['attention_mask'][idx],
        )

    
def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    original_logit = logit.clone()  
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    prob = F.softmax(logit, dim=-1)
    try:
        sampled_tokens = torch.multinomial(prob, 1)
    except:
        print(prob.max(), prob.min())
        print(logit.max(), logit.min())
        print(original_logit.max(), original_logit.min())
        print(temperature, original_logit.max()/ temperature, original_logit.min()/ temperature)
        print(indices_to_remove.any())
        raise Exception("Error in sampling")
    return sampled_tokens


def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-1e4')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def get_dynamic_sparse_tree(model_path):
    if 'vicuna-13b' in model_path.lower():
        print('Using 13b 3-1 sparse trees')
        tree = dynamic_sparse_trees_3_vicuna_13b.dynamic_sparse_trees_60
    elif 'vicuna-7b' in model_path.lower():
        print('Using 7b 3-1 sparse trees')
        tree = dynamic_sparse_trees_3_vicuna_7b.dynamic_sparse_trees_105
    elif 'mobilellama' in model_path.lower():
        print('Using MobileLLaMA 3-1 sparse trees')
        tree = dynamic_sparse_trees_3_mobilellama.dynamic_sparse_trees_285
    else:
        raise ValueError("Unknown model path")
    return tree