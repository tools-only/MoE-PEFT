import os
import json
import copy
import torch
from torch import nn
import argparse
from tqdm import tqdm

import moe_peft
import moe_peft.adapters
from transformers import AutoModelForCausalLM, AutoTokenizer
from moe_peft.utils import encode_persona_to_vector, preference_mapping

from moe_peft.common import (
    CHECKPOINT_CLASSES,
    AdapterConfig,
    Linear,
    LLMCache,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    LLMModelOutput,
    LLMMoeBlock,
    LLMOutput,
    LoraConfig,
    unpack_router_logits,
    LLMBatchConfig
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM evaluation with specified parameters")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True, 
        choices=["base", "moe", "pba"], 
        help="Type of model to evaluate (base, moe, or pba)"
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        default=None, 
        help="Path to LoRA weights (required when model_type is 'moe')"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["PRISM", "P-Soups", "UF-P-4", 'AlignX-test/Reddit_UGC', 'AlignX-test/Reddit_PAIR', 'AlignX-test/Reddit_DEMO', 'AlignX-test/Reddit_UGC_single_16'], 
        help="Dataset to evaluate on (PRISM, P-Soups, UF-P-4), or AlignX-test"
    )

    args = parser.parse_args()

    if args.model_type == "moe" and args.weights is None:
        parser.error("--weights is required when --model_type is 'moe'")

    if args.model_type != "moe" and args.weights is not None:
        print(f"Warning: --weights is provided but will be ignored for model_type '{args.model_type}'")

    return args

def main():
    args = parse_args()
    file_path = "/data1/zq/benchmark/"
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.pad_token_id = 0  
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left" 

    # Initialize model
    flash_attn = None
    load_4bit, load_8bit = None, None
    load_16bit = None

    MAX_LENGTH = 8192
    MAX_NEW_LENGTH = 512
    IGNORE_INDEX = -100

    if args.model_type == 'moe':
        model = moe_peft.LLMModel.from_pretrained(
            args.model_path,
            device=args.device,
            attn_impl="flash_attn" if flash_attn else "eager",
            bits=(8 if load_8bit else (4 if load_4bit else None)),
            load_dtype=torch.bfloat16 if load_16bit else torch.float32,
        )
        model.load_adapter(args.weights, args.weights)
    elif args.model_type == 'pba':
        model = AutoModelForCausalLM.from_pretrained(args.weights)
        model = model.to(args.device)
        model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model = model.to(args.device)
        model.eval()

    def mask_from(tokens):
        mask_tokens = [IGNORE_INDEX]
        return [int(tok not in mask_tokens) for tok in tokens[0]]

    # Load dataset
    ds = []
    if args.dataset == 'PRISM':
        file_path = file_path + "PRISM.json"
        with open(file_path, "r") as f:
            ds = json.load(f)
    elif args.dataset == 'P-Soups':
        file_path = file_path + "P-Soups/"
        files = os.listdir(file_path)
        for file in files:
            with open(os.path.join(file_path, file), "r") as f:
                tmp = json.load(f)
                ds.extend(tmp)
    elif args.dataset == 'UF-P-4':
        file_path = file_path + "UF-P-4/"
        files = os.listdir(file_path)
        for file in files:
            with open(os.path.join(file_path, file), "r") as f:
                tmp = json.load(f)
                ds.extend(tmp)
    elif 'AlignX-test' in args.dataset:
        file_path = file_path + args.dataset + '.json'
        with open(file_path, "r") as f:
            ds = json.load(f)

    # Prepare output path
    if args.model_type in ['moe', 'pba']:
        adapter_name = args.weights.split('/')[-1]
    else:
        adapter_name = 'base'

    output_path = f"./eval/{args.dataset}_{adapter_name}_eval.jsonl"
    assert not os.path.exists(output_path), f"Output file {output_path} already exists! Please delete it or change output path."

    # Begin evaluation
    for idx, data in tqdm(enumerate(ds), total=len(ds)):
        task = f"**Post:**\n{data['prompt']}\n\n"
        persona = data["profile"]

        sft_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\nGenerate a task-specific response based on user preferences.\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"***Task***\n\n{task}"
            f"***User Preferences***\n\n{persona}\n\n***Response:***\n\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>"
        )
        data["format"] = sft_prompt
        data["idx"] = idx

        prompt = data["format"]
        chosen = data["chosen"]
        rejected = data["rejected"]
        preference = encode_persona_to_vector(data['profile'])

        examples_inputs = tokenizer(prompt, padding=False, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH-MAX_NEW_LENGTH-1)["input_ids"]
        chosen_inputs = tokenizer(chosen, padding=False, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH-MAX_NEW_LENGTH-1)["input_ids"]
        rejected_inputs = tokenizer(rejected, padding=False, truncation=True, add_special_tokens=False, max_length=MAX_LENGTH-MAX_NEW_LENGTH-1)["input_ids"]

        examples_inputs = [tokenizer.bos_token_id] + examples_inputs
        prompt_length = len(examples_inputs)
        en = examples_inputs + [tokenizer.eos_token_id]
        en_chosen = examples_inputs + chosen_inputs + [tokenizer.eos_token_id]
        en_rejected = examples_inputs + rejected_inputs + [tokenizer.eos_token_id]

        label_chosen = copy.deepcopy(en_chosen)
        label_chosen[:prompt_length] = [IGNORE_INDEX] * prompt_length

        label_rejected = copy.deepcopy(en_rejected)
        label_rejected[:prompt_length] = [IGNORE_INDEX] * prompt_length

        with torch.no_grad():
            en = torch.tensor(en).reshape(1,-1)
            en_chosen = torch.tensor(en_chosen).reshape(1,-1).to(args.device)
            en_rejected = torch.tensor(en_rejected).reshape(1,-1).to(args.device)
            label_chosen = torch.tensor(label_chosen).reshape(1,-1).to(args.device)
            label_rejected = torch.tensor(label_rejected).reshape(1,-1).to(args.device)

            if args.model_type == 'moe':
                lora_batch_data_config = []
                lora_batch_data_config.append(
                    LLMBatchConfig(
                        adapter_name_=adapter_name,
                        batch_start_idx_=0,
                        batch_end_idx_=2,
                    )
                )

                input_args = LLMModelInput(
                        batch_configs_=lora_batch_data_config,
                        batch_tokens_ = en_chosen,
                        batch_chosen_tokens_=en_chosen,
                        batch_chosen_tokens_labels_=label_chosen,
                        batch_chosen_masks_ = [mask_from(en_chosen)],
                        batch_rejected_tokens_ = en_rejected,
                        batch_rejected_tokens_labels_ = label_rejected,
                        batch_rejected_masks_ = [mask_from(en_rejected)],
                        batch_preference_= [preference],
                        inference_mode_=True,
                        router_soft_mask_=False,
                    )
                result = model.dpo_eval(input_args)
            else:
                outputs_chosen = model(input_ids = en_chosen)
                outputs_rejected = model(input_ids = en_rejected)
                ##chosen
                logits = outputs_chosen["logits"] if isinstance(outputs_chosen, dict) else outputs_chosen[0]
                logits = logits[..., :-1, :].contiguous()
                label = torch.tensor(label_chosen).reshape(1,-1)
                label = label[..., 1:].contiguous()

                log_probs = -nn.functional.log_softmax(logits, dim=-1)
                if label.dim() == log_probs.dim() - 1:
                    label = label.unsqueeze(-1)
                
                padding_mask_all = label.eq(IGNORE_INDEX)
                label = torch.clamp(label, min=0)
                nll_loss_all = log_probs.gather(dim=-1, index=label)
                nll_loss_all.masked_fill_(padding_mask_all, 0.0)
                num_active_elements_all = padding_mask_all.numel() - padding_mask_all.long().sum()
                nll_loss_all_chosen = nll_loss_all.sum()

                ##rejected
                logits = outputs_rejected["logits"] if isinstance(outputs_rejected, dict) else outputs_rejected[0]
                logits = logits[..., :-1, :].contiguous()
                label = torch.tensor(label_rejected).reshape(1,-1)
                label = label[..., 1:].contiguous()

                log_probs = -nn.functional.log_softmax(logits, dim=-1)
                if label.dim() == log_probs.dim() - 1:
                    label = label.unsqueeze(-1)
                
                padding_mask_all = label.eq(IGNORE_INDEX)
                label = torch.clamp(label, min=0)
                nll_loss_all = log_probs.gather(dim=-1, index=label)
                nll_loss_all.masked_fill_(padding_mask_all, 0.0)
                num_active_elements_all = padding_mask_all.numel() - padding_mask_all.long().sum()
                nll_loss_all_rejected = nll_loss_all.sum()

                result = {
                        'nll_loss_all_chosen': nll_loss_all_chosen.item(),
                        'nll_loss_all_rejected': nll_loss_all_rejected.item()
                    }

            data['nll_loss_all_chosen'] = result['nll_loss_all_chosen']
            data['nll_loss_all_rejected'] = result['nll_loss_all_rejected']

            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()