#!/bin/bash
nohup python moe_peft.py --base_model /data1/llms/Llama-3.1-8B-Instruct/ --config moe_peft_sft.json --device cuda:0 > 711_sft.txt
cp -r casual_8B_1800/ casual_8B_1800_sft/
nohup python moe_peft.py --base_model /data1/llms/Llama-3.1-8B-Instruct/ --config moe_peft_dpo.json --device cuda:0 --load_adapter > 711_dpo.txt