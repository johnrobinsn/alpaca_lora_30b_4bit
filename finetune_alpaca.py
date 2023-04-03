import argparse
import os
# import sys
# sys.path.insert(0, './repository/transformers/src')
# sys.path.insert(0, './repository/GPTQ-for-LLaMa')
# sys.path.insert(0, './repository/peft/src')

import peft
import peft.tuners.lora
assert peft.tuners.lora.is_gptq_available()

import time
import torch
from datasets import load_dataset
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import accelerate
#from modelutils import find_layers
from autograd_4bit import make_quant_for_4bit_autograd
from autograd_4bit import load_llama_model_4bit_low_ram
from datasets import load_dataset, Dataset
import json
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel,
)


# # Parameters
# DATA_PATH = "./data.txt"
OUTPUT_DIR = "alpaca_lora_30B"
lora_path_old = ''
config_path = './llama-30b-4bit/'
model_path = './llama-30b-4bit.pt'

MICRO_BATCH_SIZE = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
# EPOCHS = 3
LEARNING_RATE = 2e-4

# hyperparams
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

GRADIENT_CHECKPOINTING = True
GRADIENT_CHECKPOINTING_RATIO = 1
warmup_steps = 50
save_steps = 50
save_total_limit = 3
logging_steps = 10

if LORA_DROPOUT > 0 and GRADIENT_CHECKPOINTING:
    LORA_DROPOUT = 0
    print('Disable Dropout.')

def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt, tokenizer):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
    

def train(
    data_path: str,
    micro_batch_size: int,
    batch_size: int,
    warmup_steps: int,
    lr: float,
    epochs: int,
    report_to: str = "none",
    logging_steps: int = 20,
    eval_steps: int = 200,
    save_steps: int = 200,
    #model_pretrained_name: str = "decapoda-research/llama-30b-hf",
    output_dir: str = "lora-alpaca",
):
# Load Basic Model
    model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if lora_path_old == '':
        model = get_peft_model(model, config)
    else:
        model = PeftModel.from_pretrained(model, lora_path_old)
        print(lora_path_old, 'loaded')

    # Scales to half
    print('Fitting 4bit scales and zeros to half')
    for n, m in model.named_modules():
        if '4bit' in str(type(m)):
            m.zeros = m.zeros.half()
            m.scales = m.scales.half()
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    data = load_dataset("json", data_files=data_path)

    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"]
    val_data = train_val["test"]
    
        
    train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

    # Use gradient checkpointing
    if GRADIENT_CHECKPOINTING:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=GRADIENT_CHECKPOINTING_RATIO)


    # gradient_accumulation_steps = batch_size // micro_batch_size

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=OUTPUT_DIR,
            report_to=report_to if report_to else "none",
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    trainer.train()
    
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data_cleaned.json")
    parser.add_argument("--micro_batch_size", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)  
    parser.add_argument("--epochs", type=int, default=3) 
    parser.add_argument("--report_to", type=str, default="wandb") 
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200) 
    #parser.add_argument("--model_pretrained_name", type=str, default="decapoda-research/llama-13b-hf")
    parser.add_argument("--output_dir", type=str, default="lora-alpaca")
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        epochs=args.epochs,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        #model_pretrained_name=args.model_pretrained_name,
        output_dir=args.output_dir,
    )
