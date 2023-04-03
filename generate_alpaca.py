import argparse
# import sys
# sys.path.insert(0, './repository/transformers/src')
# sys.path.insert(0, './repository/GPTQ-for-LLaMa')
# sys.path.insert(0, './repository/peft/src')
import time
import torch
from peft import PeftModel
import autograd_4bit
from autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from peft.tuners.lora import Linear4bitLt

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def evaluate(
        model, 
        tokenizer,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        **kwargs,
):
    prompt = generate_prompt(
        instruction, 
        input if input != "" else None
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            repetition_penalty=1.1,
            #generation_config=generation_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,#2048,
            early_stopping=True,
            # output_attentions=False,
            # output_hidden_states=False,
            # output_scores=False            
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


def load_model_llama(*args, **kwargs):

    config_path = './llama-30b-4bit/'
    model_path = './llama-30b-4bit.pt'
    #lora_path = './alpaca_lora_30B/'
    lora_path = './lora-alpaca/'

    print("Loading {} ...".format(model_path))
    t0 = time.time()
    
    model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)
    
    model = PeftModel.from_pretrained(model, lora_path, device_map={'': 0}, torch_dtype=torch.float32)
    print('{} Lora Applied.'.format(lora_path))
    
    print('Apply auto switch and half')
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear) or isinstance(m, Linear4bitLt):
            m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()
    autograd_4bit.use_new = True
    autograd_4bit.auto_switch = True
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_lora_adapters", type=str, default="alpaca_lora")#default="baseten/alpaca-30b")
    parser.add_argument("--pretrained_model", type=str, default="decapoda-research/llama-30b-hf")
    args = parser.parse_args()
    
    # config_path = './llama-13b-4bit/'
    # #config_path = args.path_to_lora_adapters
    # model_path = './llama-13b-4bit.pt'
    # model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)

    # # model = PeftModel.from_pretrained(
    # #     model, 
    # #     args.path_to_lora_adapters,
    # #     torch_dtype=torch.float16,
    # #     device_map={'': 0}
    # # )

    # model = PeftModel.from_pretrained(model, args.path_to_lora_adapters)
    # print(args.path_to_lora_adapters, 'lora loaded')
    model,tokenizer = load_model_llama()


    print('Fitting 4bit scales and zeros to half')
    for n, m in model.named_modules():
        if '4bit' in str(type(m)):
            m.zeros = m.zeros.half()
            m.scales = m.scales.half()

    model.eval()

    # Setup input loop for user to type in instruction, recieve a response, and continue unless they type quit or exit
    input_str = ""
    print("Type quit or exit to exit this loop")
    while input_str != "quit" and input_str != "exit":
        instruction_str = input("Instruction: ")
        input_str = input("Input (optional): ")
        if not any(s in ("quit", "exit") for s in (input_str, instruction_str)):
            print('{{',evaluate(model, tokenizer, instruction_str, input_str),'}}')
        else:
            break