import transformers
import torch
import sys
import copy
import time

import logging

logging.disable(logging.WARNING)
bold_on = "\033[1m"
style_off = "\033[0m"

size="17B-16E"
checkpoint_name = f"meta-llama/Llama-4-Scout-{size}-Instruct"  

print(f"{bold_on}Starting up - Checkpoint = {style_off}{checkpoint_name}")

if torch.cuda.device_count() > 0:
    print(f"{bold_on}Detected {torch.cuda.device_count()} Cuda devices.{style_off}")
    for a in range(torch.cuda.device_count()):
        print(f"{bold_on}Detected Cuda Device {a}:{style_off} {torch.cuda.get_device_name(a)}")
else: 
    print(f"{bold_on}Running on CPU.{style_off}")

tokeniser = transformers.AutoProcessor.from_pretrained(checkpoint_name)
model = transformers.Llama4ForConditionalGeneration.from_pretrained(
    checkpoint_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flex_attention",
    device_map="auto"
)

messages_ = [
]

messages = copy.deepcopy(messages_)
avatar = "🤖"


while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'reprogram' == line.strip().lower():
        print(f"Current prompt: {messages_}")
        print(f"Current avatar: {avatar}")

        new_prompt = input("New prompt: ")
        avatar = input("New avatar: ")

        messages_ = [
            {"role": "system", "content": new_prompt},
        ]
        messages = copy.deepcopy(messages_)
        continue
        
    if 'inspect' == line.strip().lower():
        print(f"Model: {checkpoint_name}")
        print(f"Chat state: {messages}")
        print(f"Avatar: {avatar}")
        continue


    if 'forget' == line.strip().lower():
        messages = copy.deepcopy(messages_)
        continue

    t_start = time.time()
    line = line.strip()
    messages.append({"role":"user","content":[{"type": "text", "text": line}]})
    transformers.set_seed(42) 
    input_ids = tokeniser.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True 
    ).to(model.device)

    terminators = [
        tokeniser.eos_token_id,
        tokeniser.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        **input_ids,
        max_new_tokens=8192,
    )
    response = tokeniser.decode(outputs[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)

    t_elapsed = time.time() - t_start
    print(f"{bold_on}---\n{avatar} :{style_off} {response}\n{bold_on}--- [{t_elapsed} seconds] {style_off}\n")

    messages.append({"role":"assistant","content":response})

