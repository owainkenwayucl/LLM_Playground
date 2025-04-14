import transformers
import torch
import sys
import copy
import time
import trl

import logging

logging.disable(logging.WARNING)
bold_on = "\033[1m"
style_off = "\033[0m"

size="3b"
local_checkpoint_name = f"./Peft_wgts_merged"
checkpoint_name = f"britllm/britllm-{size}-v0.1"

print(f"{bold_on}Starting up - Checkpoint = {style_off}{checkpoint_name}")

if torch.cuda.device_count() > 0:
    print(f"{bold_on}Detected {torch.cuda.device_count()} Cuda devices.{style_off}")
    for a in range(torch.cuda.device_count()):
        print(f"{bold_on}Detected Cuda Device {a}:{style_off} {torch.cuda.get_device_name(a)}")
else:
    print(f"{bold_on}Running on CPU.{style_off}")

tokeniser = transformers.AutoTokenizer.from_pretrained(checkpoint_name)



model = transformers.AutoModelForCausalLM.from_pretrained(
    local_checkpoint_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

_, tokeniser = trl.setup_chat_format(model=model, tokenizer=tokeniser)

avatar = "🇬🇧🤖"

messages_ = [
]

messages = copy.deepcopy(messages_)


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
        print(f"Reasoning: {reasoning}")
        continue


    if 'forget' == line.strip().lower():
        messages = copy.deepcopy(messages_)
        continue

    t_start = time.time()
    line = line.strip()
    messages.append({"role":"user","content":line}) 
    input_ids = tokeniser.apply_chat_template(
        messages,
        return_tensors="pt",
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


