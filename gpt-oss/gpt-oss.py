import transformers
import torch
import sys
import copy
import time

import logging

logging.disable(logging.WARNING)
bold_on = "\033[1m"
thought_on = "\033[32m"
style_off = "\033[0m"

size="20b"
checkpoint_name = f"openai/gpt-oss-{size}"  

print(f"{bold_on}Starting up - Checkpoint = {style_off}{checkpoint_name}")

if torch.cuda.device_count() > 0:
    print(f"{bold_on}Detected {torch.cuda.device_count()} Cuda devices.{style_off}")
    for a in range(torch.cuda.device_count()):
        print(f"{bold_on}Detected Cuda Device {a}:{style_off} {torch.cuda.get_device_name(a)}")
else: 
    print(f"{bold_on}Running on CPU.{style_off}")

tokeniser = transformers.AutoTokenizer.from_pretrained(checkpoint_name, padding_side="left")
model = transformers.AutoModelForCausalLM.from_pretrained(
    checkpoint_name,
    torch_dtype="auto",
    device_map="auto"
)

reasoning = "low"

base_prompt=""

messages_ = [{"role": "system", "content": f"{base_prompt}\nReasoning: {reasoning}"}
]

messages = copy.deepcopy(messages_)
avatar = "🤖"


while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'reprogram' == line.strip().lower():
        print(f"Current prompt: {base_prompt}")
        print(f"Current avatar: {avatar}")
        print(f"Current Reasoning: {reasoning}")
        
        base_prompt = input("New prompt: ")
        avatar = input("New avatar: ")
        reasoning = input("Reasoning (low, medium, high): ")
        messages_ = [
            {"role": "system", "content": f"{base_prompt}\nReasoning: {reasoning}"},
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
        max_new_tokens=256,
    )
    response = tokeniser.decode(outputs[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)

    decoded_response = response.split("assistantfinal")
    if len(decoded_response) > 1:
        reasoning = decoded_response[0][8:]
        response = decoded_response[1]

        t_elapsed = time.time() - t_start
        print(f"{bold_on}---\n💭 : {style_off}{thought_on}{reasoning}{style_off}\n{bold_on}---\n{avatar} : {style_off}{response}\n{bold_on}--- [{t_elapsed} seconds] {style_off}\n")
    else:
        t_elapsed = time.time() - t_start
        print(f"{bold_on}---\n{avatar} : {style_off}{response}\n{bold_on}--- [{t_elapsed} seconds] {style_off}\n")
    
    messages.append({"role":"assistant","content":response})

