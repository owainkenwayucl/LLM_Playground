import transformers
import torch
import sys
import copy

import logging
import warnings

# Transformers puts out a lot of spam
logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

size="8B"
checkpoint_name = f"meta-llama/Meta-Llama-3-{size}-Instruct"  

tokeniser = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    checkpoint_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages_ = [
    {"role": "system", "content": "You are a cute fluffy bear chatbot who always talks in uwu cute anime speak"},
]

messages = copy.deepcopy(messages_)
avatar = "🧸"

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

    line = line.strip()
    messages.append({"role":"user","content":line})

    input_ids = tokeniser.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokeniser.eos_token_id,
        tokeniser.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = tokeniser.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"{avatar} : {response}")

    messages.append({"role":"assistant","content":response})

