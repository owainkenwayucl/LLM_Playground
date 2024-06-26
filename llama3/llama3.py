import transformers
import torch
import sys

import logging

logging.disable(logging.WARNING)

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

messages = messages_

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'forget' == line.strip().lower():
        messages = messages_
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
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = tokeniser.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"🧸: {response}")

    messages.append({"role":"assistant","content":response})

