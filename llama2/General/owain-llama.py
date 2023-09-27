import huggingface_hub
from transformers import AutoTokenizer
import transformers
import torch
import os
import sys
import time

model_size = "7b"

checkpoint_name = f"meta-llama/Llama-2-{model_size}-chat-hf"    

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

llama_pipeline = transformers.pipeline(
    "text-generation",
    model=checkpoint_name,
    torch_dtype=torch.float32,
    device_map="auto",
)

print(f"Model preparation time: {time.time() - start}s")

_prompt = '''<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

'''

prompt = _prompt

DEBUG = True

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'forget' == line.strip().lower():
        prompt = _prompt
        continue

    start = time.time()

    prompt = prompt + " " + line + " [/INST] " 

    print("---")
    print(prompt)
    print("---")

    output = llama_pipeline(prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 200)[0]["generated_text"].split("[/INST]")[-1]
    print(output)

    prompt = prompt + " ".join(output) + " </s><s> [INST] "

    elapsed = time.time() - start
    print(" => Elapsed time: " + str(elapsed) + " seconds")