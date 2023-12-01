import huggingface_hub
from transformers import AutoTokenizer
import transformers
import torch
import os
import sys
import time

checkpoint_name = "codellama/CodeLlama-7b-Instruct-hf"

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

llama_pipeline = transformers.pipeline(
    "text-generation",
    model=checkpoint_name,
    torch_dtype=torch.float16,
    device_map="auto",
)


print(f"Model preparation time: {time.time() - start}s")

_prompt = '''<s>[INST] <<SYS>>
Provide answers in Python.
<</SYS>>

'''

prompt = _prompt

DEBUG = False

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'forget' == line.strip().lower():
        prompt = _prompt
        continue

    start = time.time()

    prompt = prompt + " " + line + " [/INST] " 

    if DEBUG:
        print("---")
        print(prompt)
    print("---")

    output = llama_pipeline(prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 1000)[0]["generated_text"].split("[/INST]")[-1]
    print(output)

    prompt = prompt + " " + output + " </s>\n<s> [INST] "

    elapsed = time.time() - start
    print(" => Elapsed time: " + str(elapsed) + " seconds")