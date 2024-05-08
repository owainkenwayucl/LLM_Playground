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
Provide answers only in Python. DO NOT EXPLAIN THE CODE UNDER ANY CIRCUMSTANCES.
<</SYS>>

'''

prompt = _prompt

DEBUG = False

def do_inference(userprompt):
    local_prompt = prompt + " " + userprompt + " [/INST] "
    code = llama_pipeline(local_prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 1000)[0]["generated_text"].split("[/INST]")[-1]
    codelist = code.split("```")
    for a in codelist:
        if "def " in a:
            code = a
    return code

def main():
    n = 10
    userprompt = "Write a program to estimate pi"

    for a in range(n):
        code = do_inference(userprompt)
        print(code)

if __name__ == "__main__":
    main()