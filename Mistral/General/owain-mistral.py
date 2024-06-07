import huggingface_hub
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
import sys
import time

model_size = "7B"
  
checkpoint_name = f"mistralai/Mistral-{model_size}-Instruct-v0.2"

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

pipeline = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=checkpoint_name,
    torch_dtype=torch.float16,
)

device="cuda"

print(f"Model preparation time: {time.time() - start}s")

_prompt = '''
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.


'''

__prompt = [{"role":"user","content": _prompt}]

prompt = __prompt
DEBUG = True

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    if 'forget' == line.strip().lower():
        prompt = __prompt
        continue

    start = time.time()

    if len(prompt) == 1:
        prompt = [{"role":"user","content": _prompt + line}]
    else:
        prompt.append({"role":"user","content": line})

    encodes = tokenizer.apply_chat_template(prompt, return_tensors="pt")

    pipeline_inputs = encodes.to(device)
    pipeline.to(device)

    _output = pipeline.generate(pipeline_inputs, max_new_tokens=1000, do_sample=True)

    output = tokenizer.batch_decode(_output_)[0]

    prompt.append[prompt.append({"role":"assistant","content": output})]

    print(output)

    elapsed = time.time() - start
    print(" => Elapsed time: " + str(elapsed) + " seconds")