import huggingface_hub
import os
import sys
import api
import time
from utils.setup import llama_config_setup

number_of_ipus = int(os.getenv("NUM_AVAILABLE_IPU", 16))
executable_cache_dir = os.path.join(os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache"), "llama2")
os.environ["POPXL_CACHE_DIR"] = executable_cache_dir

model_size = "7b"

checkpoint_name = f"meta-llama/Llama-2-{model_size}-chat-hf"    
config, *_ = llama_config_setup(
    "config/inference.yml", 
    "release", 
    f"llama2_{model_size}_pod4" if number_of_ipus == 4 else f"llama2_{model_size}_pod16"
)

sequence_length = 4096
micro_batch_size = 1

start = time.time()

llama_pipeline = api.LlamaPipeline(
    config, 
    sequence_length=sequence_length, 
    micro_batch_size=micro_batch_size,
    hf_llama_checkpoint=checkpoint_name
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

    output = llama_pipeline(prompt)
    print(output)

    prompt = prompt + " ".join(output) + " </s><s> [INST] "

    elapsed = time.time() - start
    print(" => Elapsed time: " + str(elapsed) + " seconds")