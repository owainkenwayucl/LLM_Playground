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

print(f"{bold_on}Starting up - Checkpoint = {style_off}{local_checkpoint_name}")

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

pipe = transformers.pipeline("text-generation", model=model, tokenizer=tokeniser, device=0)

# Define test prompts
prompts = [
    "What is the capital of Germany?",
    "Write a Python function to calculate the factorial of a number.",
]

# Generate outputs
for prompt in prompts:
    print(pipe(prompt))
