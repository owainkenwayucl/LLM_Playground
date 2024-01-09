import huggingface_hub
from transformers import AutoTokenizer
import transformers
import torch
import os
import sys
import time

def setup_llm(checkpoint = "7b", device_map="auto"):


    model_size = checkpoint

    checkpoint_name = f"meta-llama/Llama-2-{model_size}-chat-hf"    

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)

    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=checkpoint_name,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    print(f"Model preparation time: {time.time() - start}s")
    
    return llama_pipeline, tokenizer

_prompt = '''<s>[INST] <<SYS>>You are a professional proof reader who helps prepare academic documents and adjust them for tone and content. You have been presented with a text file that
             needs to be tidied and corrected before sending it to the funding agency.  Please correct the following file.<</SYS>>'''

def _generate(pipeline, prompt, tokenizer, oprint=True):
   
    start = time.time()

    output = pipeline(prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 200)[0]["generated_text"].split("[/INST]")[-1]
    elapsed = time.time() - start
    if oprint:
        print(output)
    print(" => Elapsed time: " + str(elapsed) + " seconds")

    return output

def generate(filename, checkpoint="7b", device_map="auto", oprint=True):
    pipeline, tokenizer = setup_llm(checkpoint, device_map)
    prompt = _prompt
    file_data = ""
    with open(filename, encoding="utf-8") as f:
        file_data = f.read()

    prompt = prompt +"\n" + file_data + "[/INST]"
    return _generate(pipeline, prompt, tokenizer, oprint)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        generate(sys.argv[1])
    else:
        print("Run with filename as argument.")