timing = {}
# Imports
import time

start_imports = time.time()
import transformers
import torch
import sys
import copy
import time
import logging
import json
finish_imports = time.time()
timing["imports"] = finish_imports - start_imports

logging.disable(logging.WARNING)
bold_on = "\033[1m"
style_off = "\033[0m"

size="3.2-8b"
checkpoint_name = f"ibm-granite/granite-{size}-instruct"  
jobscript_directory = "/var/opt/sge/shared/saved_job_scripts/"

print(f"{bold_on}Starting up - Checkpoint = {style_off}{checkpoint_name}")

if torch.cuda.device_count() > 0:
    print(f"{bold_on}Detected {torch.cuda.device_count()} Cuda devices.{style_off}")
    for a in range(torch.cuda.device_count()):
        print(f"{bold_on}Detected Cuda Device {a}:{style_off} {torch.cuda.get_device_name(a)}")
else: 
    print(f"{bold_on}Running on CPU.{style_off}")

tokeniser = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
model = transformers.AutoModelForCausalLM.from_pretrained(
    checkpoint_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
finish_startup = time.time()
timing["model_setup"] = finish_startup - finish_imports

print(f"Loading imports took {timing["imports"]} seconds.")
print(f"Setting up the model took {timing["model_setup"]} seconds.")

prompt = "Please identify what application this job script is running. Please only name the application and do not provide any further detail:\n\n"

def identify_job(date,id):
    messages = []
    script = ""
    script_fn = f"{jobscript_directory}/{date}/{id}"

    try:
        with open(script_fn, mode='r', encoding="ISO-8859-1") as f:
            contents = f.read()
            script = contents.replace('\\\n','')
    except IOError as err:
        print(err, file=sys.stderr)
        continue
    except UnicodeDecodeError as err:
        print("File: ", script_fn, file=sys.stderr)
        print(err, file=sys.stderr)
        continue

    t_start = time.time()
     
    messages.append({"role":"user","content":f"{prompt}{script}"})
    input_ids = tokeniser.apply_chat_template(
        messages,
        return_tensors="pt",
        thinking=reasoning,
        add_generation_prompt=True,
        return_dict=True 
    ).to(model.device)

    terminators = [
        tokeniser.eos_token_id,
        tokeniser.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        **input_ids,
        max_new_tokens=8192,
    )
    response = tokeniser.decode(outputs[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)

    t_elapsed = time.time() - t_start
    return response, t_elapsed, script

def main():
    print("Don't run application this way")


if __name__ == "__main__":
    main()