timing = {}
# Imports
import time

start_imports = time.time()
import transformers
import torch
import sys
import copy
import datetime
import logging
import json
import tqdm
import os
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

print(f"Loading imports took {timing['imports']} seconds.")
print(f"Setting up the model took {timing['model_setup']} seconds.")

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
    except UnicodeDecodeError as err:
        print("File: ", script_fn, file=sys.stderr)
        print(err, file=sys.stderr)

    t_start = time.time()
     
    messages.append({"role":"user","content":f"{prompt}{script}"})
    input_ids = tokeniser.apply_chat_template(
        messages,
        return_tensors="pt",
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

def daterange_generator(start_date, stop_date):
    for a in range(int ((stop_date - start_date).days)):
        yield start_date + datetime.timedelta(days=1)

def process_daterange(config):
    data = {}

    start_date = datetime.datetime.strptime(config["start_date"],'%Y-%m-%d').date()
    stop_date = datetime.datetime.strptime(config["stop_date"],'%Y-%m-%d').date()

    for current_date in tqdm.tqdm(daterange_generator(start_date, stop_date)):
        date_timing = 0
        data[current_date.isoformat()] = {}

        for jobfile in tqdm.tqdm(os.scandir(jobscript_directory + current_date.isoformat())):
            if os.path.isdir(jobscript_directory + current_date.isoformat()):
                if jobfile.is_file():
                    r, t, _ = identify_job(current_date.isoformat(), jobfile.name)
                    date_timing += t
                    data[current_date.isoformat()][jobfile.name] = r
        timing[current_date.isoformat()] = date_timing

    return data

def main():
    if (len(sys.argv) != 2):
        print("Run this program with a single argument that is a JSON settings file.")
    else:
        config_fn = sys.argv[1]
        with open(config_fn) as f:
            config = json.load(f)
            print(process_daterange(config))
            print(timing)

if __name__ == "__main__":
    main()