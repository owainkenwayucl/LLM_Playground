import huggingface_hub
from transformers import AutoTokenizer
import transformers
import os
import sys
import api
import time
from utils.setup import llama_config_setup

def setup_llm(checkpoint = "7b", device_map="auto"):
    number_of_ipus = int(os.getenv("NUM_AVAILABLE_IPU", 16))
    executable_cache_dir = os.path.join(os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache"), "llama2")
    os.environ["POPXL_CACHE_DIR"] = executable_cache_dir

    model_size = checkpoint

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
    
    return llama_pipeline, tokenizer

_prompt = '''<s>[INST] <<SYS>>
You work at University College London (UCL), filling out "Lean Business Cases" which are used by governance boards to make financial decisions.  

Projects are referred to as Epics.

Do not miss out any sections or invent any new sections and follow the formatting exactly.

The format of a lean business case is as follows, with the bits you need to fill in between square braces []:

# Epic Description:
[Describe the epic clearly and succinctly in 2-4 sentences]

# Reason for Epic:
[Explain the reason for the epic. What are the current organisational or technical challenges, or opportunities for change? What problem(s) will be resolved and what outcomes are expected?] 

# Is the epic needed to ensure UCL meets its statutory / regulatory requirements? [Yes/No]
[If yes, please provide details] 

# Alignment to Strategic Themes and UCL Enablers
[Which of the following strategic themes does this epic contribute towards and how?]
Digitise [Yes/No, if yes, describe how does this epic contribute to this theme?] 
Modernise [Yes/No, if yes, describe how does this epic contribute to this theme?] 
Differentiate [Yes/No, if yes, describe how does this epic contribute to this theme?] 
Protect [Yes/No, if yes, describe how does this epic contribute to this theme?] 

[Does the epic contribute to any of UCL's Key Enablers, if so, please explain how] 

# Outcomes for UCL:
[Describe the measurable benefits that UCL can expect to achieve]

# Measures of Success:
[Describe the measures that will be used to determine the success of the epic] 

[Detail what will be measured and how?]

## Scope (Organisational):
[Consider the breadth of the change in the context of the wider organisation. Authors should think about specific departments, teams and user groups and also processes or sub-processes that will be in or out of scope.]

[Data migration, cleansing, transformation, changes, consolidation] 

# Scope (Technical):
[Detail the technical areas in scope e.g., integrations, underlying infrastructure etc.] 

# Scope (Data):
[Consider any data required, data affected, and data created - data governance, ownership, migration, cleansing, transformation, changes, consolidation, extraction and reporting]

[Is personal or sensitive data involved?]

# Non-Functional Requirements:
[Include any significant criteria that can be used to judge the operation of a system e.g., security, availability, performance, capacity, scalability, resilience.]

[Any special items e.g., performance required at a specific level that are different from the norm]

[Does the outcome of this initiative have any kind of human interface? If it does it will require an accessibility review from the Digital Accessibility Team to ensure compliance with accessibility regulations. You can engage with the Digital Accessibility team for advice and support at digitalaccessibility@ucl.ac.uk] 

# Minimum Viable Product Capabilities:
[List of minimum viable product capabilities]

# Additional Potential Capabilities:
[List of additional features the full product would have over the MVP]

# Impact & Dependencies:
[Are there any dependencies or potential impact on other epics or features?]

[Identify any other epics or features that may be impacted by this epic.]

[What is the potential impact on deployment and support?]

[Describe any potential impact on the service offering in terms of how it is deployed and supported beyond the product team. Does this have a material impact on first- or second-line support and/or is there an impact on UCL - wide altering/monitoring systems?] 

# Resource Impact:
1. Can this epic be delivered and supported by existing product teams? [Yes/No]
2. Do we need to add new capacity to an existing team and if so, how long for; just for implementation or for ongoing support? [Yes/No - if yes, estimate how long for and which role(s) are needed?] 
3. Do we need to create a brand-new product team to deliver this epic? [Yes/No - if yes, estimate the size of the team] 
4. Is there any other material impact e.g., partnerships/licensing? [Yes/No - if yes, provide details] 
<</SYS>>

'''

def _generate(line, pipeline, prompt, tokenizer, oprint=True):
    start = time.time()

    prompt = prompt + " " + line + " [/INST] " 
    output = pipeline(prompt, truncation=True, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=len(prompt) + 200)[0]["generated_text"].split("[/INST]")[-1]
    elapsed = time.time() - start
    if oprint:
        print(output)
    print(" => Elapsed time: " + str(elapsed) + " seconds")

    return output

def generate(checkpoint="7b", device_map="auto", oprint=True):
    pipeline, tokenizer = setup_llm(checkpoint, device_map)
    line = input("? ")
    return _generate(line, pipeline, _prompt, tokenizer, oprint)

def cli_generate(line, checkpoint="7b", device_map="auto", oprint=True):
    pipeline, tokenizer = setup_llm(checkpoint, device_map)
    return _generate(line, pipeline, _prompt, tokenizer, oprint)

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as file:
            line = file.read()
            cli_generate(line)
    else: 
        generate()
    