# Presently to work with Granite we have to override the requirements for llama_index as it thinks it 
# wants an older version of llama-cpp-python and we need the newest to support Granite.

# So
# pip install llama-index-llms-llama-cpp
# CMAKE_ARGS="-DGGML_CUDA=ON -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" pip install llama-cpp-python==0.3.5

# This works fine so not sure what the reqs in llama_index are on.

# Pydantic causes lots of warnings on import. There's nothing we can do about this.
import warnings
warnings.filterwarnings("ignore")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import torch
import sys
import os
import time

# Some aliases to make output nicer.
bold_on = "\033[1m"
style_off = "\033[0m"
avatar = "🤖"

# Choose which embedding model to use.
# embedding_model = f"sentence-transformers/all-MiniLM-L12-v2"
embedding_model = f"ibm-granite/granite-embedding-125m-english"
print(f"{bold_on}Starting up - embedding model = {style_off}{embedding_model}")

# Choose which LLM to use - here we are using IBM's Granite 3.0 as it is very low on hallucinations.
checkpoint_name = f"{os.path.expanduser('~')}/models/granite-3.0-8b-instruct/granite-3.0-8B-instruct-F16.gguf" 
print(f"{bold_on}Starting up - LLM checkpoint = {style_off}{checkpoint_name}")

# Work out if we have a GPU.
if torch.cuda.device_count() > 0:
    print(f"{bold_on}Detected {torch.cuda.device_count()} Cuda devices.{style_off}")
    for a in range(torch.cuda.device_count()):
        print(f"{bold_on}Detected Cuda Device {a}:{style_off} {torch.cuda.get_device_name(a)}")
else: 
    print(f"{bold_on}Running on CPU.{style_off}")

# This exposes a couple of functions for constructing prompts which are not *necessary* but seem to decrease surprising extra output.
from hfhelper import messages_to_prompt, completion_to_prompt

# Read data out of data directory
directory = SimpleDirectoryReader(input_dir="data", recursive=True)
print(f"Loading {len(directory.list_resources())} documents from ./data...", end="", flush=True)
documents = directory.load_data()
print(f"done.")

# Add embedding model to the llama_index settings.
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

# Add the LLM to the llama_index settings.
Settings.llm = LlamaCPP(
    model_path=checkpoint_name,
    context_window=4096,
    max_new_tokens=512,
    temperature=0.1,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 99},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=False,
)

# Set up the vector store index.
index = VectorStoreIndex.from_documents(
    documents,
)

# Initialise a query engine from the index.
query_engine = index.as_query_engine()

# Loop until user inputs "bye", sending their input to the query engine and returning the response.
while True:
    line = input("? ")
    st = time.time()
    if 'bye' == line.strip().lower():
        sys.exit()

    response = query_engine.query(line)
    fin = time.time()
    print(f"{bold_on}---\n{avatar} :{style_off} {response}\n{bold_on}---\nTime:{style_off} {fin - st}\n")
