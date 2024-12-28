# Pydantic causes lots of warnings on import. There's nothing we can do about this.
import warnings
warnings.filterwarnings("ignore")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import sys

# Some aliases to make output nicer.
bold_on = "\033[1m"
style_off = "\033[0m"
avatar = "🤖"

# Choose which embedding model to use.
# embedding_model = f"sentence-transformers/all-MiniLM-L12-v2"
embedding_model = f"ibm-granite/granite-embedding-125m-english"
print(f"{bold_on}Starting up - embedding model = {style_off}{embedding_model}")

# Choose which LLM to use - here we are using IBM's Granite 3.0 as it is very low on hallucinations.
size="3.0-8b" # 3.1 does odd things - need to investigate
checkpoint_name = f"ibm-granite/granite-{size}-instruct"  
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
Settings.llm = HuggingFaceLLM(
    model_name=checkpoint_name,
    tokenizer_name=checkpoint_name,
    context_window=3900,
    max_new_tokens=512,
    generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
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
    if 'bye' == line.strip().lower():
        sys.exit()

    response = query_engine.query(line)
    print(f"{bold_on}---\n{avatar} :{style_off} {response}\n{bold_on}---{style_off}\n")
