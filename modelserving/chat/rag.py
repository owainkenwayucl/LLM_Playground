# Pydantic causes lots of warnings on import. There's nothing we can do about this.
import warnings
warnings.filterwarnings("ignore")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
import torch
import sys
import configparser

# Read in configuration
config = configparser.ConfigParser()
config.read("llm.ini")

# Some aliases to make output nicer.
bold_on = "\033[1m"
style_off = "\033[0m"
avatar = "🤖"

# Set up LLM endpoint
endpoint = config["OPENAI"]["endpoint"]
model = config["OPENAI"]["model"]
api_key = config["OPENAI"]["api_key"]

# Set up embedding model endpoint
embedding_endpoint = config["EMBEDDING"]["endpoint"]
embedding_model = config["EMBEDDING"]["model"]
embedding_api_key = config["EMBEDDING"]["api_key"]

print(f"{bold_on}Starting up - embedding endpoint = {style_off}{embedding_endpoint}/{embedding_model}")

print(f"{bold_on}Starting up - LLM endpoint = {style_off}{endpoint}/{model}")

# This exposes a couple of functions for constructing prompts which are not *necessary* but seem to decrease surprising extra output.
#from hfhelper import messages_to_prompt, completion_to_prompt

# Read data out of data directory
directory = SimpleDirectoryReader(input_dir="data", recursive=True)
print(f"Loading {len(directory.list_resources())} documents from ./data...", end="", flush=True)
documents = directory.load_data()
print(f"done.")

# Add embedding model to the llama_index settings.
Settings.embed_model = OpenAILikeEmbedding(
    model_name=embedding_model,
    api_base=embedding_endpoint,
    api_key=embedding_api_key,
    embed_batch_size=10,
)


# Add the LLM to the llama_index settings.
Settings.llm = OpenAILike(
    model=model,
    api_base=endpoint,
    api_key=api_key,
    context_window=3900,
    is_chat_model=True,
    is_function_calling_model=False,
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
