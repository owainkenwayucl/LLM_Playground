# Pydantic causes lots of warnings on import. There's nothing we can do about this.
import warnings
warnings.filterwarnings("ignore")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import sys

# Some aliases to make output nicer.
bold_on = "\033[1m"
style_off = "\033[0m"
avatar = "🤖"

# This exposes a couple of functions for constructing prompts which are not *necessary* but seem to decrease surprising extra output.
from hfhelper import messages_to_prompt, completion_to_prompt

# Read data out of data directory
documents = SimpleDirectoryReader("data").load_data()

# Embedding model - here we use one from the sentence-transformers library.,
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Choose which LLM to use - here we are using IBM's Granite 3.0 as it is very low on hallucinations.
size="3.0-8b"
checkpoint_name = f"ibm-granite/granite-{size}-instruct"  

Settings.llm = HuggingFaceLLM(
    model_name=checkpoint_name,
    tokenizer_name=checkpoint_name,
    context_window=3900,
    max_new_tokens=256,
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
