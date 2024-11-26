import warnings
warnings.filterwarnings("ignore")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import sys

# Read data out of data directory
documents = SimpleDirectoryReader("data").load_data()

# embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")

## STUFF TO SET UP LLM HERE
size="3.0-8b"
checkpoint_name = f"ibm-granite/granite-{size}-instruct"  

Settings.llm = HuggingFaceLLM(
    model_name=checkpoint_name,
    tokenizer_name=checkpoint_name,
    context_window=3900,
    max_new_tokens=256,
    generate_kwargs={"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="auto",
)

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()

while True:
    line = input("? ")
    if 'bye' == line.strip().lower():
        sys.exit()

    response = query_engine.query(line)
    print(response)
