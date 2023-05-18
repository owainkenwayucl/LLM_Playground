def setup_pipeline():
	import torch
	from instruct_pipeline import InstructionTextGenerationPipeline
	from transformers import AutoModelForCausalLM, AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
	model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
	pipeline = InstructionTextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, return_full_text=True)

	return pipeline

def setup_conversation(pipeline, debug=True):
	from langchain import ConversationChain
	from langchain.llms import HuggingFacePipeline

	llm = HuggingFacePipeline(pipeline=pipeline)
	conversation = ConversationChain(llm=llm, verbose=debug)
	
	return conversation

def simple_repl(debug=True):
	import time

	print("Setting up...")
	start = time.time()
	conversation = setup_conversation(setup_pipeline(), debug=debug)
	print("Finished - time taken: " + str(time.time() - start))

	line = input("? ")

	while not line.lower().strip() == "bye":
		print(conversation.predict(input=line))
		line = input("? ")
	
