def _main(memory="answer", debug=False, remote=False):
	if remote:
		from transformers import pipeline
	import torch

	# local instruct pipeline
	# requires you to download instruct_pipeline.py from:
	#   https://huggingface.co/databricks/dolly-v2-3b/blob/main/instruct_pipeline.py
	# and put it in your PYTHONPATH
	if not remote:
		from instruct_pipeline import InstructionTextGenerationPipeline
		from transformers import AutoModelForCausalLM, AutoTokenizer

	import sys
	import time

	from langchain import ConversationChain
	from langchain.llms import HuggingFacePipeline

	if (debug):
		print("Memory mode: " + memory)
	print("Setting up...")
	
	if remote:
		instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True)
	else: 
		tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
		model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
		instruct_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer, return_full_text=True)

	llm = HuggingFacePipeline(pipeline=instruct_pipeline)
	conversation = ConversationChain(llm=llm, verbose=True)

	instructions = []
	print("The sheep is now ready.")
	if (debug):
				print("(this version uses langchain)")
	while True:
		line = input("? ")
		if 'bye' == line.strip().lower():
			sys.exit()

		if 'forget' == line.strip().lower():
			conversation = ConversationChain(llm=llm, verbose=True)
			continue

		start = time.time()

		output = conversation.predict(input=line)
		print(output)

		elapsed = time.time() - start
		if (debug):
				print(" => Elapsed time: " + str(elapsed) + " seconds")

		if memory == "none":
			conversation = ConversationChain(llm=llm, verbose=True)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Simple Front-end for Dolly v2.0.")
	parser.add_argument("-m", "--memory", metavar="memory", type=str, help="Memory style - allowed values are answer, none, both.", default="answer", choices=["none","answer","both"])
	parser.add_argument("-l", action="store_true", help="Enable local pipeline code execute")
	parser.add_argument("-d", action="store_true", help="Turn on Debug mode.", default=False)
	args = parser.parse_args()
	
	_main(memory=args.memory, debug=args.d, remote=not args.l)
