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

	if (debug):
		print("Memory mode: " + memory)
	print("Setting up...")
	
	if remote:
		instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
	else: 
		tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
		model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
		instruct_pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

	instructions = []
	print("The sheep is now ready.")
	while True:
		line = input("? ")
		if 'bye' == line.strip().lower():
			sys.exit()

		if 'forget' == line.strip().lower():
			instructions = []
			continue

		instruction_text=""
		if (memory != "none"):
			if (len(instructions) > 0):
				instruction_text = "### Instruction: Your previous responses were:\n"
				for a in instructions:
					instruction_text = instruction_text + a + "\n"
				instruction_text = instruction_text + "### End\n"
				instruction_text = instruction_text + "\nPlease answer the following user query: "
		if 'prompt' == line.strip().lower():
			print("------ PROMPT ------")
			print(instruction_text)
			print("------ PROMPT ------")
			continue

		instruction_text = instruction_text + line

		if (debug):
			print("------ PROMPT ------")
			print(instruction_text)
			print("------ PROMPT ------")

		start = time.time()
		
		result = instruct_pipeline(instruction_text)

		elapsed = time.time() - start

		qr = ""
		if (memory == "both"):
			qr = "Question: " + line + "\nResponse:\n"
		for a in result:
			print("Sheep: " + a['generated_text'])
			qr = qr + a['generated_text']
			if (debug):
				print(" => Elapsed time: " + str(elapsed) + " seconds")
		instructions.append(qr)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Simple Front-end for Dolly v2.0.")
	parser.add_argument("-m", "--memory", metavar="memory", type=str, help="Memory style - allowed values are answer, none, both.", default="answer", choices=["none","answer","both"])
	parser.add_argument("-l", action="store_true", help="Enable local pipeline code execute")
	parser.add_argument("-d", action="store_true", help="Turn on Debug mode.", default=False)
	args = parser.parse_args()
	
	_main(memory=args.memory, debug=args.d, remote=not args.l)
