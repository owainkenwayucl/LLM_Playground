def _main(memory="answer", debug=False):
	from transformers import pipeline
	import torch
	import sys

	print("Setting up...")
	instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

	instructions = []
	print("The sheep is now ready.")
	while True:
		line = input("? ")
		if 'bye' == line.strip().lower():
			sys.exit()

		instruction_text=""
		if (memory != "none"):
			if (len(instructions) > 0):
				instruction_text = "Your previous responses were:\n"
				i = 0
				for a in instructions:
					i = i + 1
					instruction_text = instruction_text + str(i) + ". " + a + "\n"
				instruction_text = instruction_text + "\nPlease answer the following user query: "

			instruction_text = instruction_text + line

		if (debug):
			print("------ PROMPT ------")
			print(instruction_text)
			print("------ PROMPT ------")

		result = instruct_pipeline(instruction_text)

		qr = ""
		if (memory == "both"):
			qr = "Question: " + line + "\nResponse:\n"
		for a in result:
			print("Sheep: " + a['generated_text'])
			qr = qr + a['generated_text']

		instructions.append(qr)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Simple Front-end for Dolly v2.0.")
	parser.add_argument("-m", "--memory", metavar="memory", type=str, help="Memory style - allowed values are answer, none, both.", default="answer", choices=["none","answer","both"])
	parser.add_argument("-d", action="store_true", help="Turn on Debug mode.", default=False)
	args = parser.parse_args()
	
	_main(memory=args.memory, debug=args.d)
