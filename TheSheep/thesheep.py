def _main():
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
		if (len(instructions) > 0):
			instruction_text = "Your previous responses were:\n"
			i = 0
			for a in instructions:
				i = i + 1
				instruction_text = instruction_text + str(i) + ". " + a + "\n"
			instruction_text = instruction_text + "\nPlease answer the following user query: "

		instruction_text = instruction_text + line

		result = instruct_pipeline(instruction_text)
		qr = "Question: " + line + "\nResponse:"
		for a in result:
			print("Sheep: " + a['generated_text'])
			qr = qr + "\n" + a['generated_text']

		instructions.append(qr)

if __name__ == "__main__":
	_main()