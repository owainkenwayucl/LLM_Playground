def setup_pipeline():
	import torch
	from instruct_pipeline import InstructionTextGenerationPipeline
	from transformers import AutoModelForCausalLM, AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
	model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto", torch_dtype=torch.bfloat16)
	pipeline = InstructionTextGenerationPipeline(model=model, task="text-generation", tokenizer=tokenizer, return_full_text=True)

	return pipeline
