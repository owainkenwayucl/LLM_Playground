# This is a deliberate hacky version of my usual pipeline to allow inference of Flux.2-dev in the specific environment where we have two GPUs with less than the
# required 120-odd GiB RAM, e.g. 2x Blackwell RTX6000 Pro Server edition or 2x A100

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from utils import report_state, init_rng
import time
import gc

from transformers import Mistral3ForConditionalGeneration

model = "black-forest-labs/FLUX.2-dev"
dtype = torch.bfloat16
platform = {"name": "Nvidia", "device":"cuda", "size":torch.bfloat16, "attention_slicing":False}

def setup_pipeline():
	text_encoder = Mistral3ForConditionalGeneration.from_pretrained(model, subfolder="text_encoder", torch_dtype=dtype).to("cuda:1")
	text_encoder_pipeline = Flux2Pipeline.from_pretrained(model, text_encoder=text_encoder, transformer=None, vae=None, torch_dtype=dtype)

	transformer = Flux2Transformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=dtype).to("cuda:0")
	pipe = Flux2Pipeline.from_pretrained(model, text_encoder=None, tokenizer=None, transformer=transformer, torch_dtype=dtype).to("cuda:0")

	return pipe, text_encoder_pipeline

def inference(pipe, text_encoder_pipeline, prompt="", negative_prompt="", num_gen=1, num_iters=50, guidance_scale=3.5, seed=None, width=1024, height=1024):
	with torch.no_grad():
		prompt_embeds = text_encoder_pipeline.encode_prompt(prompt=prompt)

		if isinstance(prompt_embeds, tuple):
			prompt_embeds = tuple(t.to("cuda:0") if torch.is_tensor(t) else t for t in prompt_embeds)
			embeds = prompt_embeds[0]
		else:
			embeds = prompt_embeds.to("cuda:0")
	generator = init_rng(platform, seed)
    
	images = []
	times = []
	for a in range(num_gen):
		t_s = time.time()
		temp_s = generator.get_state()
		report_state(temp_s)
		images.append(pipe(prompt_embeds=embeds, generator=generator, num_inference_steps=num_iters, guidance_scale=guidance_scale, width=width, height=height).images[0])
		t_f = time.time()
		times.append(t_f - t_s)

	print(f"Timing Data: {times}")
	return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=50, guidance_scale=3.5, cpu_offload=False, seed=None, width=1024, height=1024):
	torch.cuda.reset_peak_memory_stats()
	pipeline,text_encoder = setup_pipeline()
	images = inference(pipe=pipeline, text_encoder_pipeline=text_encoder,prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale, seed=seed, width=width, height=height)

	for a in images:
		display(a)

	del pipeline
	del text_encoder
	gc.collect()
	torch.cuda.empty_cache()
	print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_reserved() / (1024**3)} GiB.")
