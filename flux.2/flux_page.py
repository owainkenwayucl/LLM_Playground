# This is a deliberate hacky version of my usual pipeline to allow inference of Flux.2-dev in GPUs with less than the
# required 120-odd GiB RAM, e.g. Blackwell RTX6000 Pro Server edition or A100

# It does this by using the same process as the 2 GPU version, but instead of using a second GPU, manually pages the models
# between CPU and GPU.

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from utils import report_state, init_rng
import time
import gc

from transformers import Mistral3ForConditionalGeneration

model = "black-forest-labs/FLUX.2-dev"
platform = {"name": "Nvidia", "device":"cuda", "size":torch.bfloat16, "attention_slicing":False}

def setup_pipeline():
	text_encoder = Mistral3ForConditionalGeneration.from_pretrained(model, subfolder="text_encoder", torch_dtype=platform["size"])
	text_encoder_pipeline = Flux2Pipeline.from_pretrained(model, text_encoder=text_encoder, transformer=None, vae=None, torch_dtype=platform["size"])

	transformer = Flux2Transformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=platform["size"])
	pipe = Flux2Pipeline.from_pretrained(model, text_encoder=None, tokenizer=None, transformer=transformer, torch_dtype=platform["size"])

	return pipe, text_encoder_pipeline

def inference(pipe, text_encoder_pipeline, prompt="", negative_prompt="", num_gen=1, num_iters=50, guidance_scale=3.5, seed=None, width=1024, height=1024):
	# Note we have to copy the embeds from GPU 1 to CPU and then to GPU 0 from CPU because otherwise Torch doesn't copy correctly.

	text_encoder_pipeline = text_encoder_pipeline.to("cuda:0")

	with torch.no_grad():
		prompt_embeds = text_encoder_pipeline.encode_prompt(prompt=prompt)

		if isinstance(prompt_embeds, tuple):
			prompt_embeds = tuple(t.to("cpu") if torch.is_tensor(t) else t for t in prompt_embeds)
			embeds = prompt_embeds[0]
		else:
			embeds = prompt_embeds.to("cpu")

	# for some reason if we don't copy back to CPU, it doesn't get pulled out of GPU memory which means I need to read the manual more.
	text_encoder_pipeline.to("cpu")
	del text_encoder_pipeline
	gc.collect()
	torch.cuda.empty_cache()

	pipe = pipe.to("cuda:0")
	embeds = embeds.to("cuda:0")
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
	del pipe
	gc.collect()
	torch.cuda.empty_cache()

	return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=50, guidance_scale=3.5, seed=None, width=1024, height=1024):
	torch.cuda.reset_peak_memory_stats()
	pipeline,text_encoder = setup_pipeline()
	images = inference(pipe=pipeline, text_encoder_pipeline=text_encoder,prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale, seed=seed, width=width, height=height)

	for a in images:
		display(a)

	print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_reserved() / (1024**3)} GiB.")
