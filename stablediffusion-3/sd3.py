# I'm going to do my best to make this super simple as the sdxl code is a mess!

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import StableDiffusion3Pipeline
from utils import report_state, init_rng

import time

model = "stabilityai/stable-diffusion-3-medium-diffusers"

def detect_platform():
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}
    habana = {"name": "Habana", "device":"hpu", "size":torch.float16, "attention_slicing":False}

    r = cpu

    if torch.cuda.device_count() > 0:
        print("Running on Nvidia GPU")
        r = nvidia
        r["number"] = torch.cuda.device_count()
        print(f" - {r['number']} GPUs detected")
    elif torch.backends.mps.is_available():
        print("Running on Apple GPU")
        r = metal 
    else:
        try:
            import habana_frameworks.torch.core as htcore
            print(f"Running on Habana Gaudi 2")
            r = habana

        except:
            pass
    return r

platform = detect_platform()

def setup_pipeline(model=model, exclude_t5=False, cpu_offload=False):
    if exclude_t5:
        pipe = StableDiffusion3Pipeline.from_pretrained(model, text_encoder_3=None, tokenizer_3=None, torch_dtype=platform["size"])
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(model,torch_dtype=platform["size"])

    if platform["name"] == "Habana":
        import habana_frameworks.torch.core as htcore
        #pipe.transformer.to("hpu") # not necessary it appears. 
        #pipe.vae.to("hpu")

    pipe = pipe.to(platform["device"])

    if cpu_offload:
        pipe.enable_module_cpu_offload()

    return pipe

def inference(pipeline=None, prompt="", negative_prompt="", num_gen=1, num_iters=28, guidance_scale=7.0, seed=None):
    if pipeline == None:
        pipeline = setup_pipeline()

    generator = init_rng(platform, seed)
    
    images = []
    times = []
    for a in range(num_gen):
        t_s = time.time()
        temp_s = generator.get_state()
        report_state(temp_s)
        images.append(pipeline(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=num_iters, guidance_scale=guidance_scale).images[0])
        t_f = time.time()
        times.append(t_f - t_s)

    print(f"Timing Data: {times}")
    return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=28, guidance_scale=7.0, exclude_t5=False, cpu_offload=False, seed=None):
    pipeline = setup_pipeline(exclude_t5=exclude_t5, cpu_offload=cpu_offload)
    images = inference(prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale, seed=seed)

    for a in images:
        display(a)