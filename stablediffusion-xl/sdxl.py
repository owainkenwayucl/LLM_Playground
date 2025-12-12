# I'm going to do my best to make this super simple as the old sdxl code is a mess!

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import StableDiffusionXLPipeline
from utils import report_state, init_rng

import time

model = "stabilityai/stable-diffusion-xl-base-1.0"

def detect_platform():
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}

    r = cpu

    if torch.cuda.device_count() > 0:
        print("Running on Cuda GPU")
        r = nvidia
        r["number"] = torch.cuda.device_count()
        print(f" - {r['number']} GPUs detected")
        for a in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(a)
            print(f"Detected Cuda Device {a}: {device_name}")

    elif torch.backends.mps.is_available():
        print("Running on Apple GPU")
        r = metal 

    return r

platform = detect_platform()

def setup_pipeline(model=model, cpu_offload=False):

    pipe = StableDiffusionXLPipeline.from_pretrained(model,torch_dtype=platform["size"], variant="fp16", add_watermarker=False)

    if cpu_offload:
        print(f"Enabling sequential cpu offload. This will massively decrease memory usage but may break device selection.")
        pipe.enable_sequential_cpu_offload()
    else:
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            print(f"Directly enabled VAE slicing + tiling")
        except:
            print(f"Can't directly enable VAE slicing + tiling either")
        pipe = pipe.to(platform["device"])

    return pipe

def inference(pipeline=None, prompt="", negative_prompt="", num_gen=1, num_iters=50, seed=None):
    if pipeline == None:
        pipeline = setup_pipeline()

    generator = init_rng(platform, seed)
    
    images = []
    times = []
    for a in range(num_gen):
        t_s = time.time()
        temp_s = generator.get_state()
        report_state(temp_s)
        images.append(pipeline(prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=num_iters).images[0])
        t_f = time.time()
        times.append(t_f - t_s)

    print(f"Timing Data: {times}")
    return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=50,  cpu_offload=False, seed=None):
    pipeline = setup_pipeline(cpu_offload=cpu_offload)
    images = inference(pipeline=pipeline,prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, seed=seed)

    for a in images:
        display(a)
