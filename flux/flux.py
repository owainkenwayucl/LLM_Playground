# I'm going to do my best to make this super simple as the sdxl code is a mess!

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import FluxPipeline
from utils import report_state, init_rng

import time

model = "black-forest-labs/FLUX.1-dev"

def detect_platform():
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.bfloat16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}

    r = cpu

    if torch.cuda.device_count() > 0:
        print("Running on Cuda GPU")
        r = nvidia
        r["number"] = torch.cuda.device_count()
        print(f" - {r['number']} GPUs detected")
        # Fix for https://github.com/ROCm/pytorch/issues/1500
        #applied = False
        for a in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(a)
            #if device_name.startswith("AMD"):
            #    applied = True
            #    torch.backends.cuda.enable_mem_efficient_sdp(False)
            print(f"Detected Cuda Device {a}: {device_name}")

        #if applied:
        #    print(f"Disabled memory efficient flash attention as fix for https://github.com/ROCm/pytorch/issues/1500")

    elif torch.backends.mps.is_available():
        print("Running on Apple GPU")
        r = metal 

    return r

platform = detect_platform()

def setup_pipeline(model=model, cpu_offload=False):

    pipe = FluxPipeline.from_pretrained(model,torch_dtype=platform["size"])

    if cpu_offload:
        print(f"Enabling sequential cpu offload. This will massively decrease memory usage but may break device selection.")
        pipe.enable_sequential_cpu_offload()
    else:
        try:
            pipe.enable_vae_tiling()
        except:
            print(f"This version of diffusers doesn't support VAE tiling yet.")
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                print(f"Directly enabled VAE slicing + tiling")
            except:
                print(f"Can't directly enable VAE slicing + tiling either")
        pipe = pipe.to(platform["device"])

    return pipe

def inference(pipeline=None, prompt="", negative_prompt="", num_gen=1, num_iters=50, guidance_scale=3.5, seed=None, width=1024, height=1024):
    if pipeline == None:
        pipeline = setup_pipeline()

    generator = init_rng(platform, seed)
    
    images = []
    times = []
    for a in range(num_gen):
        t_s = time.time()
        temp_s = generator.get_state()
        report_state(temp_s)
        images.append(pipeline(prompt, generator=generator, num_inference_steps=num_iters, guidance_scale=guidance_scale, width=width, height=height).images[0])
        t_f = time.time()
        times.append(t_f - t_s)

    print(f"Timing Data: {times}")
    return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=50, guidance_scale=3.5, cpu_offload=False, seed=None, width=1024, height=1024):
    pipeline = setup_pipeline(cpu_offload=cpu_offload)
    images = inference(pipeline=pipeline,prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale, seed=seed, width=width, height=height)

    for a in images:
        display(a)