# I'm going to do my best to make this super simple as the sdxl code is a mess!

import torch
from diffusers import StableDiffusion3Pipeline

model = "stabilityai/stable-diffusion-3-medium-diffusers"

def detect_platform():
    import torch
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}

    r = cpu

    if torch.cuda.device_count() > 0:
        print("Running on Nvidia GPU")
        r = nvidia
        r["number"] = torch.cuda.device_count()
        print(f" - {r['number']} GPUs detected")
    elif torch.backends.mps.is_available():
        print("Running on Apple GPU")
        r = metal 

    return r

platform = detect_platform()

def setup_pipeline(model=model, exclude_t5=False, cpu_offload=False):
    if exclude_t5:
        pipe = StableDiffusion3Pipeline.from_pretrained(model, text_encoder_3=None, tokenizer_3=None, torch_dtype=platform["size"])
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(model,torch_dtype=platform["size"])
    pipe = pipe.to(platform["device"])

    if cpu_offload:
        pip.enable_module_cpu_offload()

    return pipe

def inference(pipeline=setup_pipeline(), prompt="", negative_prompt="", num_gen=1, num_iters=28, guidance_scale=0.7):
    images = []
    for a in num_gen:
        images.append(pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=num_iters, guidance_scale=guidance_scale).images[0])

    return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=28, guidance_scale=0.7, exclude_t5=False, cpu_offload=False):
    pipeline = setup_pipeline(exclude_t5=exclude_t5, cpu_offload=cpu_offload)
    images = inference(primpt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale)

    for a in images:
        display(a)