# I'm going to do my best to make this super simple as the sdxl code is a mess!

from diffusers.utils import logging
logging.set_verbosity_error() # Decrease somewhat unhinged log spam!

import torch
from diffusers import FluxPipeline, DiffusionPipeline
from transformers import T5EncoderModel
from utils import report_state, init_rng

import time

model = "black-forest-labs/FLUX.1-dev"

def detect_platform():
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.bfloat16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}
    habana = {"name": "Habana", "device":"hpu", "size":torch.float16, "attention_slicing":False}

    r = cpu

    if torch.cuda.device_count() > 0:
        print("Running on Cuda GPU")
        r = nvidia
        r["number"] = torch.cuda.device_count()
        print(f" - {r['number']} GPUs detected")
        # Fix for https://github.com/ROCm/pytorch/issues/1500
        applied = False
        for a in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(a)
            if device_name.startswith("AMD"):
                applied = True
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            print(f"Detected Cuda Device {a}: {device_name}")

        if applied:
            print(f"Disabled memory efficient flash attention as fix for https://github.com/ROCm/pytorch/issues/1500")

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

def setup_pipeline(model=model, cpu_offload=False):

    t5_encoder = T5EncoderModel.from_pretrained(model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    text_encoder = DiffusionPipeline.from_pretrained(model, text_encoder_2 = t5_encoder, transformer = None, vae = None)
    pipe = FluxPipeline.from_pretrained(model,torch_dtype=platform["size"], text_encoder = None, text_encoder_2 = None)

    if platform["name"] == "Habana":
        import habana_frameworks.torch.core as htcore
 
    pipe = pipe.to(platform["device"])

    pipe.enable_vae_tiling()
    # pipe.enable_vae_slicing()

    if cpu_offload:
        pipe.enable_model_cpu_offload()

    return pipe, text_encoder

def inference(pipeline=None, text_encoder=None, prompt="", negative_prompt="", num_gen=1, num_iters=50, guidance_scale=3.5, seed=None, width=1024, height=1024):
    if pipeline == None:
        pipeline, text_encoder = setup_pipeline()

    generator = init_rng(platform, seed)
    
    images = []
    times = []

    print("Encoding prompt on CPU (to save VRAM)")
    e_s = time.time()
    (prompt_embeds, pooled_prompt_embeds, _) = text_encoder.encode_prompt(prompt = prompt, prompt_2 = None, max_sequence_length = 256)

    prompt_embeds = prompt_embeds.bfloat16().to(platform["device"])
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16().to(platform["device"])
    e_f = time.time()
    print(f"Prompt encoding time: {e_f - e_s} seconds.")

    for a in range(num_gen):
        t_s = time.time()
        temp_s = generator.get_state()
        report_state(temp_s)
        images.append(pipeline(prompt_embeds=prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, num_inference_steps=num_iters, guidance_scale=guidance_scale, width=width, height=height).images[0])
        t_f = time.time()
        times.append(t_f - t_s)

    print(f"Timing Data: {times}")
    return images

def interactive_inference(prompt="", negative_prompt="",num_gen=1, num_iters=50, guidance_scale=3.5, cpu_offload=False, seed=None, width=1024, height=1024):
    pipeline, text_encoder = setup_pipeline(cpu_offload=cpu_offload)
    images = inference(pipeline=pipeline, text_encoder=text_encoder ,prompt=prompt, negative_prompt=negative_prompt, num_gen=num_gen, num_iters=num_iters, guidance_scale=guidance_scale, seed=seed, width=width, height=height)

    for a in images:
        display(a)