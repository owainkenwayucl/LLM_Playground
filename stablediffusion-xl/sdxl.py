model_1_0_base="stabilityai/stable-diffusion-xl-base-1.0"
model_1_0_refiner="stabilityai/stable-diffusion-xl-refiner-1.0"

model = model_1_0_base
model_r = model_1_0_refiner

model_x2_latent_rescaler = "stabilityai/sd-x2-latent-upscaler"

default_prompt = "Space pineapple, oil paint"
default_fname = "output"

def prompt_to_filename(prompt):
    return prompt.replace(" ", "_").replace("/", "_")

def detect_platform():
    import torch
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    graphcore = {"name": "Graphcore", "device":"ipu", "size":torch.float16, "attention_slicing":False}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False}

    r = cpu
    try: 
        import poptorch
        print(f"Running on {n_ipu} Graphcore IPU(s)")
        r = graphcore
        print(f"Warning - at present time diffuses library does not support SDXL on Graphcore")
    except:
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

def setup_pipeline(model=model, model_r=model_r, refiner_enabled=True):
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=platform["size"], variant="fp16", add_watermarker=False)
    refiner = None
    if refiner_enabled:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_r, torch_dtype=platform["size"], variant="fp16", text_encoder_2=pipe.text_encoder_2, vae=pipe.vae)

    pipe.to(platform["device"])
    if refiner_enabled:
        refiner.to(platform["device"])

    return pipe,refiner

def setup_rescaler_pipeline(model=model_x2_latent_rescaler):
    from diffusers import StableDiffusionLatentUpscalePipeline
    import torch

    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(upscaler_model, torch_dtype=platform["size"])
    pipe.to(platform["device"])

    return pipe

def rescale(pipe, prompt, images, num_steps=40, fname=default_fname, save=True, start=0):
    r = []
    count = start
    for a in images:
        ir = pipe(prompt=prompt, image=image, num_inference_steps=num_steps, guidance_scale0).images[0]
        r.append(ir)
        if save:
            ir.save(f"{fname}_RESIZE_{count}.png")
        count = count + 1

    return r

def inference_denoise(pipe, refiner, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, denoise=0.8, save=True, start=0):
    images = []
    images_r = []
    for count in range(start, start+num_gen):
        image = pipe(prompt=prompt, num_inference_steps=pipe_steps, denoising_end=denoise).images[0]
        image_r = refiner(prompt=prompt, image=image, num_inference_steps=pipe_steps, denoising_start=denoise).images[0]
        if save:
            image.save(f"{fname}.png")
            image_r.save(f"{fname}_r.png")
        images.append(image)
        images_r.append(image_r)

    return images, images_r

def inference(pipe, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0:
    images = []
    for count in range(start, start+num_gen):
        image = pipe(prompt=prompt, num_inference_steps=pipe_steps).images[0]
        images.append(image)
        if save:
            image.save(f"{fname}_{count}.png")

    return images

def _inference_worker(q, model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0, rescale=False, rescale_steps=40):
    refiner = True
    if denoise == False:
        refiner = False
    pipe, pipe_r = setup_pipeline(model, model_r, refiner)
    if denoise == False:
        images = inference(pipe=pipe, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save, start=start)
    else:
        _,images = inference_denoise(pipe=pipe, refiner=pipe_r, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, denoise=denoise, save=save, start=start)
    if rescale:
        pipe_re = setup_rescaler_pipeline()
        images_r = rescale(pipe_re,prompt,a, rescale_steps, fname, save, start)
        images = images_r
    for a in images:
        q.put(a)

def parallel_inference(model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, rescale=False, rescale_steps=40:
    from torch.multiprocessing import Process, Queue, set_start_method
    import os

    # We only want to do this the first time as we get an error if we do it repeatedly.
    try:
        set_start_method("spawn")
    except:
        pass

    number = platform["number"]

    if num_gen < number:
        print(f"Number of images to generate < number of GPUs, setting number of GPUs to {num_gen}")
        number = num_gen
    
    # Decompose range into chunks
    chunks = [ int(num_gen/number) ] * number
    for a in range(num_gen - sum(chunks)):
        chunks[a] +=1

    starts = []
    starts_rt = 0

    for a in range(number):
        starts.append(starts_rt)
        starts_rt += chunks[a]

    q = Queue()
    procs = []

    images = []

    for a in range(number):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a)
        procs.append(Process(target=_inference_worker, args=(q, model, prompt, denoise, chunks[a], pipe_steps, fname, save, starts[a], rescale, rescale_steps)))
        procs[a].start()

    for a in range(num_gen):
        images.append(q.get())
    
    return images