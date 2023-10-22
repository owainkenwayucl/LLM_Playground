model_1_0_base="stabilityai/stable-diffusion-xl-base-1.0"
model_1_0_refiner="stabilityai/stable-diffusion-xl-refiner-1.0"

model = model_1_0_base
model_r = model_1_0_refiner

default_prompt = "Space pineapple, oil paint"
default_fname = "output"

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

def setup_pipeline(model=model, model_r=model_r):
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", add_watermarker=False)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_r, torch_dtype=torch.float16, variant="fp16", text_encoder_2=pipe.text_encoder_2, vae=pipe.vae)

    pipe.to("cuda")
    refiner.to("cuda")

    return pipe,refiner

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

def inference(pipe, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0):
    images = []
    for count in range(start, start+num_gen):
        image = pipe(prompt=prompt, num_inference_steps=pipe_steps).images[0]
        images.append(image)
        if save:
            image.save(f"{fname}_{count}.png")

    return images

def _inference_worker(q, model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0):
    pipe, pipe_r = setup_pipeline(model, model_r)
    if denoise == False:
        images = inference(pipe=pipe, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save, start=start)
    else:
        _,images = inference_denoise(pipe=pipe, refiner=pipe_r, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, denoise=denoise, save=save, start=start)
    for a in images:
        q.put(a)

def parallel_inference(model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True):
    from torch.multiprocessing import Process, Queue, set_start_method
    import os

    set_start_method("spawn")

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
        procs.append(Process(target=_inference_worker, args=(q, model, prompt, denoise, chunks[a], iterations, save, starts[a])))
        procs[a].start()

    for a in range(num_gen):
        images.append(q.get())
    
    return images