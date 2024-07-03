model_1_0_base="stabilityai/stable-diffusion-xl-base-1.0"
model_1_0_refiner="stabilityai/stable-diffusion-xl-refiner-1.0"

model = model_1_0_base
model_r = model_1_0_refiner

model_x2_latent_rescaler = "stabilityai/sd-x2-latent-upscaler"

default_prompt = "Space pineapple, oil paint"
default_fname = "output"

def checkseed(seed):
    mi = -pow(2, 63) 
    ma = pow(2, 63) -1 
    return mi <= seed <= ma

def restate(seed):
    import torch
    import textwrap
    he = '%X' % seed
    he = he.rjust(32, "0")

    re = reversed(textwrap.wrap(he, 2))
    lt = []

    for a in re:
        lt.append(int(a,16))
    
    rt = torch.tensor(lt, dtype=torch.uint8)
    return rt

def state_to_seed_hex(state):
    c = "0x"
    for a in reversed(state):
        b = '%X' % a
        if (len(b) < 2):
            b = f"0{b}"
        c = c + b

    return c 

def state_to_seed(state):
    c = state_to_seed_hex(state)
    return int(c,16)

def report_state(state):
    h = state_to_seed_hex(state)
    #i = state_to_seed(state) # this breaks due to the size of the state on CPU
    #print(f"State: torch.{state} || {h} || {i}")
    print(f"State: torch.{state} || {h}")

def prompt_to_filename(prompt):
    return prompt.replace(" ", "_").replace("/", "_")

def detect_platform():
    import torch
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False, "number":1}
    cpu16 = {"name": "CPU", "device":"cpu", "size":torch.float16, "attention_slicing":False, "number":1}
    graphcore = {"name": "Graphcore", "device":"ipu", "size":torch.float16, "attention_slicing":False, "number":1}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float16, "attention_slicing":False, "number":1}

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

def setup_pipeline(model=model, model_r=model_r, refiner_enabled=True, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}):
    from diffusers.utils import logging
    logging.set_verbosity_error() # Decrease somewhat unhinged log spam!
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
    import torch

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=platform["size"], variant="fp16", add_watermarker=False)
    if freeu["enabled"]:
        pipe.enable_freeu(freeu["s1"], freeu["s2"], freeu["b1"], freeu["b2"])

    if m_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    refiner = None
    if refiner_enabled:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_r, torch_dtype=platform["size"], variant="fp16", text_encoder_2=pipe.text_encoder_2, vae=pipe.vae)
        if m_compile:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)   

    pipe.to(platform["device"])
    if refiner_enabled:
        refiner.to(platform["device"])

    return pipe,refiner

def setup_rescaler_pipeline(model=model_x2_latent_rescaler, m_compile=False):
    from diffusers import StableDiffusionLatentUpscalePipeline
    import torch

    pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(model, torch_dtype=platform["size"])
    if m_compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    pipe.to(platform["device"])

    return pipe

def do_rescale(pipe, prompt, images, num_steps=40, fname=default_fname, save=True, start=0):
    r = []
    count = start
    for a in images:
        ir = pipe(prompt=prompt, image=a, num_inference_steps=num_steps, guidance_scale=0).images[0]
        r.append(ir)
        if save:
            ir.save(f"{fname}_RESIZE_{count}.png")
        count = count + 1

    return r

def inference_denoise(pipe, refiner, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, denoise=0.8, save=True, start=0, seed=None, rescale=False, width=1024, height=1024):
    import torch
    images = []
    images_r = []

    generator = torch.Generator(platform["device"])
    if seed != None:
        if type(seed) is torch.Tensor:
            print(f"Recovering generator state to: {seed}")
            generator.set_state(seed)
        else: 
            print(f"Setting seed to {seed}")
            if checkseed(seed):
                generator.manual_seed(seed)
            else:
                print(f"Seed too long to use .seed - converting to tensor.")
                tseed = restate(seed)
                print(f"Converted tensor: {tseed}")
                generator.set_state(tseed)
    else:
        print("No seed.")
        generator.seed()

    for count in range(start, start+num_gen):
        temp_s = generator.get_state()
        report_state(temp_s)
        image = pipe(prompt=prompt, generator=generator, num_inference_steps=pipe_steps, denoising_end=denoise, output_type="latent", width=width, height=height).images[0]
        if rescale:
            image_r = refiner(prompt=prompt, image=image, generator=generator, num_inference_steps=pipe_steps, denoising_start=denoise, width=width, height=height).images[0]
            #image_r = refiner(prompt=prompt, image=image, generator=generator, num_inference_steps=pipe_steps, denoising_start=denoise, output_type="latent").images[0]
        else: 
            image_r = refiner(prompt=prompt, image=image, generator=generator, num_inference_steps=pipe_steps, denoising_start=denoise, width=width, height=height).images[0]

        if save:
            image_r.save(f"{fname}_{count}.png")
        images.append(image)
        images_r.append(image_r)

    return images, images_r

def inference(pipe, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0, seed=None, rescale=False, width=1024, height=1024):
    import torch
    import time
    images = []

    generator = torch.Generator(platform["device"])
    if seed != None:
        if type(seed) is torch.Tensor:
            print(f"Recovering generator state to: {seed}")
            generator.set_state(seed)
        else: 
            print(f"Setting seed to {seed}")
            if checkseed(seed):
                generator.manual_seed(seed)
            else:
                print(f"Seed too long to use .seed - converting to tensor.")
                tseed = restate(seed)
                print(f"Converted tensor: {tseed}")
                generator.set_state(tseed)
    else:
        print("No seed.")
        generator.seed()

    times = []
    for count in range(start, start+num_gen):
        t_s = time.time()
        temp_s = generator.get_state()
        report_state(temp_s)
        if rescale:
            image = pipe(prompt=prompt, generator=generator, num_inference_steps=pipe_steps, width=width, height=height).images[0]
            #image = pipe(prompt=prompt, generator=generator, num_inference_steps=pipe_steps, output_type="latent").images[0]
        else:
            image = pipe(prompt=prompt, generator=generator, num_inference_steps=pipe_steps, width=width, height=height).images[0]
        images.append(image)
        t_f = time.time()
        times.append(t_f - t_s)
        if save:
            image.save(f"{fname}_{count}.png")
    print(f"Timing Data: {times}")

    return images

def _inference_worker(q, model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0, rescale=False, rescale_steps=40, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}, seed=None, width=1024, height=1024):
    images = serial_inference(model=model, prompt=prompt, denoise=denoise, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save, start=start, rescale=rescale, rescale_steps=rescale_steps, m_compile=m_compile, freeu=freeu, seed=seed, width=width, height=height)
    for a in images:
        q.put(a)

def serial_inference(model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0, rescale=False, rescale_steps=40, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}, seed=None, width=1024, height=1024):
    import warnings
    with warnings.catch_warnings():
        # There are a bunch of diffusers vs PyTorch dep warnings which are annoying.
        warnings.simplefilter("ignore")
        refiner = True
        if denoise == False:
            refiner = False
        pipe, pipe_r = setup_pipeline(model, model_r, refiner, m_compile=m_compile, freeu=freeu)
        if denoise == False:
            images = inference(pipe=pipe, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save, start=start, seed=seed, rescale=rescale, width=width, height=height)
        else:
            _,images = inference_denoise(pipe=pipe, refiner=pipe_r, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, denoise=denoise, save=save, start=start, seed=seed, rescale=rescale, width=width, height=height)
        if rescale:
            pipe_re = setup_rescaler_pipeline(m_compile=m_compile)
            images_r = do_rescale(pipe_re,prompt,images, rescale_steps, fname, save, start)
            images = images_r
        return images

def parallel_inference(model=model, prompt=default_prompt, denoise=False, num_gen=1, pipe_steps=100, fname=default_fname, save=True, rescale=False, rescale_steps=40, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}, seed=None, width=1024, height=1024):
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
        procs.append(Process(target=_inference_worker, args=(q, model, prompt, denoise, chunks[a], pipe_steps, fname, save, starts[a], rescale, rescale_steps, m_compile, freeu, seed, width, height)))
        procs[a].start()

    for a in range(num_gen):
        images.append(q.get())
    return images

def interactive_generate(prompt, num_gen=1, denoise=False, pipe_steps=100, save=True, rescale=False, rescale_steps=45, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}, seed=None, width=1024, height=1024):
    fname = prompt_to_filename(prompt)

    images = parallel_inference(prompt=prompt, denoise=denoise, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save, rescale=rescale, rescale_steps=rescale_steps, m_compile=m_compile, freeu=freeu, seed=seed, width=width, height=height)

    for a in images:
        display(a)