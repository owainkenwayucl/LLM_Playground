# Globals (I know!)
import os
import time

n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 8))
model_1_4="CompVis/stable-diffusion-v1-4"
model_1_5="runwayml/stable-diffusion-v1-5"
model_2_0="stabilityai/stable-diffusion-2"
model_2_1="stabilityai/stable-diffusion-2-1"
model_2_0_base="stabilityai/stable-diffusion-2-base"
model_2_1_base="stabilityai/stable-diffusion-2-1-base"

prompt = "space pineapple, oil paint"
model = model_2_1

DEFAULT_NUM_GEN=1
DEFAULT_HEIGHT=768
DEFAULT_WIDTH=768
DEFAULT_ITERATIONS=50
DEFAULT_FNAME="output"
DEFAULT_GUIDANCE_SCALE=5.0

def prompt_to_filename(prompt):
    return prompt.replace(" ", "_").replace("/", "_")

def main(prompt=prompt, model=model, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, number=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS):
    pipeline = setup_pipeline(model)
    _ = inference(pipeline, prompt, number, fname, width, height, guidance_scale, iterations)

def parallel_main(prompt=prompt, model=model, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, number=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS):
    _ = setup_parallel_pipelines()
    _ = parallel_inference(model, prompt, number, fname, width, height, guidance_scale, iterations)

def detect_platform():
    import torch
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32, "attention_slicing":False}
    graphcore = {"name": "Graphcore", "device":"ipu", "size":torch.float16, "attention_slicing":True}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16, "attention_slicing":False}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32, "attention_slicing":False} 
    habana = {"name": "Habana", "device":"hpu", "size":torch.float16, "attention_slicing":False}
    # metal = {"name": "Apple Metal", "device":"mps", "size":torch.float16, "attention_slicing":False} # presently produces black images...

    r = cpu
    try: 
        import poptorch
        print(f"Running on {n_ipu} Graphcore IPU(s)")
        r = graphcore
    except:
        try:
            import habana_frameworks.torch.core as htcore
            print(f"Running on Habana Gaudi 2")
            r = habana
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

def setup_pipeline(model=model, ipus=n_ipu, platform=platform):

    graphcore = False

    if platform["name"] == "Graphcore":
        import torch
        from diffusers import DPMSolverMultistepScheduler

        from optimum.graphcore.diffusers import IPUStableDiffusionPipeline

        if ipus > n_ipu:
            print(f">>> Warning: {ipus} is larger than available IPUs {n_ipu}.")

        executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache") + "/stablediffusion2_text2img"
        pipe = IPUStableDiffusionPipeline.from_pretrained(
            model,
            revision="fp16", 
            torch_dtype=platform["size"],
            requires_safety_checker=False,
            safety_checker=None,
            n_ipu=ipus,
            num_prompts=1,
            num_images_per_prompt=1,
            common_ipu_config_kwargs={"executable_cache_dir": executable_cache_dir}
        )
        if platform["attention_slicing"]:
            pipe.enable_attention_slicing()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    elif platform["name"] == "Habana":
        import torch
        from optimum.habana import GaudiConfig
        from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionPipeline

        gaudi_config = GaudiConfig(
            use_fused_adam = True,
            use_fused_clip_norm = True,
            use_torch_autocast = True
        )

        scheduler = GaudiDDIMScheduler.from_pretrained(model, subfolder="scheduler")
        pipe = GaudiStableDiffusionPipeline.from_pretrained(
            model,
            scheduler=scheduler,
            use_habana=True,
            use_hpu_graphs=True,
            gaudi_config=gaudi_config,
        )


    else:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=platform["size"], safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(platform["device"])
        if platform["attention_slicing"]:
            pipe.enable_attention_slicing()

    return pipe

# This sets things up as a dummy so we pull the checkpoint files etc.
def setup_parallel_pipelines(model = model):    
    pipe = setup_pipeline(model = model)
    return pipe
 
def inference(pipe, prompt=prompt, num_gen=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, image_width=DEFAULT_WIDTH, image_height=DEFAULT_HEIGHT, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS, save=True, start_item=0):
    r = []
    t = []
    for a in range(start_item, start_item + num_gen):
        start = time.time()
        if iterations > 1:
            out = pipe(prompt, height=image_height, width=image_width, num_inference_steps=iterations, guidance_scale=guidance_scale).images[0]
        else: 
            out = pipe(prompt, height=image_height, width=image_width, guidance_scale=guidance_scale).images[0]
        if save:
            out.save(f"{fname}{a}.png")
        r.append(out)
        elapsed = time.time() - start
        t.append(elapsed)
    print(f"Timing data for run: {t}")
    return r

def _inference_worker(q, model, prompt=prompt, num_gen=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, image_width=DEFAULT_WIDTH, image_height=DEFAULT_HEIGHT, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS, save=True, start_item=0):
    pipe = setup_pipeline(model = model)
    i = inference(pipe, prompt, num_gen, fname, image_width, image_height, guidance_scale, iterations, save, start_item)
    for a in i:
        q.put(a)

def parallel_inference(model, prompt=prompt, num_gen=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, image_width=DEFAULT_WIDTH, image_height=DEFAULT_HEIGHT, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS, save=True):
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
        procs.append(Process(target=_inference_worker, args=(q, model, prompt, chunks[a], fname, image_width, image_height, guidance_scale, iterations, save, starts[a])))
        procs[a].start()

    for a in range(num_gen):
        images.append(q.get())
    
    return images
    


def ask(prompt, default):
    response = input(f"{prompt}[{default}]? ")
    response = response.strip()
    if response == "":
        response = default

    return response



if __name__ == "__main__":

    print(f"Known working models: {model_1_4}, {model_1_5}, {model_2_0_base}, {model_2_0}, {model_2_1_base} and {model_2_1}")
    model = ask("Model", model)

    image_width = int(ask("Width", str(DEFAULT_WIDTH)))
    image_height = int(ask("Height", str(DEFAULT_HEIGHT)))
    num_gen = int(ask("Number to generate", str(DEFAULT_NUM_GEN)))
    
    prompt = ask("Prompt", prompt)
    fname = ask("File name", prompt_to_filename(prompt))
    guidance_scale = float(ask("Guidance scale", str(DEFAULT_GUIDANCE_SCALE)))
    iterations = int(ask("Inference iterations", str(DEFAULT_ITERATIONS)))

    if (platform["name"] == "Nvidia"):
        if (platform["number"] > 1):
            parallel_main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations)
        else:
            main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations)
    else:
        main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations)