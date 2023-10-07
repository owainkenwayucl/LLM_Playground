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

def main(prompt=prompt, model=model, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, number=DEFAULT_NUM_GEN, fname=DEFAULT_FNAME, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS):
    pipeline = setup_pipeline(model)
    _ = inference(pipeline, prompt, number, fname, width, height, guidance_scale, iterations)

def detect_platform():
    import torch
    cpu = {"name": "CPU", "device":"cpu", "size":torch.float32}
    graphcore = {"name": "Graphcore", "device":"ipu", "size":torch.float16}
    nvidia = {"name": "Nvidia", "device":"cuda", "size":torch.float16}
    metal = {"name": "Apple Metal", "device":"mps", "size":torch.float32}

    r = cpu
    try: 
        import poptorch
        print(f"Running on {n_ipu} Graphcore IPU(s)")
        r = graphcore
    except:
        if torch.cuda.device_count() > 0:
            print("Running on Nvidia GPU")
            r = nvidia
        elif torch.backends.mps.is_available():
            print("Running on Apple GPU")
            r = metal      
    return r

def setup_pipeline(model, ipus=n_ipu):

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
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    else:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=platform["size"], safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(platform["device"])

    return pipe

def inference(pipe, prompt, num_gen=DEFAULT_NUM_GEN, fname=DEFAULT_NUM_GEN, image_width=DEFAULT_WIDTH, image_height=DEFAULT_HEIGHT, guidance_scale=DEFAULT_GUIDANCE_SCALE, iterations=DEFAULT_ITERATIONS, save=True):
    r = []
    t = []
    for a in range(num_gen):
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

def ask(prompt, default):
    response = input(f"{prompt}[{default}]? ")
    response = response.strip()
    if response == "":
        response = default

    return response

platform = detect_platform()

if __name__ == "__main__":

    print(f"Known working models: {model_1_4}, {model_1_5}, {model_2_0_base}, {model_2_0}, {model_2_1_base} and {model_2_1}")
    model = ask("Model", model)

    image_width = int(ask("Width", str(DEFAULT_WIDTH)))
    image_height = int(ask("Height", str(DEFAULT_HEIGHT)))
    num_gen = int(ask("Number to generate", str(DEFAULT_NUM_GEN)))
    
    prompt = ask("Prompt", prompt)
    fname = ask("File name", DEFAULT_FNAME)
    guidance_scale = float(ask("Guidance scale", str(DEFAULT_GUIDANCE_SCALE)))
    iterations = int(ask("Inference iterations", str(DEFAULT_ITERATIONS)))

    main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations)