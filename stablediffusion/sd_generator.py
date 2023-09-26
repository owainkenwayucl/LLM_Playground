# Globals (I know!)
import os
n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 8))

def main(prompt, model, width, height, number, fname, guidance_scale, iterations):
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
            n_ipu=ipus,
            num_prompts=1,
            num_images_per_prompt=1,
            common_ipu_config_kwargs={"executable_cache_dir": executable_cache_dir}
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    else:
        import torch
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=platform["size"])
        pipe = pipe.to(platform["device"])

    return pipe

def inference(pipe, prompt, num_gen=1, fname="output", image_width=512, image_height=512, guidance_scale=7.5, iterations=1):
    r = []
    for a in range(num_gen):
        if iterations > 1:
            out = pipe(prompt, height=image_height, width=image_width, num_inference_steps=iterations, guidance_scale=guidance_scale).images[0]
        else: 
            out = pipe(prompt, height=image_height, width=image_width, guidance_scale=guidance_scale).images[0]
        out.save(f"{fname}{a}.png")
        r.append(out)
    return r

def ask(prompt, default):
    response = input(f"{prompt}[{default}]? ")
    response = response.strip()
    if response == "":
        response = default

    return response

platform = detect_platform()

if __name__ == "__main__":

    model_1_4="CompVis/stable-diffusion-v1-4"
    model_1_5="runwayml/stable-diffusion-v1-5"
    model_2_0="stabilityai/stable-diffusion-2"
    model_2_1="stabilityai/stable-diffusion-2-1"

    print(f"Known working models: {model_1_4}, {model_1_5}, {model_2_0} and {model_2_1}")
    model = ask("Model", model_2_1)

    image_width = int(ask("Width", str(768)))
    image_height = int(ask("Height", str(768)))
    num_gen = int(ask("Number to generate", str(1)))

    prompt = "space pineapple, oil paint"
    prompt = ask("Prompt", prompt)
    fname = ask("File name", "output")
    guidance_scale = float(ask("Guidance scale", str(9.0)))
    iterations = int(ask("Inference iterations", str(50)))

    main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations)