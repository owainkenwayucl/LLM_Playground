def main(prompt, model, width, height, number, fname):
    import os
    from PIL import Image

    graphcore = False

    image_width = width
    image_height = height
    num_gen = number

    try:
        import poptorch
        graphcore = True
        print("Graphcore detected!")
    except:
        graphcore = False
        print("Graphcore not detected!")

    if graphcore:
        import torch
        from diffusers import DPMSolverMultistepScheduler

        from optimum.graphcore.diffusers import IPUStableDiffusionPipeline

        n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 8))
        executable_cache_dir = os.getenv("POPLAR_EXECUTABLE_CACHE_DIR", "./exe_cache") + "/stablediffusion2_text2img"
        pipe = IPUStableDiffusionPipeline.from_pretrained(
            model,
            revision="fp16", 
            torch_dtype=torch.float16,
            requires_safety_checker=False,
            n_ipu=n_ipu,
            num_prompts=1,
            num_images_per_prompt=1,
            common_ipu_config_kwargs={"executable_cache_dir": executable_cache_dir}
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Doing precompile.")
        out = pipe("pineapple", height=image_height, width=image_width, guidance_scale=7.5).images[0]
        out.save(f"compile.png")
    else:
        import torch
        from diffusers import StableDiffusionPipeline

        if torch.cuda.device_count() > 0:
            print("Running on GPU")
            pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
        else:
            print("Running on CPU")
            pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float32)

    for a in range(num_gen):
        out = pipe(prompt, height=image_height, width=image_width, guidance_scale=7.5).images[0]
        out.save(f"{fname}{a}.png")

def ask(prompt, default):
    response = input(f"{prompt}[{default}]? ")
    response = response.strip()
    if response == "":
        response = default

    return response

if __name__ == "__main__":

    model_1_4="CompVis/stable-diffusion-v1-4"
    model_1_5="runwayml/stable-diffusion-v1-5"
    model_2_0="stabilityai/stable-diffusion-2"

    model = ask("Model", model_1_4)

    image_width = int(ask("Width", str(512)))
    image_height = int(ask("Height", str(512)))
    num_gen = int(ask("Number to generate", str(20)))

    prompt = "space pineapple, oil paint"
    prompt = ask("Prompt", prompt)
    fname = ask("File name", "output")

    main(prompt, model, image_width, image_height, num_gen, fname)