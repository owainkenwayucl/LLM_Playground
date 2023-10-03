# Globals (I know!)
import os
import time
from sd_generator import ask, setup_pipeline, detect_platform, inference

n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 8))

def main(prompt, model, width, height, number, fname, guidance_scale, iterations, dorescale=False):
    pipeline = setup_pipeline(model)
    images = inference(pipeline, prompt, number, fname, width, height, guidance_scale, iterations)
    if dorescale:
        _ = rescale(images, prompt, fname)


def setup_rescale_pipeline(model, ipus=n_ipu):

    graphcore = False

    if platform["name"] == "Graphcore":
        d = "cpu"

    else:
        d = platform["device"]
 
    import torch
    from diffusers import StableDiffusionUpscalePipeline

    pipe = StableDiffusionUpscalePipeline.from_pretrained(model, torch_dtype=platform["size"])
    pipe = pipe.to(d)

    return pipe

def rescale(images, prompt, fname="output"):
    print(f"Rescaling images")
    model_rescale_2_1="stabilityai/stable-diffusion-x4-upscaler"

    model = model_rescale_2_1
    r_pipe = setup_rescale_pipeline(model)
    r = []
    a = 0
    for image in images:
        out = r_pipe(prompt=prompt, image=image).images[0]
        out.save(f"{fname}{a}-scaled.png")
        r.append(out)
        a += 1

    return r

platform = detect_platform()

if __name__ == "__main__":

    model_1_4="CompVis/stable-diffusion-v1-4"
    model_1_5="runwayml/stable-diffusion-v1-5"
    model_2_0="stabilityai/stable-diffusion-2"
    model_2_1="stabilityai/stable-diffusion-2-1"

    print(f"Known working models: {model_1_4}, {model_1_5}, {model_2_0} and {model_2_1}")
    model = ask("Model", model_1_5)

#    image_width = int(ask("Width", str(768)))
#    image_height = int(ask("Height", str(768)))
    image_width = 256
    image_height = 256
    num_gen = int(ask("Number to generate", str(1)))

    prompt = "space pineapple, oil paint"
    prompt = ask("Prompt", prompt)
    fname = ask("File name", "output")
    guidance_scale = float(ask("Guidance scale", str(7.5)))
    iterations = int(ask("Inference iterations", str(1)))
    dorescale = ask("Use x4 upscaler", "yes").lower() == "yes"

    main(prompt, model, image_width, image_height, num_gen, fname, guidance_scale, iterations, dorescale)