model_1_0_base="stabilityai/stable-diffusion-xl-base-1.0"
model_1_0_refiner="stabilityai/stable-diffusion-xl-refiner-1.0"

model = model_1_0_base
model_r = model_1_0_refiner

default_prompt = "Space pineapple, oil paint"
default_fname = "output"

def setup_pipeline(model=model):
    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16, use_safe_tensors=True, variant="fp16", add_watermarker=False)
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_1_0_refiner, torch_dtype=torch.float16, use_safe_tensors=True, variant="fp16", text_encoder_2=pipe.text_encoder_2, vae=pipe.vae)

    pipe.to("cuda")
    refiner.to("cuda")

    return pipe,refiner

def inference(pipeline, refiner, prompt=default_prompt, pipe_steps=100, refiner_steps=100, fname=default_fname, save=True):
    image = pipe(prompt=prompt, num_inference_steps=pipe_steps).images[0]
    image_r = refiner(prompt=prompt, image=image, num_inference_steps=refiner_steps).images[0]
    if save:
        image.save(f"{fname}.png")
        image_r.save(f"{fname}_r.png")

    return image, image_r