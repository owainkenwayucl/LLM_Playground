#
# NOTE - AS OF 27/June/2024 THIS HAS NOT BEEN TESTED YET AND IS SPECULATIVE!
#

from optimum.habana.diffusers import GaudiEulerDiscreteScheduler, GaudiStableDiffusionXLPipeline
import torch
import habana_frameworks.torch.hpu.random as htrandom

from optimum.utils import logging
logging.set_verbosity_error()

model_1_0_base="stabilityai/stable-diffusion-xl-base-1.0"
model_1_0_refiner="stabilityai/stable-diffusion-xl-refiner-1.0"

model = model_1_0_base
model_r = model_1_0_refiner

model_x2_latent_rescaler = "stabilityai/sd-x2-latent-upscaler"

default_prompt = "Space pineapple, oil paint"
default_fname = "output"

platform = {"name": "Gaudi", "device":"hpu", "size":torch.float16, "attention_slicing":False}

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

def setup_pipeline(model=model):
    scheduler =  GaudiEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")

    pipeline = GaudiStableDiffusionXLPipeline.from_pretrained(
        model,
        scheduler=scheduler,
        use_habana=True,
        use_hpu_graphs=True,
        gaudi_config="Habana/stable-diffusion"
    )

    return pipeline

def inference(pipe, prompt=default_prompt, num_gen=1, pipe_steps=100, fname=default_fname, save=True, start=0, seed=None, rescale=False, width=1024, height=1024):
    import torch
    images = []

    generator = htrandom.manual_seed(1234)
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
        image = pipe(prompt=prompt, generator=generator, num_inference_steps=pipe_steps, width=width, height=height).images[0]
        images.append(image)
        if save:
            image.save(f"{fname}_{count}.png")

    return images

def interactive_generate(prompt, num_gen=1, denoise=False, pipe_steps=100, save=True, rescale=False, rescale_steps=45, m_compile=False, freeu={"enabled":False, "s1":0.9, "s2":0.2, "b1":1.3, "b2":1.6}, seed=None, width=1024, height=1024):
    fname = prompt_to_filename(prompt)

    pipeline = setup_pipeline(model)
    images = inference(pipeline, prompt=prompt, num_gen=num_gen, pipe_steps=pipe_steps, fname=fname, save=save,seed=seed, width=width, height=height)

    for a in images:
        display(a)