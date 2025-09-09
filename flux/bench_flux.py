from flux_krea import inference

prompt="A frog sitting on a lily pond in an idylic swamp."

width=1920
height=1600
num_steps=100
seed=12345

image = inference(prompt=prompt, num_iters=num_steps, width=width, height=height, seed=seed)