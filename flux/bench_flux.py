from flux_krea import inference

prompt="A frog sitting on a lily pond in an idylic swamp."
num_gen=9
width=1920
height=1600
num_steps=100
seed=12345

image = inference(prompt=prompt, num_gen=9, num_iters=num_steps, width=width, height=height, seed=seed)