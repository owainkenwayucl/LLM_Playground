from flux_krea import inference

prompt="A frog sitting on a lily pond in an idylic swamp."
num_gen=9
width=1024
height=1024
num_steps=50
seed=12345

image = inference(prompt=prompt, num_gen=9, num_iters=num_steps, width=width, height=height, seed=seed)
