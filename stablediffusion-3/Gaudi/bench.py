from sd3 import setup_pipeline, inference

pipe = setup_pipeline()

images = inference(pipeline=pipe, prompt="A very happy bulb of garlic, oil paint", num_gen=10)