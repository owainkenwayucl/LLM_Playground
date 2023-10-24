from sdxl import setup_pipeline, inference

prompt = "A very happy bulb of garlic, oil paint"

pipeline,_ = setup_pipeline(refiner_enabled = False)

images = inference(pipe=pipeline, prompt=prompt)