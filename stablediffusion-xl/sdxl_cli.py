DEFAULT_NUM_GEN=1
DEFAULT_ITERATIONS=50
DEFAULT_FNAME="output"

if __name__ == "__main__":
    from sdxl import setup_pipeline, inference
    prompt = "A very happy bulb of garlic, oil paint"

    num_gen = int(ask("Number to generate", str(DEFAULT_NUM_GEN)))
    
    prompt = ask("Prompt", prompt)
    fname = ask("File name", DEFAULT_FNAME)

    iterations = int(ask("Inference iterations", str(DEFAULT_ITERATIONS)))

    pipeline,_ = setup_pipeline(refiner_enabled = False)

    images = inference(pipe=pipeline, prompt=prompt, fname=fname, num_gen=num_gen, pipe_steps=iterations)