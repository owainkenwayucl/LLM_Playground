DEFAULT_NUM_GEN=1
DEFAULT_ITERATIONS=50
DEFAULT_FNAME="output"

def ask(prompt, default):
    response = input(f"{prompt}[{default}]? ")
    response = response.strip()
    if response == "":
        response = default

    return response

if __name__ == "__main__":
    from sdxl import setup_pipeline, inference, prompt_to_filename
    prompt = "A very happy bulb of garlic, oil paint"

    num_gen = int(ask("Number to generate", str(DEFAULT_NUM_GEN)))
    
    prompt = ask("Prompt", prompt)
    fname = ask("File name", prompt_to_filename(prompt))

    iterations = int(ask("Inference iterations", str(DEFAULT_ITERATIONS)))

    pipeline = setup_pipeline()

    images = inference(pipe=pipeline, prompt=prompt, fname=fname, num_gen=num_gen, pipe_steps=iterations)