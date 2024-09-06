model_xt = "stabilityai/stable-video-diffusion-img2vid-xt"

model = model_xt

def run(command):
    import subprocess
    return subprocess.run(command, capture_output=True, encoding='UTF-8')
    
def encode(frames, filename, framerate=7, webm=False):
    for a in range(len(frames)):
        frames[a].save(f".output{a}.png")

    library = "libopenh264"
    if webm:
        library = "vp8"
    x = run(["ffmpeg", "-framerate", str(framerate), "-i", ".output%d.png", "-c:v", library, "-pix_fmt","yuv420p",filename])
    print(x.stdout)
    print(x.stderr)

def setup_pipe(model = model, device="cuda"):
    import torch

    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model, torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device)
    pipe.enable_model_cpu_offload() 

    return pipe

def inference(pipe, image):
    return pipe(image, decode_chunk_size=8).frames[0]

