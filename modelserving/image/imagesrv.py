import base64
import io
import time
from typing import Literal, Optional
 
import torch
from diffusers import AutoPipelineForText2Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
import configparser

config = configparser.ConfigParser()
config.read("imagesrv.ini")
 
MODEL_ID = config['huggingface']['model'].strip()
STEPS = int(config['image']['steps'].strip())
SIZE = config['image']['size'].strip()
GS = float(config['image']['guidance_scale'].strip())
PORT = int(config['server']['port'])
DEVICE = "cuda" 
DTYPE = torch.float16
USE_BASE64 = True
 
app = FastAPI(title="Image Generation API")
 
print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE}) …")
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    variant="fp16" if DTYPE == torch.float16 else None,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)
print("Pipeline ready.")
 
class ImageGenerationRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = SIZE                         
    response_format: Literal["b64_json"] = "b64_json"
    model: Optional[str] = None 
    quality: Optional[str] = None
    style: Optional[str] = None
 
 
def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()
 
 
@app.post("/v1/images/generations")
async def generate_images(req: ImageGenerationRequest):
    try:
        width, height = (int(x) for x in req.size.lower().split("x"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid size format: {req.size!r}. Expected 'WxH'.")
 
    pipeline_kwargs = dict(
        prompt=[req.prompt] * req.n,
        width=width,
        height=height,
        num_inference_steps=STEPS, 
        guidance_scale=GS, 
    )
 
    try:
        result = pipe(**pipeline_kwargs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
 
    images = result.images 
 
    data = []
    for img in images:
        if USE_BASE64 or req.response_format == "b64_json":
            data.append({"b64_json": pil_to_b64(img), "revised_prompt": req.prompt})
        else:
            raise HTTPException(status_code=500, detail=str("Only Base64 mode accepted")) 
    return JSONResponse({
        "created": int(time.time()),
        "data": data,
    })
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
