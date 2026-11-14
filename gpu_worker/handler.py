import base64, io, os
import torch
import runpod
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = os.getenv("ALEX_IMAGE_MODEL_ID", "Lykon/dreamshaper-8")
pipe = None

def load_model():
    global pipe
    if pipe is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Worker sem GPU CUDA.")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe

def handler(job):
    data = job.get("input", {})
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Prompt vazio.")
    width, height = int(data.get("width", 768)), int(data.get("height", 768))
    steps, guidance = int(data.get("steps", 25)), float(data.get("guidance", 7.5))
    if width < 256 or height < 256 or width % 8 or height % 8:
        raise ValueError("Resolução inválida.")
    generator = torch.Generator(device="cuda").manual_seed(int.from_bytes(os.urandom(4), "big"))
    image = load_model()(prompt=prompt, width=width, height=height,
        num_inference_steps=steps, guidance_scale=guidance, generator=generator).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": width, "height": height}

runpod.serverless.start({"handler": handler})
