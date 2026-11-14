"""Worker Kaggle: execute esta célula/script no notebook com Internet e GPU ativadas.

Variáveis:
  ALEX_RENDER_URL=https://SEU-SERVICO.onrender.com
  KAGGLE_BRIDGE_SECRET=mesmo segredo configurado no Render
  ALEX_IMAGE_MODEL_PATH=/kaggle/working/models/dreamshaper-8
"""
import os, io, time, base64, requests, torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DEISMultistepScheduler

RENDER = os.environ["ALEX_RENDER_URL"].rstrip("/")
SECRET = os.environ["KAGGLE_BRIDGE_SECRET"]
MODEL = os.environ.get("ALEX_IMAGE_MODEL_PATH", "/kaggle/working/models/dreamshaper-8")
HEADERS = {"x-api-secret": SECRET}
assert torch.cuda.is_available(), "Ative GPU no Kaggle antes de executar."
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, local_files_only=True
).to("cuda:0")
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()
pipe.set_progress_bar_config(disable=True)
print("Alex Kaggle Worker pronto:", torch.cuda.get_device_name(0))

while True:
    try:
        response = requests.get(RENDER + "/api/kaggle/jobs/next", headers=HEADERS, timeout=40)
        response.raise_for_status()
        job = response.json().get("job")
        if not job:
            continue
        job_id, prompt = job["job_id"], job["prompt"]
        try:
            with torch.inference_mode():
                image = pipe(prompt, width=768, height=768, num_inference_steps=25).images[0]
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            body = {"job_id": job_id, "success": True,
                    "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
        except Exception as exc:
            body = {"job_id": job_id, "success": False, "error": str(exc)}
        done = requests.post(RENDER + "/api/kaggle/jobs/complete",
                             headers=HEADERS, json=body, timeout=120)
        done.raise_for_status()
        print("Concluído:", job_id, body["success"])
    except KeyboardInterrupt:
        print("Worker encerrado.")
        break
    except Exception as exc:
        print("Erro de conexão/worker:", repr(exc))
        time.sleep(5)
