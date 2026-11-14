"""Ponte Render <-> Kaggle para geração de imagens.

Configure REDIS_URL e KAGGLE_BRIDGE_SECRET no Render.
No Kaggle, configure ALEX_RENDER_URL e KAGGLE_BRIDGE_SECRET.
A GPU do Kaggle faz polling de tarefas; o Render nunca tenta acessar a GPU diretamente.
"""
import base64
import json
import os
import time
import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()
QUEUE = "alex:kaggle:images:queue"
RESULT_PREFIX = "alex:kaggle:images:result:"
SECRET = os.getenv("KAGGLE_BRIDGE_SECRET", "")
REDIS_URL = os.getenv("REDIS_URL", "")

class JobRequest(BaseModel):
    prompt: str

class JobResult(BaseModel):
    job_id: str
    success: bool
    image_base64: str | None = None
    error: str | None = None

def client():
    if not REDIS_URL:
        raise HTTPException(503, "REDIS_URL não configurada no Render.")
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)

def auth(value):
    if not SECRET:
        raise HTTPException(503, "KAGGLE_BRIDGE_SECRET não configurado.")
    if value != SECRET:
        raise HTTPException(401, "Credencial da ponte inválida.")

def generate_on_kaggle(prompt: str, timeout: int = 900) -> bytes:
    r = client()
    job_id = uuid.uuid4().hex
    key = RESULT_PREFIX + job_id
    r.rpush(QUEUE, json.dumps({"job_id": job_id, "prompt": prompt, "created_at": time.time()}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = r.get(key)
        if raw:
            r.delete(key)
            result = json.loads(raw)
            if not result.get("success"):
                raise RuntimeError(result.get("error") or "Falha no worker Kaggle.")
            data = base64.b64decode(result["image_base64"], validate=True)
            if not data:
                raise RuntimeError("Kaggle devolveu imagem vazia.")
            return data
        time.sleep(2)
    raise TimeoutError("Tempo esgotado aguardando a GPU Kaggle.")

@router.get("/api/kaggle/health")
def kaggle_health():
    return {"success": True, "service": "alex-kaggle-bridge", "redis": bool(REDIS_URL), "secret_configured": bool(SECRET)}

@router.get("/api/kaggle/jobs/next")
def next_job(x_api_secret: str | None = Header(default=None)):
    auth(x_api_secret)
    item = client().blpop(QUEUE, timeout=25)
    if not item:
        return {"job": None}
    return {"job": json.loads(item[1])}

@router.post("/api/kaggle/jobs/complete")
def complete_job(payload: JobResult, x_api_secret: str | None = Header(default=None)):
    auth(x_api_secret)
    if payload.success and not payload.image_base64:
        raise HTTPException(400, "Imagem Base64 ausente.")
    client().set(RESULT_PREFIX + payload.job_id, payload.model_dump_json(), ex=1800)
    return {"success": True, "job_id": payload.job_id}
