"""Cliente remoto de geração de imagens da Alex IA Ultra via RunPod Serverless."""
import base64, io, os, time, uuid
from pathlib import Path
import requests
from PIL import Image

MOTOR_IMAGEM = "Alex Image Engine — RunPod Serverless GPU"

def obter_pasta_imagens():
    pasta = Path(os.getenv("ALEX_IMAGE_OUTPUT_DIR", "/tmp/alex_ia_ultra_imagens"))
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta

def _config():
    endpoint = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
    key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not endpoint or not key:
        raise RuntimeError("Configure RUNPOD_ENDPOINT_ID e RUNPOD_API_KEY nas variáveis do Render.")
    return endpoint, key

def gerar_imagem_local(prompt):
    if not prompt or not prompt.strip():
        raise ValueError("O prompt da imagem está vazio.")
    endpoint, key = _config()
    largura = int(os.getenv("ALEX_IMAGE_WIDTH", "768"))
    altura = int(os.getenv("ALEX_IMAGE_HEIGHT", "768"))
    passos = int(os.getenv("ALEX_IMAGE_STEPS", "25"))
    guidance = float(os.getenv("ALEX_IMAGE_GUIDANCE", "7.5"))
    if largura < 256 or altura < 256 or largura % 8 or altura % 8:
        raise ValueError("Largura e altura devem ser >=256 e múltiplas de 8.")
    base = f"https://api.runpod.ai/v2/{endpoint}"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = requests.post(base + "/run", headers=headers, json={"input": {
        "prompt": prompt.strip(), "width": largura, "height": altura,
        "steps": passos, "guidance": guidance}}, timeout=45)
    resp.raise_for_status()
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError("RunPod não retornou ID do trabalho.")
    limite = time.monotonic() + int(os.getenv("ALEX_IMAGE_TIMEOUT_SECONDS", "600"))
    while time.monotonic() < limite:
        r = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status == "COMPLETED":
            output = data.get("output") or {}
            if isinstance(output, list) and output:
                output = output[0]
            encoded = output.get("image_base64") if isinstance(output, dict) else None
            if not encoded:
                raise RuntimeError("Worker concluiu sem retornar image_base64.")
            raw = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            if image.size != (largura, altura):
                image = image.resize((largura, altura))
            path = obter_pasta_imagens() / f"alex_{uuid.uuid4().hex}.png"
            image.save(path, format="PNG")
            with Image.open(path) as check:
                check.verify()
            return str(path)
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"RunPod {status}: {data.get('error', 'sem detalhes')}")
        time.sleep(2)
    raise TimeoutError(f"GPU excedeu o limite de espera. Job: {job_id}")

def gerar_imagem_zimage(prompt):
    return gerar_imagem_local(prompt)

def guardar_ultima_imagem(imagem, prompt, caminho=None, motor=None):
    try:
        import streamlit as st
        st.session_state.ultima_imagem = imagem
        st.session_state.ultima_imagem_caminho = caminho
        st.session_state.ultimo_prompt_imagem = prompt
        st.session_state.ultimo_motor_imagem = motor or MOTOR_IMAGEM
        return True
    except Exception:
        return False

def gerar_imagem(prompt):
    try:
        path = gerar_imagem_local(prompt)
        guardar_ultima_imagem(path, prompt, path, MOTOR_IMAGEM)
        return path, "Imagem gerada e validada na GPU remota."
    except Exception as exc:
        return None, f"Falha na GPU remota: {exc}"

def obter_ultima_imagem():
    try:
        import streamlit as st
        return st.session_state.get("ultima_imagem")
    except Exception:
        return None

def obter_caminho_ultima_imagem():
    try:
        import streamlit as st
        return st.session_state.get("ultima_imagem_caminho")
    except Exception:
        return None

def obter_prompt_ultima_imagem():
    try:
        import streamlit as st
        return st.session_state.get("ultimo_prompt_imagem", "")
    except Exception:
        return ""

def obter_motor_ultima_imagem():
    try:
        import streamlit as st
        return st.session_state.get("ultimo_motor_imagem", "")
    except Exception:
        return ""

def limpar_ultima_imagem():
    try:
        import streamlit as st
        for key in ("ultima_imagem", "ultima_imagem_caminho", "ultimo_prompt_imagem", "ultimo_motor_imagem"):
            st.session_state.pop(key, None)
    except Exception:
        pass

def mostrar_imagem(prompt):
    try:
        import streamlit as st
        with st.spinner("Alex está gerando na GPU remota..."):
            path, message = gerar_imagem(prompt)
        if not path:
            st.error(message)
            return False
        st.image(path, caption="Imagem gerada pela GPU remota", use_container_width=True)
        st.success(message)
        return True
    except Exception:
        return False

def executar_teste():
    import streamlit as st
    st.title("Alex Image Engine — GPU remota")
    prompt = st.text_input("Descreva a imagem")
    if st.button("Gerar imagem"):
        mostrar_imagem(prompt)
