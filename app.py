import io
import os
import urllib.request
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro_onnx import Kokoro

app = FastAPI(title="Free TTS API")

# Allow requests from any origin so freettss.com (or any frontend) can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "kokoro-v1.0.int8.onnx")
VOICES_PATH = os.path.join(MODEL_DIR, "voices-v1.0.bin")

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading Kokoro ONNX model (int8, quantized)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(VOICES_PATH):
        print("Downloading Kokoro voices file...")
        urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)


ensure_model_files()
kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

LANGUAGES = {
    "American English": "en-us",
    "British English": "en-gb",
    "Spanish": "es",
    "French": "fr-fr",
    "Hindi": "hi",
    "Italian": "it",
    "Brazilian Portuguese": "pt-br",
    "Japanese": "ja",
    "Mandarin Chinese": "cmn",
}

VOICES_BY_LANG = {
    "en-us": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael"],
    "en-gb": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
    "es": ["ef_dora", "em_alex"],
    "fr-fr": ["ff_siwis"],
    "hi": ["hf_alpha", "hm_omega"],
    "it": ["if_sara", "im_nicola"],
    "pt-br": ["pf_dora", "pm_alex"],
    "ja": ["jf_alpha", "jm_kumo"],
    "cmn": ["zf_xiaobei", "zm_yunjian"],
}


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "American English"
    voice: str = "af_heart"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def voices():
    return {"languages": LANGUAGES, "voices_by_lang": VOICES_BY_LANG}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    lang_code = LANGUAGES.get(req.language, "en-us")
    voice = req.voice or VOICES_BY_LANG[lang_code][0]
    speed = max(0.5, min(2.0, req.speed or 1.0))

    try:
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    if samples is None or len(samples) == 0:
        raise HTTPException(status_code=500, detail="No audio was generated for this text.")

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    buf.seek(0)

    duration = len(samples) / sample_rate
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=speech.wav",
            "X-Audio-Duration": f"{duration:.2f}",
        },
    )

