import io
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from kokoro import KPipeline

app = FastAPI(title="Free TTS API")

# Allow requests from any origin so freettss.com (or any frontend) can call this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LANGUAGES = {
    "American English": "a",
    "British English": "b",
    "Spanish": "e",
    "French": "f",
    "Hindi": "h",
    "Italian": "i",
    "Brazilian Portuguese": "p",
    "Japanese": "j",
    "Mandarin Chinese": "z",
}

VOICES_BY_LANG = {
    "a": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael"],
    "b": ["bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
    "e": ["ef_dora", "em_alex"],
    "f": ["ff_siwis"],
    "h": ["hf_alpha", "hm_omega"],
    "i": ["if_sara", "im_nicola"],
    "p": ["pf_dora", "pm_alex"],
    "j": ["jf_alpha", "jm_kumo"],
    "z": ["zf_xiaobei", "zm_yunjian"],
}

# Pipelines are created lazily per language and cached in memory.
_pipelines = {}


def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "American English"
    voice: str = "af_heart"
    speed: float = 1.0


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {
        "message": "TTS API is running",
        "status": "ok"
    }

@app.get("/voices")
def voices():
    return {"languages": LANGUAGES, "voices_by_lang": VOICES_BY_LANG}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    lang_code = LANGUAGES.get(req.language, "a")
    voice = req.voice or VOICES_BY_LANG[lang_code][0]
    speed = max(0.5, min(2.0, req.speed or 1.0))

    try:
        pipeline = get_pipeline(lang_code)
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        chunks = [audio for _, _, audio in generator]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    if not chunks:
        raise HTTPException(status_code=500, detail="No audio was generated for this text.")

    full_audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, full_audio, 24000, format="WAV")
    buf.seek(0)

    duration = len(full_audio) / 24000
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "inline; filename=speech.wav",
            "X-Audio-Duration": f"{duration:.2f}",
        },
    )
