import io
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

MOCK       = os.environ.get("MOCK", "false").lower() == "true"
HF_REPO    = os.environ.get("MODEL_ID", "chethan1988/f5tts-ljspeech")
CKPT_FILE  = os.environ.get("CKPT_FILE", "model_final.pt")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "/app/vocab.txt")

DEFAULT_REF_REPO = os.environ.get("DEFAULT_REF_REPO")
DEFAULT_REF_FILE = os.environ.get("DEFAULT_REF_FILE")
DEFAULT_REF_TEXT = os.environ.get("DEFAULT_REF_TEXT", "")

TARGET_SR  = 24_000
N_MEL      = 100
HOP_LENGTH = 256
N_FFT      = 1024

model                                    = None
vocoder                                  = None
default_ref_audio_path: Optional[str]    = None
lock                                     = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocoder, default_ref_audio_path

    if MOCK:
        yield
        return

    from huggingface_hub import hf_hub_download
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer
    from f5_tts.infer.utils_infer import load_vocoder

    cache_dir = os.environ.get("HF_HOME", "/workspace/models")
    os.makedirs(cache_dir, exist_ok=True)

    vocab_char_map, vocab_size = get_tokenizer(VOCAB_PATH, "custom")

    m = CFM(
        transformer=DiT(
            dim             = 1024,
            depth           = 22,
            heads           = 16,
            ff_mult         = 2,
            text_dim        = 512,
            conv_layers     = 4,
            text_num_embeds = vocab_size,
            mel_dim         = N_MEL,
        ),
        mel_spec_kwargs=dict(
            n_fft              = N_FFT,
            hop_length         = HOP_LENGTH,
            n_mel_channels     = N_MEL,
            target_sample_rate = TARGET_SR,
        ),
        vocab_char_map=vocab_char_map,
    )

    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=CKPT_FILE, cache_dir=cache_dir)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[serve] loaded {CKPT_FILE} from {HF_REPO} | missing={len(missing)} unexpected={len(unexpected)}")

    m = m.to("cuda").eval()
    model = m

    vocoder = load_vocoder(vocoder_name="vocos", is_local=False)

    if DEFAULT_REF_REPO and DEFAULT_REF_FILE:
        try:
            default_ref_audio_path = hf_hub_download(
                repo_id  = DEFAULT_REF_REPO,
                filename = DEFAULT_REF_FILE,
                cache_dir = cache_dir,
            )
            print(f"[serve] default ref audio: {default_ref_audio_path}")
        except Exception as e:
            print(f"[serve] no default ref audio ({e}); callers must supply ref_audio")

    yield
    
app = FastAPI(title="F5-TTS Serve", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mock": MOCK, "has_default_ref": default_ref_audio_path is not None}

@app.post("/generate")
async def generate(
    target_text:  str                     = Form(...),
    ref_text:     str                     = Form(default=""),
    ref_audio:    Optional[UploadFile]    = File(default=None),
    num_steps:    int                     = Form(default=32,  ge=4,   le=64),
    cfg_strength: float                   = Form(default=2.0, ge=0.0, le=5.0),
    speed:        float                   = Form(default=1.0, ge=0.5, le=2.0),
):
    if MOCK:
        buf = io.BytesIO()
        torchaudio.save(buf, torch.zeros(1, TARGET_SR), TARGET_SR, format="wav")
        return Response(content=buf.getvalue(), media_type="audio/wav")

    from f5_tts.infer.utils_infer import preprocess_ref_audio_text, infer_process

    tmp_path = None
    if ref_audio is not None:
        if not ref_text:
            raise HTTPException(400, "ref_text is required when ref_audio is provided")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(await ref_audio.read())
        tmp.close()
        ref_audio_path = tmp.name
        tmp_path = tmp.name
    else:
        if default_ref_audio_path is None:
            raise HTTPException(400, "no default ref audio configured; upload ref_audio + ref_text")
        ref_audio_path = default_ref_audio_path
        if not ref_text:
            ref_text = DEFAULT_REF_TEXT

    try:
        with lock:
            ref_audio_pre, ref_text_pre = preprocess_ref_audio_text(ref_audio_path, ref_text)
            final_wave, final_sr, _ = infer_process(
                ref_audio_pre,
                ref_text_pre,
                target_text,
                model,
                vocoder,
                mel_spec_type       = "vocos",
                target_rms          = 0.1,
                cross_fade_duration = 0.15,
                nfe_step            = num_steps,
                cfg_strength        = cfg_strength,
                sway_sampling_coef  = -1.0,
                speed               = speed,
                fix_duration        = None,
            )
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    wave = torch.as_tensor(final_wave, dtype=torch.float32)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)

    buf = io.BytesIO()
    torchaudio.save(buf, wave, final_sr, format="wav")
    return Response(content=buf.getvalue(), media_type="audio/wav")