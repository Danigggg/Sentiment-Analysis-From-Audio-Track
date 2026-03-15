import numpy as np
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
import re

app = FastAPI()

model = WhisperModel(
    "small.en",
    device="cuda",
    compute_type="float32",
    num_workers=8
)

samplerate = 16000
chunk_duration = 2
frames_per_chunk = int(samplerate * chunk_duration)

def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return text.upper()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    audio_buffer = np.array([], dtype=np.float32)
    last_text = ""

    try:

        while True:

            data = await ws.receive_bytes()
            audio = np.frombuffer(data, dtype=np.float32)

            audio_buffer = np.concatenate((audio_buffer, audio))

            if len(audio_buffer) >= frames_per_chunk:

                audio_chunk = audio_buffer[:frames_per_chunk]
                audio_buffer = []

                # Skip silence
                if np.abs(audio_chunk).mean() < 0.01:
                    continue

                segments, _ = model.transcribe(
                    audio_chunk,
                    language="en",
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    vad_filter=False
                )

                texts = []

                for seg in segments:
                    if seg.avg_logprob > -1.0:
                        texts.append(seg.text)

                text = " ".join(texts).strip()

                if text and text != last_text:
                    last_text = text
                    await ws.send_text(text)

    except Exception as e:
        print("Client disconnected:", e)