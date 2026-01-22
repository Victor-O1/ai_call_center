import numpy as np
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper...")
model = whisper.load_model("base")
print("Whisper ready")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            msg = await ws.receive()

            # 🔥 THIS IS THE FIX
            if msg.get("bytes") is None:
                continue  # ignore text frames

            audio_chunk = np.frombuffer(msg["bytes"], dtype=np.float32)
            audio_buffer = np.concatenate((audio_buffer, audio_chunk))

            if len(audio_buffer) >= 32000:
                chunk = audio_buffer[:32000]
                audio_buffer = audio_buffer[32000:]

                result = model.transcribe(
                    chunk,
                    fp16=False,
                    language="en"
                )

                text = result["text"].strip()
                if text:
                    await ws.send_text(text)

    except WebSocketDisconnect:
        print("Client disconnected")
