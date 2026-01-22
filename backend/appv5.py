from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions
)
import asyncio
import json
import os
from dotenv import load_dotenv
import base64
from groq import Groq

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print(f"Deepgram API Key: {'Set' if DEEPGRAM_API_KEY else 'NOT SET'}")
print(f"Groq API Key: {'Set' if GROQ_API_KEY else 'NOT SET'}")

groq_client = Groq(api_key=GROQ_API_KEY)


class ConversationManager:
    def __init__(self):
        self.history = []
        self.system_prompt = {
            "role": "system",
            "content": """You are a helpful AI assistant.
Keep responses concise and natural for voice conversation.
Respond in 2-3 sentences unless asked for more detail."""
        }

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_messages(self):
        return [self.system_prompt] + self.history[-10:]


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")

    conversation = ConversationManager()
    dg_connection = None

    # ✅ Control STT
    ai_speaking = False

    # ✅ Event that frontend will trigger when audio playback ends
    tts_done_event = asyncio.Event()

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.live.v("1")
        loop = asyncio.get_event_loop()

        # -------------------- transcript handler --------------------
        def on_message(self, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence:
                    return

                asyncio.run_coroutine_threadsafe(
                    handle_transcript(sentence, result.is_final),
                    loop
                )
            except Exception as e:
                print(f"Error in on_message: {e}")

        async def handle_transcript(sentence, is_final):
            nonlocal ai_speaking

            # ✅ Ignore ALL transcripts while AI speaking
            if ai_speaking:
                return

            if not is_final:
                await websocket.send_json({
                    "type": "transcript",
                    "text": sentence,
                    "is_final": False
                })
                return

            # ✅ Final transcript
            await websocket.send_json({
                "type": "transcript",
                "text": sentence,
                "is_final": True
            })

            conversation.add_message("user", sentence)

            # ✅ AI thinking
            await websocket.send_json({"type": "ai_thinking"})

            # ---------- Groq Streaming ----------
            try:
                messages = conversation.get_messages()

                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True
                )

                full_response = ""
                for chunk in chat_completion:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        await websocket.send_json({
                            "type": "ai_response_chunk",
                            "text": content
                        })

                conversation.add_message("assistant", full_response)

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": f"AI processing failed: {str(e)}"
                })
                return

            # ---------- TTS ----------
            await websocket.send_json({"type": "generating_audio"})

            # ✅ Lock STT until frontend confirms playback finished
            ai_speaking = True
            tts_done_event.clear()

            try:
                speak_options = SpeakOptions(
                    model="aura-asteria-en",
                    encoding="linear16",
                    sample_rate=24000
                )

                # ✅ Generate WAV file
                deepgram.speak.v("1").save(
                    "audio_output.wav",
                    {"text": full_response},
                    speak_options
                )

                # ✅ Send audio in chunks
                with open("audio_output.wav", "rb") as f:
                    audio_data = f.read()

                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(chunk).decode("utf-8")
                    })

                if os.path.exists("audio_output.wav"):
                    os.remove("audio_output.wav")

                # ✅ Signal sending finished (NOT playback finished)
                await websocket.send_json({"type": "audio_complete"})

                # ✅ WAIT until frontend says playback actually ended
                await tts_done_event.wait()

                # ✅ Now unlock STT
                ai_speaking = False
                await websocket.send_json({"type": "listening"})

            except Exception as e:
                ai_speaking = False
                await websocket.send_json({
                    "type": "error",
                    "message": f"TTS failed: {str(e)}"
                })

        def on_error(self, error, **kwargs):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "error",
                    "message": f"STT error: {str(error)}"
                }),
                loop
            )

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            endpointing=300,
            utterance_end_ms=1500,
            vad_events=True,
            smart_format=True,
            punctuate=True
        )

        if dg_connection.start(options):
            await websocket.send_json({"type": "ready"})
        else:
            raise Exception("Failed to start Deepgram")

        # -------------------- receive loop --------------------
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

                if "bytes" in message:
                    if not ai_speaking:
                        dg_connection.send(message["bytes"])

                elif "text" in message:
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        break

                    elif data.get("type") == "clear_history":
                        conversation.history = []
                        await websocket.send_json({"type": "history_cleared"})

                    elif data.get("type") == "tts_done":
                        # ✅ Frontend playback really ended
                        tts_done_event.set()

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            except Exception as e:
                print("Receive loop error:", e)
                break

    finally:
        if dg_connection:
            dg_connection.finish()
        try:
            await websocket.close()
        except:
            pass


@app.get("/")
async def root():
    return {"message": "Voice AI Backend Running", "status": "OK"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
