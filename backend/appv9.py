from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import asyncio
import json
import os
import time
import base64
import re
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
import httpx

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

AGENTS = {
    "primary": {
        "name": "Ava (Primary Agent)",
        "voice": "aura-asteria-en",
        "intro": "Hey! I’m Ava, your primary assistant. Tell me what you need help with.",
    },
    "supervisor": {
        "name": "Noah (Supervisor Agent)",
        "voice": "aura-luna-en",
        "intro": "Hi, I’m Noah, the supervisor agent. I’ll help you resolve this smoothly.",
    },
    "escalation": {
        "name": "Maya (Escalation Agent)",
        "voice": "aura-hera-en",
        "intro": "Hello, I’m Maya from escalations. I’ll handle this with priority.",
    },
}

EmotionLabel = Literal["calm", "neutral", "happy", "confused", "frustrated", "angry"]
AgentKey = Literal["primary", "supervisor", "escalation"]


class HeaderPacket(BaseModel):
    agent_key: AgentKey
    intent: str
    emotion: EmotionLabel
    confidence: int = Field(..., ge=0, le=100)
    csat_prediction: int = Field(..., ge=1, le=5)
    summary: str
    resolved: bool


def build_stream_prompt(history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    system = f"""
You are an AI call center brain with 3 agents.

Pick one agent_key:
- primary: general help
- supervisor: angry/complaints/policy/manager
- escalation: refund/billing/cancel/security/fraud/human handoff

You MUST reply in TWO parts:

PART 1: One-line JSON header ONLY (no markdown):
{{"agent_key":"primary|supervisor|escalation","intent":"snake_case","emotion":"calm|neutral|happy|confused|frustrated|angry","confidence":0-100,"csat_prediction":1-5,"summary":"one line","resolved":true|false}}

PART 2: On next line output:
---REPLY---
Then write the assistant reply in short voice-friendly sentences.

Rules:
- Keep reply short and natural.
- Ask max 1 question.
- No extra JSON.
"""
    msgs = [{"role": "system", "content": system}]
    msgs += history[-10:]
    msgs.append({"role": "user", "content": user_text})
    return msgs


async def deepgram_tts_stream(text: str, voice_model: str):
    url = "https://api.deepgram.com/v1/speak"
    params = {"model": voice_model, "encoding": "linear16", "sample_rate": "24000"}
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, params=params, headers=headers, json={"text": text}) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk


_sentence_splitter = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _sentence_splitter.split(text)
    return [p.strip() for p in parts if p.strip()]


def make_silence_frame(duration_ms: int = 80, sample_rate: int = 16000) -> bytes:
    """
    linear16 mono silence.
    80ms @ 16k = 1280 samples -> 2560 bytes.
    """
    samples = int(sample_rate * duration_ms / 1000.0)
    return b"\x00\x00" * samples


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted ✅")

    history: List[Dict[str, str]] = []
    active_agent: AgentKey = "primary"
    introduced = {"primary": False, "supervisor": False, "escalation": False}
    first_user_ts: Optional[float] = None

    dg_connection = None
    ai_speaking = False

    # ✅ this task sends silence frames to Deepgram while AI speaks (prevents STT close)
    silence_task: Optional[asyncio.Task] = None

    async def silence_keepalive_loop():
        try:
            while True:
                if dg_connection and ai_speaking:
                    dg_connection.send(make_silence_frame())
                await asyncio.sleep(0.08)
        except Exception:
            return

    async def stream_tts(text: str, voice: str):
        await websocket.send_json({"type": "generating_audio"})
        async for chunk in deepgram_tts_stream(text, voice):
            await websocket.send_json(
                {"type": "audio_chunk", "data": base64.b64encode(chunk).decode("utf-8")}
            )
        await websocket.send_json({"type": "audio_complete"})

    async def intro_agent(agent_key: AgentKey):
        nonlocal ai_speaking
        if introduced.get(agent_key):
            return

        introduced[agent_key] = True
        agent_name = AGENTS[agent_key]["name"]
        intro = AGENTS[agent_key]["intro"]
        voice = AGENTS[agent_key]["voice"]

        await websocket.send_json({
            "type": "ai_message",
            "agent_key": agent_key,
            "agent_name": agent_name,
            "text": intro,
            "is_intro": True,
        })

        ai_speaking = True
        await stream_tts(intro, voice)
        ai_speaking = False

        await websocket.send_json({"type": "listening"})

    async def handle_user_turn(user_text: str):
        nonlocal ai_speaking, active_agent, first_user_ts

        if first_user_ts is None:
            first_user_ts = time.time()

        await websocket.send_json({"type": "ai_thinking"})

        # ✅ STREAM GROQ OUTPUT
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=build_stream_prompt(history, user_text),
            temperature=0.2,
            max_tokens=350,
            stream=True,
        )

        # read first JSON header line quickly
        buffer = ""
        header_json = None
        reply_started = False
        reply_text_accum = ""

        ai_speaking = True

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                buffer += chunk.choices[0].delta.content

                # if header not parsed yet, try parse first line
                if header_json is None and "\n" in buffer:
                    first_line, rest = buffer.split("\n", 1)

                    # parse header
                    try:
                        header_json = HeaderPacket.model_validate(json.loads(first_line.strip()))
                    except Exception:
                        # fallback header if LLM bad
                        header_json = HeaderPacket(
                            agent_key="primary",
                            intent="general_inquiry",
                            emotion="neutral",
                            confidence=55,
                            csat_prediction=3,
                            summary="Fallback header",
                            resolved=False,
                        )

                    buffer = rest

                # detect reply marker
                if header_json and (not reply_started) and "---REPLY---" in buffer:
                    buffer = buffer.split("---REPLY---", 1)[1]
                    reply_started = True

                # once reply started, speak sentence-by-sentence
                if reply_started:
                    reply_text_accum += buffer
                    buffer = ""

                    # speak complete sentences early
                    sentences = split_into_sentences(reply_text_accum)

                    # keep last incomplete sentence in accumulator
                    if sentences:
                        # if last char is not sentence end, last sentence may be partial
                        last_char = reply_text_accum.strip()[-1]
                        complete_count = len(sentences)
                        if last_char not in ".!?":
                            complete_count -= 1

                        # speak complete sentences
                        for i in range(complete_count):
                            s = sentences[i]
                            if s:
                                await stream_tts(s, AGENTS[active_agent]["voice"])

                        # keep remaining
                        if complete_count > 0:
                            remaining = " ".join(sentences[complete_count:])
                            reply_text_accum = remaining

        # after stream ends, speak remaining text
        final_reply = reply_text_accum.strip()
        if final_reply:
            await stream_tts(final_reply, AGENTS[active_agent]["voice"])

        # update context memory
        history.append({"role": "user", "content": user_text})

        # build final full reply (not perfect but good for context)
        # (we store summary + partial reply)
        stored_reply = final_reply if final_reply else "Okay."
        history.append({"role": "assistant", "content": stored_reply})

        # route agent
        new_agent = header_json.agent_key if header_json else "primary"

        if new_agent != active_agent:
            active_agent = new_agent
            await websocket.send_json({
                "type": "agent_switch",
                "agent_key": active_agent,
                "agent_name": AGENTS[active_agent]["name"],
            })
            if not introduced[active_agent]:
                await intro_agent(active_agent)

        resolution_time_sec = int(time.time() - first_user_ts)

        # send packet for dashboard + chat
        await websocket.send_json({
            "type": "ai_packet",
            "agent_key": active_agent,
            "agent_name": AGENTS[active_agent]["name"],
            "reply": stored_reply,
            "intent": header_json.intent if header_json else "general_inquiry",
            "emotion": header_json.emotion if header_json else "neutral",
            "confidence": header_json.confidence if header_json else 55,
            "csat_prediction": header_json.csat_prediction if header_json else 3,
            "summary": header_json.summary if header_json else "Fallback summary",
            "resolved": header_json.resolved if header_json else False,
            "resolution_time_sec": resolution_time_sec,
        })

        ai_speaking = False
        await websocket.send_json({"type": "listening"})

    try:
        # ✅ KEEP YOUR WORKING Deepgram STT setup unchanged
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.live.v("1")
        loop = asyncio.get_event_loop()

        # start silence keepalive loop
        silence_task = asyncio.create_task(silence_keepalive_loop())

        def on_message(self, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence:
                    return

                if ai_speaking:
                    return

                if not result.is_final:
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json({"type": "transcript", "text": sentence, "is_final": False}),
                        loop,
                    )
                    return

                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": "transcript", "text": sentence, "is_final": True}),
                    loop,
                )

                asyncio.run_coroutine_threadsafe(handle_user_turn(sentence), loop)

            except Exception as e:
                print("on_message error:", e)

        def on_error(self, error, **kwargs):
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({"type": "error", "message": f"STT error: {str(error)}"}),
                loop,
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
            endpointing=250,
            utterance_end_ms=900,
            vad_events=True,
            smart_format=True,
            punctuate=True,
        )

        if dg_connection.start(options):
            await websocket.send_json({"type": "ready"})
            await intro_agent("primary")
        else:
            raise Exception("Failed to start Deepgram STT")

        # receive loop
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

                if "bytes" in message:
                    dg_connection.send(message["bytes"])

                elif "text" in message:
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        break

                    if data.get("type") == "clear_history":
                        history.clear()
                        active_agent = "primary"
                        introduced = {"primary": False, "supervisor": False, "escalation": False}
                        first_user_ts = None
                        await websocket.send_json({"type": "history_cleared"})
                        await intro_agent("primary")

                    if data.get("type") == "text_message":
                        text = (data.get("text") or "").strip()
                        if text:
                            await websocket.send_json({"type": "typed_message_ack", "text": text})
                            if not ai_speaking:
                                await handle_user_turn(text)

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print("Receive loop error:", e)
                break

    finally:
        try:
            if silence_task:
                silence_task.cancel()
        except Exception:
            pass

        try:
            if dg_connection:
                dg_connection.finish()
        except Exception:
            pass

        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/")
async def root():
    return {"message": "Ultra Realtime Voice Backend Running", "status": "OK"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
