from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import asyncio
import base64
import json
import os
import time
from dotenv import load_dotenv

from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal
import httpx

import websockets
import httpx
import base64
import time
# ==============================
# ✅ Setup
# ==============================
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

print("\n==============================")
print("BOOT DEBUG (RAW STT + RAW TTS)")
print("==============================")
print(f"[ENV] Deepgram API Key: {'SET ✅' if DEEPGRAM_API_KEY else 'NOT SET ❌'}")
print(f"[ENV] Deepgram Key len: {len(DEEPGRAM_API_KEY)}")
print(f"[ENV] Groq API Key: {'SET ✅' if GROQ_API_KEY else 'NOT SET ❌'}")
print("==============================\n")

groq_client = Groq(api_key=GROQ_API_KEY)

# ==============================
# ✅ Agents (voices + names)
# ==============================
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


# ==============================
# ✅ Contract
# ==============================
class AgentPacket(BaseModel):
    agent_key: AgentKey
    reply: str
    intent: str
    emotion: EmotionLabel
    confidence: int = Field(..., ge=0, le=100)
    csat_prediction: int = Field(..., ge=1, le=5)
    summary: str
    resolved: bool


# ==============================
# ✅ Groq agent router
# ==============================
def groq_chat(messages: List[Dict[str, str]], max_tokens: int = 450) -> str:
    print(f"\n[GROQ] >>> calling | messages={len(messages)} max_tokens={max_tokens}")
    t0 = time.time()
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.25,
        max_tokens=max_tokens,
    )
    dt = (time.time() - t0) * 1000
    out = completion.choices[0].message.content.strip()
    print(f"[GROQ] <<< done in {dt:.0f} ms")
    print("[GROQ] RAW:", out[:450], "..." if len(out) > 450 else "")
    return out


def build_prompt(history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    system = """
You are a multi-agent AI call center brain.

You must choose ONE agent_key:
- "primary": general help / fast support
- "supervisor": complaints, angry customer, manager request, policy confusion
- "escalation": refund, billing/payment dispute, cancellation, account/security, OTP, fraud, human handoff

Return ONLY valid JSON. No markdown. No explanation. No extra keys.

Schema EXACTLY:
{
  "agent_key": "primary|supervisor|escalation",
  "reply": "2-3 short sentences, voice friendly, max 1 question",
  "intent": "snake_case label like refund_request, complaint, order_status, general_inquiry",
  "emotion": "calm|neutral|happy|confused|frustrated|angry",
  "confidence": 0-100 integer,
  "csat_prediction": 1-5 integer,
  "summary": "one line summary",
  "resolved": true|false
}
"""
    msgs = [{"role": "system", "content": system}]
    msgs += history[-10:]
    msgs.append({"role": "user", "content": user_text})
    return msgs


def safe_parse_packet(raw: str) -> Optional[AgentPacket]:
    print("[PARSE] parsing JSON...")
    try:
        data = json.loads(raw)
        pkt = AgentPacket.model_validate(data)
        print("[PARSE] ✅ OK agent:", pkt.agent_key, "| intent:", pkt.intent)
        return pkt
    except Exception as e:
        print("[PARSE] ❌ invalid JSON:", str(e))
        return None


def repair_with_llm(raw: str) -> str:
    print("[REPAIR] repairing JSON...")
    repair_prompt = [
        {
            "role": "system",
            "content": """Convert the following into VALID JSON for this schema ONLY.
Return ONLY JSON.

Schema:
{
  "agent_key": "primary|supervisor|escalation",
  "reply": "string",
  "intent": "string",
  "emotion": "calm|neutral|happy|confused|frustrated|angry",
  "confidence": 0-100 integer,
  "csat_prediction": 1-5 integer,
  "summary": "string",
  "resolved": true|false
}""",
        },
        {"role": "user", "content": raw},
    ]
    return groq_chat(repair_prompt, max_tokens=220)


def one_call(history: List[Dict[str, str]], user_text: str) -> AgentPacket:
    raw = groq_chat(build_prompt(history, user_text), max_tokens=500)

    pkt = safe_parse_packet(raw)
    if pkt:
        return pkt

    repaired = repair_with_llm(raw)
    pkt2 = safe_parse_packet(repaired)
    if pkt2:
        return pkt2

    print("[FALLBACK] returning fallback packet")
    return AgentPacket(
        agent_key="primary",
        reply="Sorry, I didn’t catch that properly. Can you repeat it in one sentence?",
        intent="general_inquiry",
        emotion="neutral",
        confidence=55,
        csat_prediction=3,
        summary="Fallback invalid model output",
        resolved=False,
    )


# ==============================
# ✅ Deepgram RAW STT WebSocket
# ==============================
def dg_stt_url():
    base = (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&language=en-US"
        "&encoding=linear16"
        "&sample_rate=16000"
        "&channels=1"
        "&interim_results=true"
        "&smart_format=true"
        "&punctuate=true"
    )
    print("[STT] Using STT URL:", base)
    return base


# ==============================
# ✅ Deepgram RAW TTS WebSocket
# ==============================
def dg_tts_url(voice_model: str):
    return (
        "wss://api.deepgram.com/v1/speak"
        f"?model={voice_model}"
        f"&encoding=wav"
        f"&sample_rate=24000"
    )


async def stream_tts_to_client(frontend_ws: WebSocket, text: str, voice_model: str):
    url = dg_tts_url(voice_model)

    print("\n[TTS] ==============================")
    print("[TTS] connecting:", url)
    print("[TTS] voice:", voice_model)
    print("[TTS] text:", text)
    print("[TTS] ==============================\n")

    t0 = time.time()
    chunks = 0
    total_bytes = 0
    got_audio = False

    try:
        async with websockets.connect(
            url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
            ping_interval=10,
            ping_timeout=20,
        ) as dg_ws:
            await dg_ws.send(json.dumps({"text": text}))
            print("[TTS] ✅ sent payload")

            while True:
                try:
                    msg = await dg_ws.recv()
                except websockets.exceptions.ConnectionClosedOK:
                    print("[TTS] ✅ Deepgram closed connection (done)")
                    break
                except websockets.exceptions.ConnectionClosedError as e:
                    print("[TTS] ❌ Deepgram connection closed with error:", repr(e))
                    break

                if isinstance(msg, bytes):
                    chunks += 1
                    total_bytes += len(msg)
                    if not got_audio:
                        got_audio = True
                        print("[TTS] ✅ first audio bytes received, len=", len(msg))

                    await frontend_ws.send_json(
                        {"type": "audio_chunk", "data": base64.b64encode(msg).decode("utf-8")}
                    )

                else:
                    # metadata / status text
                    try:
                        meta = json.loads(msg)
                        print("[TTS] meta:", meta)
                    except Exception:
                        print("[TTS] non-json:", msg)

            await frontend_ws.send_json({"type": "audio_complete"})

    finally:
        dt = (time.time() - t0) * 1000
        print(f"[TTS] ✅ complete {dt:.0f}ms chunks={chunks} bytes={total_bytes}")





async def tts_rest_wav_to_client(frontend_ws: WebSocket, text: str, voice_model: str):
    """
    Deepgram Speak REST → get WAV bytes → stream to frontend
    """
    url = "https://api.deepgram.com/v1/speak"

    params = {
        "model": voice_model,
        "encoding": "mp3",
        "sample_rate": 24000,
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"text": text}

    print("\n[TTS-REST] ==============================")
    print("[TTS-REST] URL:", url)
    print("[TTS-REST] params:", params)
    print("[TTS-REST] text:", text)
    print("[TTS-REST] ==============================\n")

    t0 = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, params=params, headers=headers, json=payload)

    print("[TTS-REST] status:", resp.status_code)
    if resp.status_code != 200:
        print("[TTS-REST] ❌ error body:", resp.text[:500])
        await frontend_ws.send_json({"type": "error", "message": "TTS failed"})
        return

    audio_data = resp.content
    print("[TTS-REST] ✅ got wav bytes:", len(audio_data))

    # stream chunks
    chunk_size = 8192
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        await frontend_ws.send_json({
            "type": "audio_chunk",
            "data": base64.b64encode(chunk).decode("utf-8"),
            "format": "wav"
        })

    await frontend_ws.send_json({"type": "audio_complete"})
    dt = (time.time() - t0) * 1000
    print(f"[TTS-REST] ✅ streamed in {dt:.0f} ms")




import httpx
import base64
import time

async def tts_rest_mp3_to_client(frontend_ws: WebSocket, text: str, voice_model: str):
    """
    Deepgram Speak REST → MP3 bytes → stream to frontend
    """
    url = "https://api.deepgram.com/v1/speak"

    params = {
        "model": voice_model,
        "encoding": "mp3",
    }

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {"text": text}

    print("\n[TTS-REST] ==============================")
    print("[TTS-REST] URL:", url)
    print("[TTS-REST] params:", params)
    print("[TTS-REST] voice:", voice_model)
    print("[TTS-REST] text:", text)
    print("[TTS-REST] ==============================\n")

    t0 = time.time()

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, params=params, headers=headers, json=payload)

    print("[TTS-REST] status:", resp.status_code)

    if resp.status_code != 200:
        print("[TTS-REST] ❌ error body:", resp.text[:600])
        await frontend_ws.send_json({"type": "error", "message": f"TTS failed: {resp.text}"})
        return

    audio_data = resp.content
    print("[TTS-REST] ✅ got mp3 bytes:", len(audio_data))

    # stream chunks
    chunk_size = 8192
    chunks = 0
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        chunks += 1
        if chunks == 1:
            print("[TTS-REST] ✅ sending first audio chunk len:", len(chunk))

        await frontend_ws.send_json({
            "type": "audio_chunk",
            "data": base64.b64encode(chunk).decode("utf-8"),
            "format": "mp3"
        })

    await frontend_ws.send_json({"type": "audio_complete"})
    dt = (time.time() - t0) * 1000
    print(f"[TTS-REST] ✅ streamed mp3 in {dt:.0f} ms | chunks={chunks}")


# ==============================
# ✅ Main WebSocket
# ==============================
@app.websocket("/ws/voice")
async def voice_endpoint(frontend_ws: WebSocket):
    await frontend_ws.accept()
    print("\n[WS] ✅ frontend websocket accepted /ws/voice")

    history: List[Dict[str, str]] = []
    active_agent_key: AgentKey = "primary"
    introduced = {"primary": False, "supervisor": False, "escalation": False}
    first_user_ts: Optional[float] = None

    # avoid overlapping turns
    ai_lock = asyncio.Lock()
    allow_barge_in = True
    tts_task: Optional[asyncio.Task] = None

    async def cancel_tts_if_running():
        nonlocal tts_task
        if tts_task and not tts_task.done():
            print("[BARGE-IN] Cancelling TTS task...")
            tts_task.cancel()
            try:
                await tts_task
            except Exception as e:
                print("[BARGE-IN] cancel result:", str(e))
        tts_task = None

    async def process_text(user_text: str):
        nonlocal history, active_agent_key, introduced, first_user_ts, tts_task

        async with ai_lock:
            print("\n[PIPELINE] ==============================")
            print("[PIPELINE] process_text() user_text:", user_text)
            print("[PIPELINE] ==============================\n")

            if first_user_ts is None:
                first_user_ts = time.time()

            await frontend_ws.send_json({"type": "ai_thinking"})

            pkt = one_call(history, user_text)

            switched = pkt.agent_key != active_agent_key
            active_agent_key = pkt.agent_key

            agent_name = AGENTS[pkt.agent_key]["name"]
            agent_voice = AGENTS[pkt.agent_key]["voice"]

            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": pkt.reply})

            resolution_time_sec = int(time.time() - first_user_ts)

            if switched:
                print("[ROUTER] ✅ agent switch ->", pkt.agent_key)
                await frontend_ws.send_json(
                    {"type": "agent_switch", "agent_key": pkt.agent_key, "agent_name": agent_name}
                )

            # intro once
            if not introduced[pkt.agent_key]:
                introduced[pkt.agent_key] = True
                intro = AGENTS[pkt.agent_key]["intro"]

                await frontend_ws.send_json(
                    {
                        "type": "ai_message",
                        "agent_key": pkt.agent_key,
                        "agent_name": agent_name,
                        "text": intro,
                        "is_intro": True,
                    }
                )

                tts_task = asyncio.create_task(tts_rest_mp3_to_client(frontend_ws, pkt.reply, agent_voice))
                await tts_task


            # send ai_packet
            await frontend_ws.send_json(
                {
                    "type": "ai_packet",
                    "agent_key": pkt.agent_key,
                    "agent_name": agent_name,
                    "reply": pkt.reply,
                    "intent": pkt.intent,
                    "emotion": pkt.emotion,
                    "confidence": pkt.confidence,
                    "csat_prediction": pkt.csat_prediction,
                    "summary": pkt.summary,
                    "resolved": pkt.resolved,
                    "resolution_time_sec": resolution_time_sec,
                }
            )
            print("[WS] ✅ sent ai_packet")

            # tts_task = asyncio.create_task(stream_tts_to_client(frontend_ws, pkt.reply, agent_voice))
            tts_task = asyncio.create_task(tts_rest_mp3_to_client(frontend_ws, pkt.reply, agent_voice))
            await tts_task

            await frontend_ws.send_json({"type": "listening"})
            print("[PIPELINE] ✅ listening")

    # ==============================
    # ✅ Connect to Deepgram STT WS
    # ==============================
    dg_url = dg_stt_url()
    print("\n[STT] ==============================")
    print("[STT] connecting to Deepgram RAW WS:", dg_url)
    print("[STT] ==============================\n")

    try:
        async with websockets.connect(
            dg_url,
            additional_headers=[("Authorization", f"Token {DEEPGRAM_API_KEY}")],
            ping_interval=10,
            ping_timeout=20,
        ) as dg_stt_ws:
            print("[STT] ✅ Deepgram STT WS connected!")
            await frontend_ws.send_json({"type": "ready"})

            async def dg_reader():
                """
                Reads transcripts from Deepgram and triggers process_text on final.
                """
                while True:
                    msg = await dg_stt_ws.recv()
                    if not isinstance(msg, str):
                        # STT messages are JSON text; ignore bytes
                        continue

                    try:
                        data = json.loads(msg)
                    except Exception:
                        print("[STT] non-json message:", msg)
                        continue

                    # DEBUG full payload (careful: can be huge)
                    # print("[STT] RAW JSON:", data)

                    # Deepgram Transcript result structure
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alts = channel.get("alternatives", [])
                        transcript = alts[0].get("transcript", "") if alts else ""
                        is_final = data.get("is_final", False)

                        if transcript:
                            print(f"[STT] transcript final={is_final} text='{transcript}'")
                            await frontend_ws.send_json(
                                {"type": "transcript", "text": transcript, "is_final": is_final}
                            )

                        if is_final and transcript.strip():
                            if allow_barge_in:
                                await cancel_tts_if_running()
                            await process_text(transcript.strip())

                    # VAD event logs
                    if "speech_final" in data:
                        print("[STT] speech_final:", data.get("speech_final"))

            dg_reader_task = asyncio.create_task(dg_reader())

            # ==============================
            # ✅ Receive from frontend loop
            # ==============================
            print("[WS] ✅ entering frontend receive loop...")
            while True:
                msg = await frontend_ws.receive()

                # audio bytes -> forward to Deepgram
                if msg.get("bytes") is not None:
                    b = msg["bytes"]
                    print(f"[WS] audio bytes in: {len(b)}")
                    await dg_stt_ws.send(b)

                # control json messages
                elif msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    print("[WS] json in:", data)

                    if data.get("type") == "stop":
                        print("[WS] stop received, closing...")
                        break

                    if data.get("type") == "clear_history":
                        history = []
                        active_agent_key = "primary"
                        introduced = {"primary": False, "supervisor": False, "escalation": False}
                        first_user_ts = None
                        await frontend_ws.send_json({"type": "history_cleared"})
                        print("[WS] ✅ history cleared")

                    if data.get("type") == "text_message":
                        user_text = (data.get("text") or "").strip()
                        if user_text:
                            await frontend_ws.send_json({"type": "typed_message_ack", "text": user_text})
                            if allow_barge_in:
                                await cancel_tts_if_running()
                            await process_text(user_text)

            # cleanup reader
            dg_reader_task.cancel()

    except WebSocketDisconnect:
        print("[WS] frontend disconnected")
    except Exception as e:
        print("[FATAL] Exception:", repr(e))
        try:
            await frontend_ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        print("[CLEANUP] closing frontend ws...")
        try:
            await frontend_ws.close()
        except Exception:
            pass
        print("[CLEANUP] ✅ done")


@app.get("/")
async def root():
    return {"message": "Voice AI Backend Running", "status": "OK", "endpoints": {"websocket": "/ws/voice"}}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Voice AI Backend (RAW STT RAW TTS DEBUG)...")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
