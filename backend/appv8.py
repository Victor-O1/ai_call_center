# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from deepgram import (
#     DeepgramClient,
#     DeepgramClientOptions,
#     LiveTranscriptionEvents,
#     LiveOptions,
#     SpeakOptions,
# )
# import asyncio
# import json
# import os
# import time
# import base64
# from dotenv import load_dotenv
# from groq import Groq
# from pydantic import BaseModel, Field, ValidationError
# from typing import Optional, List, Dict, Literal

# load_dotenv()

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# groq_client = Groq(api_key=GROQ_API_KEY)

# # ==============================
# # ✅ Agents
# # ==============================
# AGENTS = {
#     "primary": {
#         "name": "Ava (Primary Agent)",
#         "voice": "aura-asteria-en",
#         "intro": "Hey! I’m Ava, your primary assistant. Tell me what you need help with.",
#     },
#     "supervisor": {
#         "name": "Noah (Supervisor Agent)",
#         "voice": "aura-luna-en",
#         "intro": "Hi, I’m Noah, the supervisor agent. I’ll help you resolve this smoothly.",
#     },
#     "escalation": {
#         "name": "Maya (Escalation Agent)",
#         "voice": "aura-hera-en",
#         "intro": "Hello, I’m Maya from escalations. I’ll handle this with priority.",
#     },
# }

# # ==============================
# # ✅ Pydantic contract (STRICT)
# # ==============================
# EmotionLabel = Literal["calm", "neutral", "happy", "confused", "frustrated", "angry"]

# class AgentPacket(BaseModel):
#     agent_key: Literal["primary", "supervisor", "escalation"]
#     reply: str

#     intent: str
#     emotion: EmotionLabel
#     confidence: int = Field(..., ge=0, le=100)
#     csat_prediction: int = Field(..., ge=1, le=5)
#     summary: str
#     resolved: bool

#     # server will fill these (not required from LLM)
#     agent_name: Optional[str] = None


# def groq_chat(messages: List[Dict[str, str]], max_tokens=450) -> str:
#     completion = groq_client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=messages,
#         temperature=0.25,
#         max_tokens=max_tokens,
#     )
#     return completion.choices[0].message.content.strip()


# def build_single_call_prompt(history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
#     """
#     Single prompt that forces output JSON.
#     """
#     system = f"""
# You are a multi-agent AI call center brain.

# You must pick exactly ONE agent_key:
# - "primary": general help / fast support
# - "supervisor": complaints, angry customer, manager request, policy confusion
# - "escalation": refund, billing/payment dispute, cancellation, account/security, OTP, fraud, human handoff

# Return ONLY valid JSON. No markdown. No extra keys.
# Must follow this schema EXACTLY:

# {{
#   "agent_key": "primary|supervisor|escalation",
#   "reply": "string (2-3 sentences, voice friendly, ask max 1 question)",
#   "intent": "snake_case label like refund_request, complaint, order_status, general_inquiry",
#   "emotion": "calm|neutral|happy|confused|frustrated|angry",
#   "confidence": integer 0-100,
#   "csat_prediction": integer 1-5,
#   "summary": "one line summary",
#   "resolved": true|false
# }}

# Rules:
# - reply should be short, natural for speech
# - ask only 1 question max
# - if user asks refund/billing/cancel/account/security/human -> escalation
# - if user is angry/complaint/manager/policy -> supervisor
# - otherwise -> primary
# """

#     messages = [{"role": "system", "content": system}]
#     messages += history[-10:]
#     messages.append({"role": "user", "content": user_text})
#     return messages


# def try_parse_packet(raw: str) -> AgentPacket:
#     """
#     Parse and validate with Pydantic.
#     """
#     data = json.loads(raw)
#     packet = AgentPacket.model_validate(data)

#     # server-side enforce agent_name from config
#     packet.agent_name = AGENTS[packet.agent_key]["name"]
#     return packet


# def repair_json_with_llm(raw: str) -> str:
#     """
#     Only used if initial model output is invalid JSON / wrong schema.
#     Small + fast.
#     """
#     repair_prompt = [
#         {
#             "role": "system",
#             "content": """Fix the following into VALID JSON that matches schema exactly.
# Return ONLY JSON, no markdown, no explanation.

# Schema:
# {
#   "agent_key": "primary|supervisor|escalation",
#   "reply": "string",
#   "intent": "string",
#   "emotion": "calm|neutral|happy|confused|frustrated|angry",
#   "confidence": 0-100 integer,
#   "csat_prediction": 1-5 integer,
#   "summary": "string",
#   "resolved": true|false
# }
# """
#         },
#         {"role": "user", "content": raw},
#     ]
#     return groq_chat(repair_prompt, max_tokens=220)


# def one_call_packet(history: List[Dict[str, str]], user_text: str) -> AgentPacket:
#     """
#     ONE call normally.
#     Only if invalid output -> repair once.
#     """
#     messages = build_single_call_prompt(history, user_text)
#     raw = groq_chat(messages, max_tokens=500)

#     # 1) Try parse
#     try:
#         return try_parse_packet(raw)
#     except Exception:
#         pass

#     # 2) Try repair once
#     try:
#         repaired = repair_json_with_llm(raw)
#         return try_parse_packet(repaired)
#     except Exception:
#         # 3) Hard fallback (always valid)
#         fallback = AgentPacket(
#             agent_key="primary",
#             reply="Sorry, I didn’t catch that properly. Can you say it again in one sentence?",
#             intent="general_inquiry",
#             emotion="neutral",
#             confidence=55,
#             csat_prediction=3,
#             summary="Fallback due to invalid model output",
#             resolved=False,
#             agent_name=AGENTS["primary"]["name"],
#         )
#         return fallback


# @app.websocket("/ws/voice")
# async def voice_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("WebSocket connected ✅")

#     dg_connection = None
#     ai_speaking = False
#     tts_done_event = asyncio.Event()

#     history: List[Dict[str, str]] = []
#     active_agent_key = "primary"
#     introduced = {"primary": False, "supervisor": False, "escalation": False}
#     first_user_ts: Optional[float] = None

#     async def speak_tts(deepgram: DeepgramClient, text: str, voice_model: str):
#         nonlocal ai_speaking
#         ai_speaking = True
#         tts_done_event.clear()

#         speak_options = SpeakOptions(model=voice_model, encoding="linear16", sample_rate=24000)
#         deepgram.speak.v("1").save("audio_output.wav", {"text": text}, speak_options)

#         with open("audio_output.wav", "rb") as f:
#             audio_data = f.read()

#         chunk_size = 4096
#         for i in range(0, len(audio_data), chunk_size):
#             chunk = audio_data[i : i + chunk_size]
#             await websocket.send_json({
#                 "type": "audio_chunk",
#                 "data": base64.b64encode(chunk).decode("utf-8"),
#             })

#         if os.path.exists("audio_output.wav"):
#             os.remove("audio_output.wav")

#         await websocket.send_json({"type": "audio_complete"})
#         await tts_done_event.wait()
#         ai_speaking = False

#     async def process_user_text(deepgram: DeepgramClient, text: str, source: str):
#         nonlocal history, active_agent_key, first_user_ts

#         if first_user_ts is None:
#             first_user_ts = time.time()

#         await websocket.send_json({"type": "ai_thinking"})

#         # ✅ ONE validated packet
#         packet = one_call_packet(history, text)

#         switched = packet.agent_key != active_agent_key
#         active_agent_key = packet.agent_key

#         # Update history
#         history.append({"role": "user", "content": text})
#         history.append({"role": "assistant", "content": packet.reply})

#         resolution_time = int(time.time() - first_user_ts)

#         # Inform frontend about switch
#         if switched:
#             await websocket.send_json({
#                 "type": "agent_switch",
#                 "agent_key": packet.agent_key,
#                 "agent_name": packet.agent_name,
#             })

#         # Agent intro once (fast)
#         if not introduced[packet.agent_key]:
#             introduced[packet.agent_key] = True
#             intro = AGENTS[packet.agent_key]["intro"]

#             # show intro message in UI
#             await websocket.send_json({
#                 "type": "ai_message",
#                 "agent_key": packet.agent_key,
#                 "agent_name": packet.agent_name,
#                 "text": intro,
#                 "is_intro": True
#             })

#             await speak_tts(deepgram, intro, AGENTS[packet.agent_key]["voice"])

#         # Send assistant message + analytics in one event
#         await websocket.send_json({
#             "type": "ai_packet",
#             "agent_key": packet.agent_key,
#             "agent_name": packet.agent_name,
#             "reply": packet.reply,
#             "intent": packet.intent,
#             "emotion": packet.emotion,
#             "confidence": packet.confidence,
#             "csat_prediction": packet.csat_prediction,
#             "summary": packet.summary,
#             "resolved": packet.resolved,
#             "resolution_time_sec": resolution_time,
#         })

#         # Speak reply
#         await speak_tts(deepgram, packet.reply, AGENTS[packet.agent_key]["voice"])

#         await websocket.send_json({"type": "listening"})

#     try:
#         config = DeepgramClientOptions(options={"keepalive": "true"})
#         deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

#         dg_connection = deepgram.listen.live.v("1")
#         loop = asyncio.get_event_loop()

#         def on_message(self, result, **kwargs):
#             try:
#                 sentence = result.channel.alternatives[0].transcript
#                 if not sentence:
#                     return
#                 asyncio.run_coroutine_threadsafe(
#                     handle_voice_transcript(sentence, result.is_final),
#                     loop,
#                 )
#             except Exception as e:
#                 print("on_message error:", e)

#         async def handle_voice_transcript(sentence: str, is_final: bool):
#             nonlocal ai_speaking
#             if ai_speaking:
#                 return

#             if not is_final:
#                 await websocket.send_json({"type": "transcript", "text": sentence, "is_final": False})
#                 return

#             await websocket.send_json({"type": "transcript", "text": sentence, "is_final": True})
#             await process_user_text(deepgram, sentence, source="voice")

#         def on_error(self, error, **kwargs):
#             asyncio.run_coroutine_threadsafe(
#                 websocket.send_json({"type": "error", "message": f"STT error: {str(error)}"}),
#                 loop,
#             )

#         dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
#         dg_connection.on(LiveTranscriptionEvents.Error, on_error)

#         options = LiveOptions(
#             model="nova-2",
#             language="en-US",
#             encoding="linear16",
#             sample_rate=16000,
#             channels=1,
#             interim_results=True,
#             endpointing=300,
#             utterance_end_ms=1500,
#             vad_events=True,
#             smart_format=True,
#             punctuate=True,
#         )

#         if dg_connection.start(options):
#             await websocket.send_json({"type": "ready"})
#         else:
#             raise Exception("Failed to start Deepgram STT")

#         while True:
#             try:
#                 message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

#                 if "bytes" in message:
#                     if not ai_speaking:
#                         dg_connection.send(message["bytes"])

#                 elif "text" in message:
#                     data = json.loads(message["text"])

#                     if data.get("type") == "stop":
#                         break

#                     elif data.get("type") == "tts_done":
#                         tts_done_event.set()

#                     elif data.get("type") == "clear_history":
#                         history = []
#                         active_agent_key = "primary"
#                         introduced = {"primary": False, "supervisor": False, "escalation": False}
#                         first_user_ts = None
#                         await websocket.send_json({"type": "history_cleared"})

#                     # ✅ typed chat
#                     elif data.get("type") == "text_message":
#                         user_text = (data.get("text") or "").strip()
#                         if user_text:
#                             await websocket.send_json({"type": "typed_message_ack", "text": user_text})
#                             if not ai_speaking:
#                                 await process_user_text(deepgram, user_text, source="typed")

#             except asyncio.TimeoutError:
#                 continue
#             except WebSocketDisconnect:
#                 break
#             except Exception as e:
#                 print("Receive loop error:", e)
#                 break

#     finally:
#         if dg_connection:
#             dg_connection.finish()
#         try:
#             await websocket.close()
#         except:
#             pass


# @app.get("/")
# async def root():
#     return {"message": "Voice AI Backend Running", "status": "OK"}


# @app.get("/health")
# async def health():
#     return {"status": "healthy"}


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    SpeakOptions,
)
import asyncio
import json
import os
import time
import base64
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal

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
# ✅ Pydantic contract
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


def groq_chat(messages: List[Dict[str, str]], max_tokens: int = 450) -> str:
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.25,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


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

Routing rules:
- refund/billing/cancel/account/security/human -> escalation
- complaint/angry/manager/policy -> supervisor
- otherwise -> primary
"""
    msgs = [{"role": "system", "content": system}]
    msgs += history[-10:]
    msgs.append({"role": "user", "content": user_text})
    return msgs


def safe_parse_packet(raw: str) -> Optional[AgentPacket]:
    """
    Parse JSON + validate with Pydantic.
    """
    try:
        data = json.loads(raw)
        return AgentPacket.model_validate(data)
    except Exception:
        return None


def repair_with_llm(raw: str) -> str:
    """
    Fix bad JSON into valid JSON for our schema.
    """
    repair_prompt = [
        {
            "role": "system",
            "content": """Convert the following into VALID JSON for this schema ONLY.
Return ONLY JSON, no markdown.

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
}
""",
        },
        {"role": "user", "content": raw},
    ]
    return groq_chat(repair_prompt, max_tokens=220)


def one_call(history: List[Dict[str, str]], user_text: str) -> AgentPacket:
    """
    Single Groq call per message normally.
    If output invalid -> auto repair once.
    If still invalid -> fallback.
    """
    raw = groq_chat(build_prompt(history, user_text), max_tokens=500)

    packet = safe_parse_packet(raw)
    if packet:
        return packet

    # repair once
    repaired = repair_with_llm(raw)
    packet2 = safe_parse_packet(repaired)
    if packet2:
        return packet2

    # fallback always valid
    return AgentPacket(
        agent_key="primary",
        reply="Sorry, I didn’t catch that properly. Can you repeat it in one sentence?",
        intent="general_inquiry",
        emotion="neutral",
        confidence=55,
        csat_prediction=3,
        summary="Fallback due to invalid model output",
        resolved=False,
    )


@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted ✅")

    dg_connection = None
    ai_speaking = False
    tts_done_event = asyncio.Event()

    history: List[Dict[str, str]] = []
    active_agent_key: AgentKey = "primary"
    introduced = {"primary": False, "supervisor": False, "escalation": False}
    first_user_ts: Optional[float] = None

    async def speak_tts(deepgram: DeepgramClient, text: str, voice_model: str):
        """
        Generate WAV → stream chunks → wait tts_done from frontend.
        """
        nonlocal ai_speaking
        ai_speaking = True
        tts_done_event.clear()

        speak_options = SpeakOptions(model=voice_model, encoding="linear16", sample_rate=24000)

        deepgram.speak.v("1").save("audio_output.wav", {"text": text}, speak_options)

        with open("audio_output.wav", "rb") as f:
            audio_data = f.read()

        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            await websocket.send_json(
                {"type": "audio_chunk", "data": base64.b64encode(chunk).decode("utf-8")}
            )

        if os.path.exists("audio_output.wav"):
            os.remove("audio_output.wav")

        await websocket.send_json({"type": "audio_complete"})

        # ✅ wait until audio playback finished on client
        await tts_done_event.wait()
        ai_speaking = False

    async def process_text(deepgram: DeepgramClient, user_text: str):
        """
        Process voice-final OR typed message.
        """
        nonlocal history, active_agent_key, first_user_ts

        if first_user_ts is None:
            first_user_ts = time.time()

        await websocket.send_json({"type": "ai_thinking"})

        # ✅ one-call packet
        packet = one_call(history, user_text)

        # agent switch detection
        switched = packet.agent_key != active_agent_key
        active_agent_key = packet.agent_key

        agent_name = AGENTS[packet.agent_key]["name"]
        agent_voice = AGENTS[packet.agent_key]["voice"]

        # save history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": packet.reply})

        # metrics
        resolution_time_sec = int(time.time() - first_user_ts)

        # notify switch
        if switched:
            await websocket.send_json({
                "type": "agent_switch",
                "agent_key": packet.agent_key,
                "agent_name": agent_name,
            })

        # intro once per agent (shows as chat + spoken)
        if not introduced[packet.agent_key]:
            introduced[packet.agent_key] = True
            intro = AGENTS[packet.agent_key]["intro"]

            await websocket.send_json({
                "type": "ai_message",
                "agent_key": packet.agent_key,
                "agent_name": agent_name,
                "text": intro,
                "is_intro": True
            })

            await speak_tts(deepgram, intro, agent_voice)

        # ✅ send the packet event frontend expects
        await websocket.send_json({
            "type": "ai_packet",
            "agent_key": packet.agent_key,
            "agent_name": agent_name,
            "reply": packet.reply,
            "intent": packet.intent,
            "emotion": packet.emotion,
            "confidence": packet.confidence,
            "csat_prediction": packet.csat_prediction,
            "summary": packet.summary,
            "resolved": packet.resolved,
            "resolution_time_sec": resolution_time_sec,
        })

        # speak reply
        await speak_tts(deepgram, packet.reply, agent_voice)

        await websocket.send_json({"type": "listening"})

    try:
        # deepgram
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)

        dg_connection = deepgram.listen.live.v("1")
        loop = asyncio.get_event_loop()

        def on_message(self, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence:
                    return
                asyncio.run_coroutine_threadsafe(
                    handle_transcript(sentence, result.is_final),
                    loop,
                )
            except Exception as e:
                print("on_message error:", e)

        async def handle_transcript(sentence: str, is_final: bool):
            nonlocal ai_speaking

            # block STT while AI speaking
            if ai_speaking:
                return

            if not is_final:
                await websocket.send_json({"type": "transcript", "text": sentence, "is_final": False})
                return

            # final transcript (display user msg)
            await websocket.send_json({"type": "transcript", "text": sentence, "is_final": True})
            await process_text(deepgram, sentence)

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
            endpointing=300,
            utterance_end_ms=1500,
            vad_events=True,
            smart_format=True,
            punctuate=True,
        )

        if dg_connection.start(options):
            await websocket.send_json({"type": "ready"})
        else:
            raise Exception("Failed to start Deepgram STT")

        # receive loop
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)

                # audio bytes → Deepgram STT
                if "bytes" in message:
                    if not ai_speaking:
                        dg_connection.send(message["bytes"])

                # json messages
                elif "text" in message:
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        break

                    elif data.get("type") == "tts_done":
                        tts_done_event.set()

                    elif data.get("type") == "clear_history":
                        history = []
                        active_agent_key = "primary"
                        introduced = {"primary": False, "supervisor": False, "escalation": False}
                        first_user_ts = None
                        await websocket.send_json({"type": "history_cleared"})

                    # ✅ typed messages
                    elif data.get("type") == "text_message":
                        user_text = (data.get("text") or "").strip()
                        if user_text:
                            await websocket.send_json({"type": "typed_message_ack", "text": user_text})
                            if not ai_speaking:
                                await process_text(deepgram, user_text)

            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                print("Client disconnected")
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
    return {
        "message": "Voice AI Backend Running",
        "status": "OK",
        "endpoints": {"websocket": "/ws/voice"},
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    print("Starting Voice AI Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info")
