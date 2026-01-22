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
from dotenv import load_dotenv
import base64
from groq import Groq

# ✅ LangGraph + typing
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Optional

# ✅ Pydantic result model
from pydantic import BaseModel, Field, ValidationError

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

groq_client = Groq(api_key=GROQ_API_KEY)

# ==============================
# ✅ AGENT CONFIG
# ==============================
AGENTS = {
    "primary": {
        "name": "Ava (Primary Agent)",
        "voice": "aura-asteria-en",
        "system": """You are Ava, the Primary AI call center agent.
Speak friendly and short, like real phone conversation.
Goal: solve the issue quickly. Ask 1 question at a time.
If the user is angry, asks for manager, policy dispute, repeated unresolved issue → route to Supervisor.
If billing/refund/account security/cancellation → route to Escalation.""",
        "intro": "Hey! I’m Ava, your primary assistant. Tell me what you need help with.",
    },
    "supervisor": {
        "name": "Noah (Supervisor Agent)",
        "voice": "aura-luna-en",
        "system": """You are Noah, the Supervisor agent.
You handle complaints, angry users, policy clarification, and complex decision making.
Be calm and structured. Confirm the issue and propose steps.
If billing/refund/account/cancellation/human request → route to Escalation.""",
        "intro": "Hi, I’m Noah, the supervisor agent. I’ll help you resolve this smoothly.",
    },
    "escalation": {
        "name": "Maya (Escalation Agent)",
        "voice": "aura-hera-en",
        "system": """You are Maya, the Escalation agent.
Handle refunds, payment disputes, cancellations, account issues, and human-handoff.
Be professional and concise.
If user wants a human, prepare an escalation summary with key details.""",
        "intro": "Hello, I’m Maya from escalations. I’ll handle this with priority.",
    },
}

# ==============================
# ✅ Pydantic model for analysis
# ==============================
class LLMAnalysis(BaseModel):
    intent: str = Field(..., description="User intent label, e.g. refund_request, complaint, inquiry")
    emotion: str = Field(..., description="Emotion label, e.g. calm, angry, frustrated, confused, happy")
    confidence: int = Field(..., ge=0, le=100, description="Confidence percentage 0-100")
    csat_prediction: int = Field(..., ge=1, le=5, description="Predicted CSAT 1-5")
    summary: str = Field(..., description="Short 1-line summary of the situation")


def groq_chat(messages: List[Dict[str, str]], max_tokens=400) -> str:
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def analyze_user_text(user_text: str, last_assistant: str = "") -> LLMAnalysis:
    """
    Use Groq to return structured analysis: intent + emotion + confidence + csat + summary
    """
    prompt = [
        {
            "role": "system",
            "content": """You are a strict JSON generator for call-center analytics.
Return ONLY valid JSON (no markdown).
Schema:
{
  "intent": "string",
  "emotion": "string",
  "confidence": 0-100 integer,
  "csat_prediction": 1-5 integer,
  "summary": "short 1-line summary"
}
Intent should be short snake_case label."""
        },
        {
            "role": "user",
            "content": f"""User said: {user_text}
Previous assistant message (if any): {last_assistant}

Return JSON now."""
        }
    ]
    raw = groq_chat(prompt, max_tokens=200)

    # Try parse JSON safely
    try:
        data = json.loads(raw)
        return LLMAnalysis(**data)
    except (json.JSONDecodeError, ValidationError):
        # fallback if model outputs messy text
        return LLMAnalysis(
            intent="general_inquiry",
            emotion="neutral",
            confidence=60,
            csat_prediction=3,
            summary="User asked something; auto-fallback analytics",
        )


# ==============================
# ✅ LangGraph State
# ==============================
class GraphState(TypedDict):
    messages: List[Dict[str, str]]
    active_agent: str
    user_text: str
    agent_reply: str
    agent_switched: bool
    switch_to: Optional[str]
    analysis: Dict


def supervisor_router(state: GraphState) -> GraphState:
    text = state["user_text"].lower()

    # Fast routing rules
    if any(k in text for k in ["human", "agent", "representative", "manager", "complaint"]):
        state["switch_to"] = "supervisor"

    if any(k in text for k in ["refund", "payment", "charged", "billing", "cancel", "account", "otp", "fraud"]):
        state["switch_to"] = "escalation"

    if not state.get("switch_to"):
        # LLM routing
        routing_prompt = [
            {"role": "system", "content": "Choose one word only: primary OR supervisor OR escalation."},
            {"role": "user", "content": f"Message: {state['user_text']}"},
        ]
        decision = groq_chat(routing_prompt, max_tokens=10).lower()
        if "supervisor" in decision:
            state["switch_to"] = "supervisor"
        elif "escalation" in decision:
            state["switch_to"] = "escalation"
        else:
            state["switch_to"] = "primary"

    state["agent_switched"] = (state["switch_to"] != state["active_agent"])
    state["active_agent"] = state["switch_to"]
    return state


def agent_node(agent_key: str):
    def _node(state: GraphState) -> GraphState:
        cfg = AGENTS[agent_key]
        agent_messages = [{"role": "system", "content": cfg["system"]}]
        agent_messages += state["messages"][-12:]
        agent_messages.append({"role": "user", "content": state["user_text"]})

        reply = groq_chat(agent_messages, max_tokens=450)

        state["agent_reply"] = reply
        state["messages"].append({"role": "user", "content": state["user_text"]})
        state["messages"].append({"role": "assistant", "content": reply})

        return state

    return _node


graph = StateGraph(GraphState)
graph.add_node("router", supervisor_router)
graph.add_node("primary", agent_node("primary"))
graph.add_node("supervisor", agent_node("supervisor"))
graph.add_node("escalation", agent_node("escalation"))
graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["active_agent"],
    {"primary": "primary", "supervisor": "supervisor", "escalation": "escalation"},
)

graph.add_edge("primary", END)
graph.add_edge("supervisor", END)
graph.add_edge("escalation", END)

LANGGRAPH_APP = graph.compile()

# ==============================
# ✅ WebSocket
# ==============================
@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")

    dg_connection = None
    ai_speaking = False
    tts_done_event = asyncio.Event()

    state: GraphState = {
        "messages": [],
        "active_agent": "primary",
        "user_text": "",
        "agent_reply": "",
        "agent_switched": False,
        "switch_to": None,
        "analysis": {},
    }

    introduced = {"primary": False, "supervisor": False, "escalation": False}

    # ✅ tracking metrics
    session_start = time.time()
    first_user_ts = None
    resolved = False

    async def speak_tts(deepgram: DeepgramClient, text: str, voice_model: str):
        nonlocal ai_speaking
        ai_speaking = True
        tts_done_event.clear()

        speak_options = SpeakOptions(
            model=voice_model,
            encoding="linear16",
            sample_rate=24000,
        )

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

        await tts_done_event.wait()
        ai_speaking = False

    async def handle_user_message(deepgram: DeepgramClient, text: str, source: str):
        """
        This handles BOTH voice and typed messages.
        """
        nonlocal state, first_user_ts, resolved

        if first_user_ts is None:
            first_user_ts = time.time()

        # send transcript type based on source
        if source == "voice":
            await websocket.send_json({"type": "transcript", "text": text, "is_final": True})
        else:
            await websocket.send_json({"type": "typed_message_ack", "text": text})

        await websocket.send_json({"type": "ai_thinking"})

        # ✅ analytics (intent/emotion/confidence/csat)
        last_assistant = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        analysis = analyze_user_text(text, last_assistant)
        state["analysis"] = analysis.model_dump()

        await websocket.send_json({
            "type": "analysis",
            **analysis.model_dump()
        })

        # ✅ langgraph routing + agent response
        state["user_text"] = text
        state["switch_to"] = None

        new_state = LANGGRAPH_APP.invoke(state)
        state = new_state

        agent_key = state["active_agent"]
        agent_cfg = AGENTS[agent_key]

        # Switch event
        if state["agent_switched"]:
            await websocket.send_json({
                "type": "agent_switch",
                "agent_key": agent_key,
                "agent_name": agent_cfg["name"],
            })

        # Intro once per agent
        if not introduced[agent_key]:
            introduced[agent_key] = True
            await websocket.send_json({
                "type": "ai_message",
                "agent_key": agent_key,
                "agent_name": agent_cfg["name"],
                "text": agent_cfg["intro"],
            })
            await speak_tts(deepgram, agent_cfg["intro"], agent_cfg["voice"])

        reply = state["agent_reply"]

        # ✅ send assistant message
        await websocket.send_json({
            "type": "ai_message",
            "agent_key": agent_key,
            "agent_name": agent_cfg["name"],
            "text": reply,
        })

        # ✅ metrics package
        resolution_time = int(time.time() - first_user_ts) if first_user_ts else 0

        # naive "resolved" detection (you can improve)
        if any(k in reply.lower() for k in ["anything else", "glad i could help", "resolved", "happy to help"]):
            resolved = True

        await websocket.send_json({
            "type": "metrics",
            "confidence": analysis.confidence,
            "csat_prediction": analysis.csat_prediction,
            "resolution_time_sec": resolution_time,
            "resolved": resolved,
        })

        # ✅ speak assistant reply
        await speak_tts(deepgram, reply, agent_cfg["voice"])

        await websocket.send_json({"type": "listening"})

    try:
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
                    handle_voice_transcript(sentence, result.is_final),
                    loop,
                )
            except Exception as e:
                print("on_message error:", e)

        async def handle_voice_transcript(sentence: str, is_final: bool):
            nonlocal ai_speaking
            if ai_speaking:
                return

            if not is_final:
                await websocket.send_json({"type": "transcript", "text": sentence, "is_final": False})
                return

            await handle_user_message(deepgram, sentence, source="voice")

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

                    elif data.get("type") == "tts_done":
                        tts_done_event.set()

                    elif data.get("type") == "clear_history":
                        state["messages"] = []
                        state["active_agent"] = "primary"
                        introduced = {"primary": False, "supervisor": False, "escalation": False}
                        first_user_ts = None
                        resolved = False
                        await websocket.send_json({"type": "history_cleared"})

                    # ✅ TYPED MESSAGE SUPPORT
                    elif data.get("type") == "text_message":
                        user_text = (data.get("text") or "").strip()
                        if user_text:
                            # typed messages should also stop mic while responding
                            if not ai_speaking:
                                await handle_user_message(deepgram, user_text, source="typed")

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
