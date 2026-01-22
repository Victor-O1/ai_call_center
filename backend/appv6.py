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
from dotenv import load_dotenv
import base64
from groq import Groq

# ✅ LangGraph + LangChain
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import List, Dict, Optional

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
# ✅ AGENT CONFIG
# ==============================
AGENTS = {
    "primary": {
        "name": "Ava (Primary Agent)",
        "voice": "aura-asteria-en",
        "system": """You are Ava, the Primary AI Call Center Agent.
You greet the customer warmly, gather details, and try to resolve quickly.
Keep responses short and friendly for voice. Ask 1 question at a time.
If the user is angry, confused, asks policy/legal/complaint OR wants escalation, route to Supervisor.""",
        "intro": "Hey! I’m Ava, your primary assistant. Tell me what you need help with.",
    },
    "supervisor": {
        "name": "Noah (Supervisor Agent)",
        "voice": "aura-luna-en",
        "system": """You are Noah, the Supervisor AI Agent.
You handle angry customers, complaint handling, policy clarifications, and decision making.
Be calm, confident, and structured. Confirm the issue, provide resolution steps.
If the user requests a human, payment disputes, cancellations, or account-sensitive issues → route to Escalation.""",
        "intro": "Hi, I’m Noah, the supervisor agent. I’ll help you sort this out calmly and quickly.",
    },
    "escalation": {
        "name": "Maya (Escalation Agent)",
        "voice": "aura-hera-en",
        "system": """You are Maya, the Escalation AI Agent.
Your job is to either: (1) prepare a clean escalation summary for a human agent,
or (2) handle high priority / account / dispute issues carefully.
Be brief, professional, and ask for only required details.
If the user wants a human, confirm and provide an escalation summary.""",
        "intro": "Hello, I’m Maya from escalations. I’ll take care of this with priority.",
    },
}

# ==============================
# ✅ LangGraph State
# ==============================
class GraphState(TypedDict):
    messages: List[Dict[str, str]]        # chat history
    active_agent: str                     # primary/supervisor/escalation
    user_text: str                        # latest user text
    agent_reply: str                      # final reply text
    agent_switched: bool                  # did we switch agent?
    switch_to: Optional[str]              # who to switch to


# ==============================
# ✅ Helper: Groq call
# ==============================
def groq_chat(messages: List[Dict[str, str]]) -> str:
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.6,
        max_tokens=512,
    )
    return completion.choices[0].message.content.strip()


# ==============================
# ✅ Supervisor router node
# ==============================
def supervisor_router(state: GraphState) -> GraphState:
    """
    This decides which agent should handle the user message.
    """
    user_text = state["user_text"].lower()

    # ✅ quick deterministic rules for speed + reliability
    if any(k in user_text for k in ["human", "agent", "representative", "manager", "complaint"]):
        state["switch_to"] = "supervisor"
    if any(k in user_text for k in ["refund", "payment", "charged", "billing", "cancel", "account", "escalate"]):
        state["switch_to"] = "escalation"

    # If not decided, ask Groq to route
    if not state.get("switch_to"):
        routing_prompt = [
            {
                "role": "system",
                "content": """You are a routing brain for a call center.
Choose one of these: primary, supervisor, escalation.
Return ONLY the agent key, nothing else."""
            },
            {"role": "user", "content": f"User message: {state['user_text']}"}
        ]
        decision = groq_chat(routing_prompt).strip().lower()
        if "supervisor" in decision:
            state["switch_to"] = "supervisor"
        elif "escalation" in decision:
            state["switch_to"] = "escalation"
        else:
            state["switch_to"] = "primary"

    # Check switch
    state["agent_switched"] = (state["switch_to"] != state["active_agent"])
    state["active_agent"] = state["switch_to"]

    return state


# ==============================
# ✅ Agent nodes
# ==============================
def agent_node(agent_key: str):
    def _node(state: GraphState) -> GraphState:
        agent_cfg = AGENTS[agent_key]

        # Build messages
        agent_messages = [
            {"role": "system", "content": agent_cfg["system"]},
        ]

        # Include last conversation
        agent_messages += state["messages"][-10:]
        agent_messages.append({"role": "user", "content": state["user_text"]})

        reply = groq_chat(agent_messages)

        state["agent_reply"] = reply
        state["messages"].append({"role": "user", "content": state["user_text"]})
        state["messages"].append({"role": "assistant", "content": reply})

        return state

    return _node


# ==============================
# ✅ Build LangGraph
# ==============================
graph = StateGraph(GraphState)

graph.add_node("router", supervisor_router)
graph.add_node("primary", agent_node("primary"))
graph.add_node("supervisor", agent_node("supervisor"))
graph.add_node("escalation", agent_node("escalation"))

graph.set_entry_point("router")

# route → chosen agent → END
graph.add_conditional_edges(
    "router",
    lambda s: s["active_agent"],
    {
        "primary": "primary",
        "supervisor": "supervisor",
        "escalation": "escalation",
    },
)
graph.add_edge("primary", END)
graph.add_edge("supervisor", END)
graph.add_edge("escalation", END)

LANGGRAPH_APP = graph.compile()


# ==============================
# ✅ WebSocket server
# ==============================
@app.websocket("/ws/voice")
async def voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")

    dg_connection = None
    ai_speaking = False
    tts_done_event = asyncio.Event()

    # conversation state for graph
    state: GraphState = {
        "messages": [],
        "active_agent": "primary",
        "user_text": "",
        "agent_reply": "",
        "agent_switched": False,
        "switch_to": None,
    }

    # track if we already sent intro for agent
    introduced = {"primary": False, "supervisor": False, "escalation": False}

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
                    handle_transcript(sentence, result.is_final),
                    loop,
                )
            except Exception as e:
                print("on_message error:", e)

        async def speak_tts(text: str, voice_model: str):
            """
            Sends audio chunks → waits for frontend tts_done event.
            """
            nonlocal ai_speaking
            ai_speaking = True
            tts_done_event.clear()

            speak_options = SpeakOptions(
                model=voice_model,
                encoding="linear16",
                sample_rate=24000,
            )

            deepgram.speak.v("1").save(
                "audio_output.wav",
                {"text": text},
                speak_options,
            )

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

            # ✅ wait until playback truly ends
            await tts_done_event.wait()
            ai_speaking = False

        async def handle_transcript(sentence, is_final):
            nonlocal ai_speaking, state

            if ai_speaking:
                return

            # interim
            if not is_final:
                await websocket.send_json({"type": "transcript", "text": sentence, "is_final": False})
                return

            # final user transcript
            await websocket.send_json({"type": "transcript", "text": sentence, "is_final": True})
            await websocket.send_json({"type": "ai_thinking"})

            # ✅ run langgraph
            state["user_text"] = sentence
            state["switch_to"] = None

            new_state = LANGGRAPH_APP.invoke(state)
            state = new_state

            active = state["active_agent"]
            agent_cfg = AGENTS[active]

            # ✅ if switched, notify frontend + say intro ONCE
            if state["agent_switched"]:
                await websocket.send_json({
                    "type": "agent_switch",
                    "agent_key": active,
                    "agent_name": agent_cfg["name"],
                })

                # intro only first time
                if not introduced[active]:
                    introduced[active] = True
                    await websocket.send_json({
                        "type": "ai_response_chunk",
                        "agent_key": active,
                        "agent_name": agent_cfg["name"],
                        "text": agent_cfg["intro"] + " "
                    })
                    await speak_tts(agent_cfg["intro"], agent_cfg["voice"])

            # ✅ send final assistant text to chat UI
            reply = state["agent_reply"]

            await websocket.send_json({
                "type": "ai_response_full",
                "agent_key": active,
                "agent_name": agent_cfg["name"],
                "text": reply
            })

            # ✅ speak with agent voice
            await speak_tts(reply, agent_cfg["voice"])

            # backend says listening
            await websocket.send_json({"type": "listening"})

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
            raise Exception("Failed to start Deepgram")

        # receive loop
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
                        state["messages"] = []
                        state["active_agent"] = "primary"
                        introduced = {"primary": False, "supervisor": False, "escalation": False}
                        await websocket.send_json({"type": "history_cleared"})

                    elif data.get("type") == "tts_done":
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
