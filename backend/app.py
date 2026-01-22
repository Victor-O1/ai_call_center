# agentic_callcenter.py

import os
import json
from typing import TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()
import requests



# -------------------------
# CONFIG
# -------------------------

# os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     temperature=0.2,
#     api_key=os.environ["GOOGLE_API_KEY"]
# )
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # free-tier friendly    
    temperature=0,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)

CONFIDENCE_THRESHOLD = 0.6

# -------------------------
# STATE DEFINITION
# -------------------------

class CallCenterState(TypedDict):
    user_input: str
    agent_response: str
    confidence_score: float
    sentiment: Literal["positive", "neutral", "negative"]
    decision: Literal["resolve", "escalate"]
    final_response: str

# -------------------------
# PRIMARY AGENT
# -------------------------

def primary_agent(state: CallCenterState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support AI agent for a digital call center. "
         "Answer the user's query clearly, politely, and concisely."),
        ("human", "{input}")
    ])

    response = llm.invoke(
        prompt.format_messages(input=state["user_input"])
    )

    return {
        "agent_response": response.content
    }

# -------------------------
# SUPERVISOR AGENT
# -------------------------

def supervisor_agent(state: CallCenterState):
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a supervisor AI overseeing a call center.\n\n"
         "Evaluate the AI response and return STRICT JSON ONLY:\n"
         "{{\n"
         '  "confidence_score": number between 0 and 1,\n'
         '  "sentiment": "positive" | "neutral" | "negative",\n'
         '  "decision": "resolve" | "escalate"\n'
         "}}\n\n"
         "Rules:\n"
         "- Escalate if confidence_score < 0.6\n"
         "- Escalate if sentiment is negative\n"),
        ("human",
         "User Query:\n{user_input}\n\n"
         "AI Response:\n{agent_response}")
    ])

    judgment = llm.invoke(
        judge_prompt.format_messages(
            user_input=state["user_input"],
            agent_response=state["agent_response"]
        )
    )

    # ---- SAFE JSON PARSING ----
    import re, json
    match = re.search(r"\{.*\}", judgment.content, re.DOTALL)
    parsed = json.loads(match.group())

    return {
        "confidence_score": parsed["confidence_score"],
        "sentiment": parsed["sentiment"],
        "decision": parsed["decision"]
    }


# -------------------------
# ESCALATION AGENT
# -------------------------

def escalation_agent(state: CallCenterState):
    escalation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an escalation AI. Summarize the issue clearly "
         "for a human support agent. Be brief and structured."),
        ("human",
         "User Query:\n{user_input}\n\n"
         "AI Response:\n{agent_response}")
    ])

    summary = llm.invoke(
        escalation_prompt.format_messages(
            user_input=state["user_input"],
            agent_response=state["agent_response"]
        )
    )

    return {
        "final_response": (
            "⚠️ Your issue has been escalated to a human agent.\n\n"
            f"Summary for human agent:\n{summary.content}"
        )
    }

# -------------------------
# RESOLUTION PATH
# -------------------------

def resolve(state: CallCenterState):
    return {
        "final_response": state["agent_response"]
    }

# -------------------------
# ROUTER
# -------------------------

def route_decision(state: CallCenterState):
    if state["decision"] == "escalate":
        return "escalate"
    return "resolve"

# -------------------------
# LANGGRAPH PIPELINE
# -------------------------

def build_agentic_graph():
    graph = StateGraph(CallCenterState)

    graph.add_node("primary_agent", primary_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("escalation_agent", escalation_agent)
    graph.add_node("resolve", resolve)

    graph.set_entry_point("primary_agent")

    graph.add_edge("primary_agent", "supervisor_agent")

    graph.add_conditional_edges(
        "supervisor_agent",
        route_decision,
        {
            "resolve": "resolve",
            "escalate": "escalation_agent"
        }
    )

    graph.add_edge("resolve", END)
    graph.add_edge("escalation_agent", END)

    return graph.compile()



# ------------------------
# STT and TTS
# ------------------------
import requests

MURF_API_KEY = os.getenv("MURF_API_KEY")

import whisper

whisper_model = whisper.load_model("base")
import re

def sanitize_for_tts(text: str, max_length: int = 500) -> str:
    """
    Cleans AI text for TTS engines like Murf
    """
    # Remove markdown
    text = re.sub(r"[*_#>`~-]", "", text)

    # Remove emojis & non-ASCII
    text = text.encode("ascii", "ignore").decode()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + "."

    return text

def speech_to_text(audio_file_path: str) -> str:
    """
    Local Whisper STT (stable, free, offline)
    """
    result = whisper_model.transcribe(audio_file_path)
    return result["text"]



def murf_text_to_speech(text: str, output_path="response.wav"):
    url = "https://api.murf.ai/v1/speech/generate"

    clean_text = sanitize_for_tts(text)

    headers = {
        "Authorization": f"Bearer {MURF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "voiceId": "en-US-natalie",
        "text": clean_text,
        "format": "wav",
        "sampleRate": 44100
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()

    # Murf returns a URL, not raw audio
    audio_url = data.get("audioFile") or data.get("audio_url")
    if not audio_url:
        raise ValueError(f"Unexpected Murf response: {data}")

    audio_data = requests.get(audio_url).content
    with open(output_path, "wb") as f:
        f.write(audio_data)

    return output_path


def handle_voice_interaction(audio_file_path: str):
    """
    End-to-end voice interaction:
    Audio → STT → Agentic AI → TTS → Audio
    """

    # 1. Speech to Text
    user_text = speech_to_text(audio_file_path)
    print("🗣️ User said:", user_text)

    # 2. Agentic Pipeline
    ai_response_text = handle_customer_interaction(user_text)
    print("🤖 AI response:", ai_response_text)

    # 3. Text to Speech
    audio_response_path = murf_text_to_speech(ai_response_text)

    return {
        "user_text": user_text,
        "ai_text": ai_response_text,
        "audio_file": audio_response_path
    }



# -------------------------
# PUBLIC FUNCTION (API-READY)
# -------------------------

agentic_pipeline = build_agentic_graph()

def handle_customer_interaction(user_input: str):
    result = agentic_pipeline.invoke({
        "user_input": user_input
    })
    return result["final_response"]

# -------------------------
# LOCAL TEST
# -------------------------

if __name__ == "__main__":
    # print("\n--- TEST 1 (Normal) ---")
    # print(handle_customer_interaction(
    #     "Can you tell me the status of my order?"
    # ))

    # print("\n--- TEST 2 (Escalation) ---")
    # print(handle_customer_interaction(
    #     "I was charged twice and this is unacceptable"
    # ))
    result = handle_voice_interaction("./output.wav")
    print(result)
