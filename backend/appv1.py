# agentic_callcenter.py

import os
import json
import re
import io
import base64
from typing import TypedDict, Literal, Generator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from murf import Murf, MurfRegion
import whisper

load_dotenv()

# -------------------------
# CONFIG
# -------------------------


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"]
)


# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0,
#     google_api_key=os.environ.get("GOOGLE_API_KEY"),
# )

# Initialize Murf client
murf_client = Murf(
    api_key=os.getenv("MURF_API_KEY"),
    region=MurfRegion.GLOBAL  # Change to MurfRegion.IN for India, etc.
)

CONFIDENCE_THRESHOLD = 0.6

# Load Whisper model for STT
whisper_model = whisper.load_model("base")

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
         "Answer the user's query clearly, politely, and concisely. "
         "Keep responses under 200 words for voice interactions."),
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

    # Safe JSON parsing
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
            "Your issue has been escalated to a human agent. "
            f"{summary.content}"
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

# -------------------------
# STT AND TTS FUNCTIONS
# -------------------------
import pyaudio
from murf import Murf, MurfRegion
# def sanitize_for_tts(text: str, max_length: int = 500) -> str:
#     """Clean AI text for TTS"""
#     text = re.sub(r"[*_#>`~-]", "", text)
#     text = re.sub(r"[⚠️🗣️🤖]", "", text)
#     text = text.encode("ascii", "ignore").decode()
#     text = re.sub(r"\s+", " ", text).strip()
    
#     if len(text) > max_length:
#         text = text[:max_length].rsplit(".", 1)[0] + "."
    
#     return text


def speech_to_text(audio_file_path: str) -> str:
    """Local Whisper STT"""
    try:
        result = whisper_model.transcribe(audio_file_path)
        return result["text"].strip()
    except Exception as e:
        raise Exception(f"Speech-to-text failed: {str(e)}")


def murf_text_to_speech(text: str, output_path: str = "response.wav") -> str:
    """
    Generate speech using Murf SDK (non-streaming)
    Returns path to saved audio file
    """
    try:
        # clean_text = sanitize_for_tts(text)
        clean_text = text
        print(f"🎙️ Generating speech: '{clean_text[:50]}...'")
        
        audio_stream = murf_client.text_to_speech.stream(
            text=clean_text,
            voice_id="Matthew",  # Change to your preferred voice
            model="FALCON",
            multi_native_locale="en-US",
            sample_rate=24000,
            format="WAV"
        )
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
        try:
            print("Starting audio playback...")
            for chunk in audio_stream:
                if chunk:  # Check if chunk has data
                    stream.write(chunk)
        except Exception as e:
            print(f"Error during streaming: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("Audio streaming and playback complete!")
        # Save the audio file
    #     with open(output_path, "wb") as f:
    #         f.write(audio_res.audio_file)
        
    #     print(f"✅ Audio saved to {output_path}")
    #     return output_path
        
    except Exception as e:
        raise Exception(f"Murf TTS failed: {str(e)}")


def murf_stream_to_base64(text: str) -> Generator[str, None, None]:
    """
    Stream audio from Murf as base64 chunks for browser playback
    Yields base64-encoded PCM audio chunks
    """
    try:
        # clean_text = sanitize_for_tts(text)
        clean_text = text
        
        print(f"🎙️ Streaming speech: '{clean_text[:50]}...'")
        
        audio_stream = murf_client.text_to_speech.stream(
            text=clean_text,
            voice_id="en-US-terrell",
            model="FALCON",
            multi_native_locale="en-US",
            sample_rate=24000,
            format="PCM"  # Raw PCM for streaming
        )
        
        for chunk in audio_stream:
            if chunk:
                # Encode chunk to base64 for JSON transmission
                base64_chunk = base64.b64encode(chunk).decode('utf-8')
                yield base64_chunk
                
    except Exception as e:
        raise Exception(f"Murf streaming failed: {str(e)}")


# -------------------------
# PUBLIC FUNCTIONS
# -------------------------

agentic_pipeline = build_agentic_graph()

def handle_customer_interaction(user_input: str) -> str:
    """Process user input through agentic pipeline"""
    result = agentic_pipeline.invoke({
        "user_input": user_input
    })
    return result["final_response"]


def handle_voice_interaction(audio_file_path: str) -> dict:
    """
    End-to-end voice interaction (for testing)
    Audio → STT → Agentic AI → TTS → Audio
    """
    try:
        print("\n" + "="*50)
        print("🎧 Converting speech to text...")
        user_text = speech_to_text(audio_file_path)
        print(f"🗣️ User said: {user_text}")
        
        print("\n🤖 Processing with AI agents...")
        ai_response_text = handle_customer_interaction(user_text)
        print(f"💬 AI response: {ai_response_text}")
        
        print("\n🔊 Converting response to speech...")
        audio_response_path = murf_text_to_speech(ai_response_text)
        print("="*50 + "\n")
        
        return {
            "user_text": user_text,
            "ai_text": ai_response_text,
            "audio_file": audio_response_path
        }
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


# -------------------------
# FASTAPI WEB SERVER FOR BROWSER
# -------------------------

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import asyncio

app = FastAPI(title="AI Call Center API")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/voice/upload")
async def voice_interaction_upload(audio: UploadFile = File(...)):
    """
    Upload audio file, get AI response as audio file
    """
    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process through pipeline
        result = handle_voice_interaction(tmp_path)
        
        # Return audio file
        return FileResponse(
            result["audio_file"],
            media_type="audio/wav",
            headers={
                "X-User-Text": result["user_text"],
                "X-AI-Text": result["ai_text"]
            }
        )
    
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws/voice-stream")
async def voice_stream_websocket(websocket: WebSocket):
    """
    WebSocket for real-time streaming audio
    Client sends audio → Server sends back streaming TTS
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive audio data from browser
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                # Decode base64 audio from browser
                audio_bytes = base64.b64decode(data["audio"])
                
                # Save temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                # STT
                user_text = speech_to_text(tmp_path)
                await websocket.send_json({
                    "type": "transcription",
                    "text": user_text
                })
                
                # AI processing
                ai_response = handle_customer_interaction(user_text)
                await websocket.send_json({
                    "type": "ai_response",
                    "text": ai_response
                })
                
                # Stream TTS back to browser
                await websocket.send_json({"type": "audio_start"})
                
                for audio_chunk in murf_stream_to_base64(ai_response):
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": audio_chunk
                    })
                
                await websocket.send_json({"type": "audio_end"})
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


@app.post("/api/text-to-speech")
async def text_to_speech_endpoint(data: dict):
    """
    Simple TTS endpoint
    """
    text = data.get("text", "")
    output_path = murf_text_to_speech(text)
    return FileResponse(output_path, media_type="audio/wav")


@app.get("/")
async def root():
    return {
        "message": "AI Call Center API",
        "endpoints": {
            "/api/voice/upload": "Upload audio, get AI response",
            "/ws/voice-stream": "WebSocket for real-time streaming",
            "/api/text-to-speech": "Convert text to speech"
        }
    }


# -------------------------
# LOCAL TEST
# -------------------------

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run FastAPI server
        import uvicorn
        print("\n🚀 Starting AI Call Center Server...")
        print("📡 WebSocket: ws://localhost:8000/ws/voice-stream")
        print("📁 Upload API: http://localhost:8000/api/voice/upload")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run local tests
        # print("\n--- TEST 1 (Text Query) ---")
        # response = handle_customer_interaction(
        #     "Can you tell me the status of my order?"
        # )
        # print(response)
        
        # print("\n--- TEST 2 (Escalation) ---")
        # response = handle_customer_interaction(
        #     "I was charged twice and this is unacceptable"
        # )
        # print(response)
        
        # Voice test if file exists
        # if os.path.exists("./output.wav"):
        #     print("\n--- TEST 3 (Voice) ---")
        #     result = handle_voice_interaction("./output.wav")
        #     print(f"User: {result['user_text']}")
        #     print(f"AI: {result['ai_text']}")
        murf_text_to_speech("Hello, how can I help you today?", "fuck.wav")