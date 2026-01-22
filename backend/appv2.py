import os
import json
import base64
import asyncio
from typing import TypedDict, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from murf import Murf, MurfRegion
import whisper
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import wave
import io
import subprocess
import time
from datetime import datetime

load_dotenv()

# -------------------------
# CONFIG
# -------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ["GROQ_API_KEY"]
)

murf_client = Murf(
    api_key=os.getenv("MURF_API_KEY"),
    region=MurfRegion.GLOBAL
)

whisper_model = whisper.load_model("tiny")

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}", flush=True)

# -------------------------
# STATE
# -------------------------
class CallCenterState(TypedDict):
    user_input: str
    agent_response: str
    confidence_score: float
    sentiment: Literal["positive", "neutral", "negative"]
    decision: Literal["resolve", "escalate"]
    final_response: str

# -------------------------
# AGENTS (FAST)
# -------------------------
def primary_agent(state: CallCenterState):
    log(f"🧠 LLM: Processing '{state['user_input']}'")
    start = time.time()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Keep responses under 30 words."),
        ("human", "{input}")
    ])
    response = llm.invoke(prompt.format_messages(input=state["user_input"]))
    
    log(f"🧠 LLM: Responded in {time.time()-start:.2f}s: '{response.content}'")
    return {"agent_response": response.content}

def supervisor_agent(state: CallCenterState):
    return {
        "confidence_score": 0.9,
        "sentiment": "neutral",
        "decision": "resolve"
    }

def resolve(state: CallCenterState):
    return {"final_response": state["agent_response"]}

def route_decision(state: CallCenterState):
    return "resolve"

# -------------------------
# GRAPH
# -------------------------
def build_graph():
    graph = StateGraph(CallCenterState)
    graph.add_node("primary_agent", primary_agent)
    graph.add_node("supervisor_agent", supervisor_agent)
    graph.add_node("resolve", resolve)
    graph.set_entry_point("primary_agent")
    graph.add_edge("primary_agent", "supervisor_agent")
    graph.add_conditional_edges("supervisor_agent", route_decision, {"resolve": "resolve"})
    graph.add_edge("resolve", END)
    return graph.compile()

pipeline = build_graph()

# -------------------------
# AUDIO FUNCTIONS
# -------------------------
def convert_webm_to_wav(webm_data: bytes) -> bytes:
    """Convert WebM to WAV using FFmpeg"""
    try:
        log(f"🔄 Converting {len(webm_data)} bytes WebM to WAV...")
        
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(webm_data)
            webm_path = f.name
        
        wav_path = webm_path.replace('.webm', '.wav')
        
        result = subprocess.run([
            'ffmpeg', '-i', webm_path,
            '-ar', '16000', '-ac', '1', '-y', wav_path
        ], capture_output=True, timeout=5)
        
        if result.returncode != 0:
            log(f"❌ FFmpeg failed: {result.stderr.decode()[:200]}")
            os.unlink(webm_path)
            return None
        
        with open(wav_path, 'rb') as f:
            wav_data = f.read()
        
        os.unlink(webm_path)
        os.unlink(wav_path)
        
        log(f"✅ Converted to {len(wav_data)} bytes WAV")
        return wav_data
        
    except Exception as e:
        log(f"❌ Conversion error: {e}")
        return None

def transcribe_audio(wav_bytes: bytes) -> str:
    """Transcribe audio using Whisper"""
    try:
        log(f"🎙️ Transcribing {len(wav_bytes)} bytes...")
        start = time.time()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            path = f.name
        
        result = whisper_model.transcribe(path, fp16=False, language="en")
        os.unlink(path)
        
        text = result["text"].strip()
        log(f"🎙️ Transcribed in {time.time()-start:.2f}s: '{text}'")
        return text
        
    except Exception as e:
        log(f"❌ Transcription error: {e}")
        return ""

def pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """Convert PCM to WAV"""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()

# -------------------------
# FASTAPI
# -------------------------
app = FastAPI(title="AI Call Center")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# WEBSOCKET
# -------------------------
@app.websocket("/ws/realtime-call")
async def realtime_call(websocket: WebSocket):
    await websocket.accept()
    log("\n" + "="*80)
    log("🎧 CALL STARTED")
    log("="*80)
    
    audio_buffer = []
    last_audio_time = time.time()
    is_speaking = False
    call_active = True
    chunk_count = 0
    
    await websocket.send_json({"type": "connected", "message": "Ready"})
    log("✅ WebSocket connected")
    
    async def process_audio():
        print("Processing audio...")
        nonlocal audio_buffer, last_audio_time, is_speaking, call_active
        
        while call_active:
            try:
                await asyncio.sleep(0.3)
                
                current_time = time.time()
                silence = current_time - last_audio_time
                
                # Process if we have audio and silence detected
                if len(audio_buffer) >= 3 and silence > 0.8 and not is_speaking:
                    log(f"\n{'='*80}")
                    log(f"🎤 PROCESSING: {len(audio_buffer)} chunks, {silence:.1f}s silence")
                    
                    # Combine audio
                    combined = b''.join(audio_buffer)
                    audio_buffer.clear()
                    log(f"📦 Combined: {len(combined)} bytes")
                    
                    # Convert
                    wav = convert_webm_to_wav(combined)
                    if not wav:
                        log("❌ Conversion failed")
                        continue
                    
                    # Transcribe
                    text = transcribe_audio(wav)
                    if not text or len(text) < 3:
                        log(f"⚠️ No speech: '{text}'")
                        continue
                    
                    log(f"👤 USER: '{text}'")
                    
                    # Send to frontend
                    await websocket.send_json({"type": "transcription", "text": text})
                    
                    # Get AI response
                    is_speaking = True
                    log("🤖 Getting AI response...")
                    
                    result = pipeline.invoke({"user_input": text})
                    ai_text = result["final_response"]
                    
                    log(f"🤖 AI: '{ai_text}'")
                    
                    await websocket.send_json({
                        "type": "ai_response",
                        "text": ai_text,
                        "sentiment": "neutral",
                        "confidence": 0.9,
                        "decision": "resolve"
                    })
                    
                    # TTS
                    await websocket.send_json({"type": "ai_speaking_start"})
                    log("🔊 Generating TTS...")
                    
                    try:
                        stream = murf_client.text_to_speech.stream(
                            text=ai_text,
                            voice_id="en-US-matthew",
                            model="FALCON",
                            sample_rate=24000,
                            format="PCM"
                        )
                        
                        count = 0
                        for pcm in stream:
                            if not call_active:
                                break
                            if pcm:
                                wav_chunk = pcm_to_wav(pcm, 24000)
                                b64 = base64.b64encode(wav_chunk).decode()
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "data": b64
                                })
                                count += 1
                        
                        log(f"🔊 Sent {count} TTS chunks")
                        
                    except Exception as e:
                        log(f"❌ TTS error: {e}")
                    
                    is_speaking = False
                    await websocket.send_json({"type": "ai_speaking_end"})
                    log("✅ Done\n")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                log(f"❌ Processing error: {e}")
                import traceback
                traceback.print_exc()
    
    # Start processor
    task = asyncio.create_task(process_audio())
    
    try:
        # Main loop - receive audio
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio":
                chunk_count += 1
                
                # Decode and store
                audio_data = base64.b64decode(data["data"])
                audio_buffer.append(audio_data)
                last_audio_time = time.time()
                
                if chunk_count % 30 == 0:
                    log(f"📥 Received {chunk_count} chunks (buffer: {len(audio_buffer)})")
            
            elif data.get("type") == "end_call":
                log("📞 End call received")
                break
    
    except WebSocketDisconnect:
        log("📞 Disconnected")
    except Exception as e:
        log(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        call_active = False
        task.cancel()
        try:
            await task
        except:
            pass
        
        log("="*80)
        log("🔚 CALL ENDED")
        log("="*80 + "\n")

@app.get("/")
async def root():
    return {"status": "ready"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    log("\n🚀 STARTING AI CALL CENTER")
    log("📡 ws://localhost:8000/ws/realtime-call")
    log("💡 Ensure FFmpeg is installed!\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")