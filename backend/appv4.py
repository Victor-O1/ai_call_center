"""
Voice AI Backend with Deepgram STT, Groq LLM, and Deepgram TTS
Install: pip install fastapi uvicorn websockets deepgram-sdk groq python-dotenv httpx
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

print(f"Deepgram API Key: {'Set' if DEEPGRAM_API_KEY else 'NOT SET'}")
print(f"Groq API Key: {'Set' if GROQ_API_KEY else 'NOT SET'}")

if not DEEPGRAM_API_KEY or not GROQ_API_KEY:
    print("WARNING: API keys not set! Please configure .env file")

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
    
    try:
        # Initialize Deepgram client
        config = DeepgramClientOptions(
            options={"keepalive": "true"}
        )
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        
        # Create connection
        dg_connection = deepgram.listen.live.v("1")
        
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Message handler - must be sync, not async
        def on_message(self, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                
                if len(sentence) == 0:
                    return
                
                print(f"Transcript: {sentence} (final: {result.is_final})")
                
                # Schedule async tasks in the event loop
                asyncio.run_coroutine_threadsafe(
                    handle_transcript(sentence, result.is_final),
                    loop
                )
                    
            except Exception as e:
                print(f"Error in on_message: {e}")
        
        # Async handler for transcript processing
        async def handle_transcript(sentence, is_final):
            try:
                if is_final:
                    # Send final transcript
                    await websocket.send_json({
                        "type": "transcript",
                        "text": sentence,
                        "is_final": True
                    })
                    
                    # Add to conversation
                    conversation.add_message("user", sentence)
                    
                    # Signal AI is thinking
                    await websocket.send_json({"type": "ai_thinking"})
                    
                    # Get AI response from Groq
                    try:
                        messages = conversation.get_messages()
                        print(f"Sending to Groq: {messages}")
                        
                        chat_completion = groq_client.chat.completions.create(
                            messages=messages,
                            model="llama-3.3-70b-versatile",
                            temperature=0.7,
                            max_tokens=1024,
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in chat_completion:
                            if chunk.choices[0].delta.content:
                                content = chunk.choices[0].delta.content
                                full_response += content
                                await websocket.send_json({
                                    "type": "ai_response_chunk",
                                    "text": content
                                })
                        
                        print(f"AI Response: {full_response}")
                        conversation.add_message("assistant", full_response)
                        
                        # Generate TTS
                        await websocket.send_json({"type": "generating_audio"})
                        
                        try:
                            # Use Deepgram TTS
                            speak_options = SpeakOptions(
                                model="aura-asteria-en",
                                encoding="linear16",
                                sample_rate=24000
                            )
                            
                            response = deepgram.speak.v("1").save(
                                "audio_output.wav",
                                {"text": full_response},
                                speak_options
                            )
                            
                            # Read and send audio file
                            with open("audio_output.wav", "rb") as audio_file:
                                audio_data = audio_file.read()
                                # Send in chunks
                                chunk_size = 4096
                                for i in range(0, len(audio_data), chunk_size):
                                    chunk = audio_data[i:i+chunk_size]
                                    audio_b64 = base64.b64encode(chunk).decode('utf-8')
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "data": audio_b64
                                    })
                            
                            # Clean up
                            if os.path.exists("audio_output.wav"):
                                os.remove("audio_output.wav")
                            
                            await websocket.send_json({"type": "audio_complete"})
                            
                        except Exception as e:
                            print(f"TTS Error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "message": f"TTS failed: {str(e)}"
                            })
                    
                    except Exception as e:
                        print(f"Groq Error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"AI processing failed: {str(e)}"
                        })
                
                else:
                    # Interim results
                    await websocket.send_json({
                        "type": "transcript",
                        "text": sentence,
                        "is_final": False
                    })
            except Exception as e:
                print(f"Error handling transcript: {e}")
        
        # Error handler - must be sync
        def on_error(self, error, **kwargs):
            print(f"Deepgram error: {error}")
            asyncio.run_coroutine_threadsafe(
                websocket.send_json({
                    "type": "error",
                    "message": f"STT error: {str(error)}"
                }),
                loop
            )
        
        # Register handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        
        # Configure options
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
        
        # Start connection
        if dg_connection.start(options):
            print("Deepgram connection started successfully")
            await websocket.send_json({"type": "ready"})
        else:
            raise Exception("Failed to start Deepgram connection")
        
        # Handle incoming messages
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                
                if "bytes" in message:
                    # Audio data
                    audio_data = message["bytes"]
                    dg_connection.send(audio_data)
                    
                elif "text" in message:
                    data = json.loads(message["text"])
                    
                    if data.get("type") == "stop":
                        print("Stop requested")
                        break
                    elif data.get("type") == "clear_history":
                        conversation.history = []
                        await websocket.send_json({"type": "history_cleared"})
                        
            except asyncio.TimeoutError:
                # Keep connection alive
                continue
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except Exception as e:
                print(f"Error processing message: {e}")
                break
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    
    finally:
        print("Cleaning up connection")
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
        "endpoints": {
            "websocket": "/ws/voice"
        },
        "config": {
            "deepgram_configured": bool(DEEPGRAM_API_KEY),
            "groq_configured": bool(GROQ_API_KEY)
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Starting Voice AI Backend...")
    print(f"Server will be available at: http://localhost:8000")
    print(f"WebSocket endpoint: ws://localhost:8000/ws/voice")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")