# AI Call Center – Realtime Multi-Agent Voice Intelligence

A **real-time AI call center system** that supports **live voice conversations**, **multi-agent routing**, **emotion + intent analysis**, and **text-to-speech responses**, built using **FastAPI, WebSockets, Deepgram, and Groq LLMs**.

This project simulates a modern AI-powered call center with **primary, supervisor, and escalation agents**, complete with analytics such as confidence score, CSAT prediction, resolution time, and emotion detection.

---

## ✨ Key Features

### 🎙️ Realtime Voice Conversation
- Live microphone streaming from browser → backend
- Raw audio streamed via WebSockets
- Speech-to-Text using **Deepgram Nova-3**
- Supports **barge-in** (user can interrupt AI while it’s speaking)

### 🧠 Multi-Agent AI Routing
AI dynamically selects one of three agents:
- **Primary Agent** – General support
- **Supervisor Agent** – Complaints, angry users, policy confusion
- **Escalation Agent** – Billing, refunds, security, cancellations

Agent selection is decided **per message** using an LLM-based router.

### 🤖 LLM Intelligence (Groq)
- Uses **LLaMA-3.3-70B** via Groq API
- Strict JSON contract enforced with schema validation
- Automatic **LLM output repair** if invalid JSON is returned
- Maintains rolling conversation memory

### 🔊 Text-to-Speech (TTS)
- AI replies are converted to speech using **Deepgram Speak**
- Streams MP3 audio chunks to frontend
- Frontend buffers and plays AI audio smoothly

### 📊 Live Analytics Dashboard
Displayed in real time:
- Active agent
- Detected intent
- Emotion classification
- Confidence score (0–100)
- CSAT prediction (1–5)
- Resolution time (seconds)
- Resolution status (resolved / unresolved)

### 💬 Typed + Voice Input
- Speak naturally via microphone
- Or send text messages
- Both go through the same AI pipeline

---

## 🧩 Architecture Overview

Browser (React)
│ WebSocket (audio + JSON)


FastAPI Backend
├─ Deepgram STT (WebSocket)
├─ Groq LLM (Intent + Agent Routing)
├─ JSON Validation + Repair
└─ Deepgram TTS (MP3 streaming)


---

## 🛠️ Tech Stack

### Backend
- **FastAPI**
- **WebSockets**
- **Groq LLM API**
- **Deepgram STT & TTS**
- **Pydantic** (strict schema validation)
- **Uvicorn**

### Frontend
- **React (Next.js compatible)**
- **Web Audio API**
- **WebSockets**
- **Tailwind CSS**
- **Lucide Icons**

---

## ⚙️ Environment Variables

Create a `.env` file in backend:

```env
DEEPGRAM_API_KEY=your_deepgram_key
GROQ_API_KEY=your_groq_key
```
