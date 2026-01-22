"use client";
import React, { useState, useRef, useEffect } from "react";
import {
  Trash2,
  Send,
  Mic,
  Square,
  Activity,
  Zap,
  TrendingUp,
  Clock,
} from "lucide-react";

export default function VoiceAIAssistant() {
  const WS_URL = "ws://localhost:8003/ws/voice";

  // connection + status
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState({
    message: "Click to start conversation",
    type: "ready",
  });

  // chat
  const [messages, setMessages] = useState<any[]>([]);
  const [interimMessage, setInterimMessage] = useState<any>(null);

  // input
  const [typedText, setTypedText] = useState("");

  // voice visualization
  const [audioLevel, setAudioLevel] = useState(0);

  // debug + thinking
  const [debugLog, setDebugLog] = useState("Ready");
  const [isThinking, setIsThinking] = useState(false);

  // agent + analytics dashboard
  const [activeAgent, setActiveAgent] = useState("Ava (Primary Agent)");
  const [intent, setIntent] = useState("-");
  const [emotion, setEmotion] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [csat, setCsat] = useState(3);
  const [summary, setSummary] = useState("-");
  const [resolutionTime, setResolutionTime] = useState(0);
  const [resolved, setResolved] = useState(false);

  // refs
  const websocketRef = useRef<WebSocket | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const conversationBoxRef = useRef<HTMLDivElement | null>(null);

  // ✅ mic sending gate
  const isSendingAudioRef = useRef(true);

  // ============================================================
  // ✅ MP3 AUDIO BUFFER (backend sends mp3 chunks)
  // ============================================================
  const mp3ChunksRef = useRef<BlobPart[]>([]);
  const aiAudioRef = useRef<HTMLAudioElement | null>(null);
  const isAIAudioActiveRef = useRef(false);

  const log = (message: string) => {
    console.log(message);
    setDebugLog(message);
  };

  const updateStatus = (message: string, type: string) => {
    setStatus({ message, type });
  };

  const scrollToBottom = () => {
    if (conversationBoxRef.current) {
      conversationBoxRef.current.scrollTop =
        conversationBoxRef.current.scrollHeight;
    }
  };

  const addMessage = (
    role: string,
    text: string,
    agent_name: string | null = null,
  ) => {
    setMessages((prev) => [
      ...prev,
      { role, text, agent_name, id: Date.now() + Math.random() },
    ]);
    setInterimMessage(null);
    setTimeout(scrollToBottom, 50);
  };

  // mic visualization
  const visualizeAudio = () => {
    if (!isConnected || !analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

    setAudioLevel(Math.min(100, avg * 1.4));
    animationFrameRef.current = requestAnimationFrame(visualizeAudio);
  };

  // ============================================================
  // ✅ MP3 decode helpers
  // ============================================================
  const pushMp3Chunk = (b64Data: string) => {
    try {
      const binary = atob(b64Data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }

      mp3ChunksRef.current.push(bytes);

      // mark AI speaking + lock mic
      if (!isAIAudioActiveRef.current) {
        isAIAudioActiveRef.current = true;
        isSendingAudioRef.current = false;
        updateStatus("AI speaking...", "speaking");
        setIsThinking(false);
      }
    } catch (e) {
      console.error("MP3 chunk decode error:", e);
    }
  };

  const playBufferedMp3 = async () => {
    try {
      const chunks = mp3ChunksRef.current;
      if (!chunks.length) {
        console.warn("No MP3 chunks to play");
        isAIAudioActiveRef.current = false;
        isSendingAudioRef.current = true;
        updateStatus("Listening...", "listening");
        return;
      }

      const blob = new Blob(chunks, { type: "audio/mpeg" });
      mp3ChunksRef.current = [];

      const url = URL.createObjectURL(blob);

      // stop previous audio
      if (aiAudioRef.current) {
        try {
          aiAudioRef.current.pause();
        } catch {}
        aiAudioRef.current = null;
      }

      const audio = new Audio(url);
      aiAudioRef.current = audio;

      audio.onended = () => {
        URL.revokeObjectURL(url);
        aiAudioRef.current = null;

        isAIAudioActiveRef.current = false;
        isSendingAudioRef.current = true;
        updateStatus("Listening...", "listening");
      };

      audio.onerror = (err) => {
        console.error("Audio playback error:", err);
        URL.revokeObjectURL(url);
        aiAudioRef.current = null;

        isAIAudioActiveRef.current = false;
        isSendingAudioRef.current = true;
        updateStatus("Listening...", "listening");
      };

      await audio.play();
    } catch (e) {
      console.error("playBufferedMp3() failed:", e);
      isAIAudioActiveRef.current = false;
      isSendingAudioRef.current = true;
      updateStatus("Listening...", "listening");
    }
  };

  // handle websocket events
  const handleMessage = async (data: any) => {
    // debug
    // console.log("WS JSON:", data.type, data);

    switch (data.type) {
      case "ready":
        log("Backend ready ✅");
        break;

      case "transcript":
        if (data.is_final) {
          addMessage("user", data.text);
        } else {
          setInterimMessage({ role: "user", text: data.text });
        }
        break;

      case "typed_message_ack":
        addMessage("user", data.text);
        break;

      case "ai_thinking":
        updateStatus("AI thinking...", "thinking");
        setIsThinking(true);
        isSendingAudioRef.current = false;
        break;

      case "agent_switch":
        setActiveAgent(data.agent_name);
        addMessage(
          "system",
          `🔁 Switched to ${data.agent_name}`,
          data.agent_name,
        );
        break;

      case "ai_message":
        setActiveAgent(data.agent_name || activeAgent);
        addMessage("assistant", data.text, data.agent_name);
        break;

      case "ai_packet":
        setActiveAgent(data.agent_name || activeAgent);

        setIntent(data.intent ?? "-");
        setEmotion(data.emotion ?? "-");
        setConfidence(data.confidence ?? 0);
        setCsat(data.csat_prediction ?? 3);
        setSummary(data.summary ?? "-");

        setResolutionTime(data.resolution_time_sec ?? 0);
        setResolved(Boolean(data.resolved));

        addMessage("assistant", data.reply, data.agent_name);
        break;

      case "audio_chunk":
        // ✅ backend sends MP3 chunks now
        pushMp3Chunk(data.data);
        break;

      case "audio_complete":
        // ✅ play once we have full mp3
        await playBufferedMp3();
        break;

      case "listening":
        updateStatus("Listening...", "listening");
        setIsThinking(false);
        isSendingAudioRef.current = true;
        break;

      case "history_cleared":
        setMessages([]);
        setInterimMessage(null);

        setActiveAgent("Ava (Primary Agent)");
        setIntent("-");
        setEmotion("-");
        setConfidence(0);
        setCsat(3);
        setSummary("-");
        setResolutionTime(0);
        setResolved(false);

        updateStatus("Listening... Speak now!", "listening");
        log("History cleared ✅");
        break;

      case "error":
        log("Error: " + data.message);
        updateStatus("Error: " + data.message, "error");
        setIsThinking(false);
        isSendingAudioRef.current = true;
        break;

      default:
        break;
    }
  };

  // connect + audio pipeline
  const connect = async () => {
    try {
      log("Connecting...");
      updateStatus("Connecting...", "connecting");

      const ws = new WebSocket(WS_URL);
      websocketRef.current = ws;

      ws.onopen = async () => {
        log("WebSocket connected ✅");

        try {
          const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              channelCount: 1,
              sampleRate: 16000,
            },
          });

          mediaStreamRef.current = stream;

          const audioCtx = new AudioContext({ sampleRate: 16000 });
          audioContextRef.current = audioCtx;

          const source = audioCtx.createMediaStreamSource(stream);

          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 256;
          analyserRef.current = analyser;
          source.connect(analyser);

          visualizeAudio();

          const processor = audioCtx.createScriptProcessor(4096, 1, 1);
          processorRef.current = processor;

          processor.onaudioprocess = (e) => {
            if (ws.readyState === WebSocket.OPEN && isSendingAudioRef.current) {
              const input = e.inputBuffer.getChannelData(0);
              const int16 = new Int16Array(input.length);

              for (let i = 0; i < input.length; i++) {
                const s = Math.max(-1, Math.min(1, input[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
              }

              ws.send(int16.buffer);
            }
          };

          source.connect(processor);
          processor.connect(audioCtx.destination);

          setIsConnected(true);
          isSendingAudioRef.current = true;
          updateStatus("Listening... Speak now!", "listening");

          log("Mic ready ✅");
        } catch (err: any) {
          console.error("Mic error:", err);
          alert("Microphone access failed: " + err.message);
          disconnect();
        }
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
      };

      ws.onerror = () => {
        log("WebSocket error ❌");
      };

      ws.onclose = () => {
        log("WebSocket closed");
        disconnect();
      };
    } catch (err) {
      console.error(err);
      updateStatus("Connection failed", "error");
    }
  };

  const disconnect = () => {
    setIsConnected(false);
    isSendingAudioRef.current = false;

    if (animationFrameRef.current)
      cancelAnimationFrame(animationFrameRef.current);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    // close websocket
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      try {
        websocketRef.current.close();
      } catch {}
    }
    websocketRef.current = null;

    // close audio context
    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
    }
    audioContextRef.current = null;

    // stop AI audio if playing
    if (aiAudioRef.current) {
      try {
        aiAudioRef.current.pause();
      } catch {}
      aiAudioRef.current = null;
    }
    mp3ChunksRef.current = [];
    isAIAudioActiveRef.current = false;

    setIsThinking(false);
    updateStatus("Click to start conversation", "ready");
    setAudioLevel(0);

    log("Disconnected");
  };

  const toggleConnection = () => {
    if (isConnected) disconnect();
    else connect();
  };

  const sendTypedMessage = () => {
    const text = typedText.trim();
    if (!text) return;

    if (
      !websocketRef.current ||
      websocketRef.current.readyState !== WebSocket.OPEN
    ) {
      alert("Not connected!");
      return;
    }

    websocketRef.current.send(JSON.stringify({ type: "text_message", text }));
    setTypedText("");
  };

  const clearHistory = () => {
    websocketRef.current?.send(JSON.stringify({ type: "clear_history" }));
  };

  useEffect(() => {
    return () => {
      if (isConnected) disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getEmotionColor = (emotion: string) => {
    const colors: Record<string, string> = {
      happy: "text-green-300",
      angry: "text-red-300",
      neutral: "text-gray-200",
      frustrated: "text-orange-300",
      confused: "text-yellow-300",
      calm: "text-cyan-300",
    };
    return colors[(emotion || "").toLowerCase()] || "text-gray-200";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-violet-950 flex items-center justify-center p-4 relative overflow-hidden">
      {/* background blobs */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
        <div
          className="absolute top-40 right-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute bottom-20 left-1/2 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"
          style={{ animationDelay: "4s" }}
        ></div>
      </div>

      <div className="relative backdrop-blur-xl bg-white/10 rounded-3xl p-8 max-w-7xl w-full shadow-2xl border border-white/20">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white px-6 py-3 rounded-2xl mb-4 shadow-lg">
            <Activity className="w-6 h-6 animate-pulse" />
            <h1 className="text-3xl font-bold">AI Call Center</h1>
          </div>
          <p className="text-white/70 text-sm">
            Realtime Multi-Agent Voice Intelligence
          </p>
        </div>

        {/* Analytics Dashboard */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <div className="text-xs text-white/60 font-semibold uppercase tracking-wider">
                Agent
              </div>
            </div>
            <div className="font-bold text-white text-sm truncate">
              {activeAgent}
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-blue-400" />
              <div className="text-xs text-white/60 font-semibold uppercase tracking-wider">
                Intent
              </div>
            </div>
            <div className="font-bold text-white text-sm">{intent}</div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="w-4 h-4 text-pink-400" />
              <div className="text-xs text-white/60 font-semibold uppercase tracking-wider">
                Emotion
              </div>
            </div>
            <div className={`font-bold text-sm ${getEmotionColor(emotion)}`}>
              {emotion}
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-2">
              Confidence
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-white/20 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-green-400 to-emerald-500 h-full transition-all duration-500 rounded-full"
                  style={{ width: `${confidence}%` }}
                />
              </div>
              <span className="font-bold text-white text-sm">
                {confidence}%
              </span>
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-2">
              CSAT
            </div>
            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map((star) => (
                <span
                  key={star}
                  className={`text-lg ${star <= csat ? "text-yellow-400" : "text-white/20"}`}
                >
                  ★
                </span>
              ))}
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              <div className="text-xs text-white/60 font-semibold uppercase tracking-wider">
                Resolution
              </div>
            </div>
            <div className="font-bold text-white text-sm">
              {resolutionTime}s{" "}
              {resolved && <span className="text-green-400">✓</span>}
            </div>
          </div>

          <div className="backdrop-blur-md bg-white/10 rounded-2xl p-4 border border-white/20 col-span-2 md:col-span-3 lg:col-span-6">
            <div className="text-xs text-white/60 font-semibold uppercase tracking-wider mb-2">
              Summary
            </div>
            <div className="font-semibold text-white text-sm">{summary}</div>
          </div>
        </div>

        {/* Voice Control */}
        <div className="flex flex-col items-center mb-8">
          <div className="relative mb-6">
            {isConnected && (
              <>
                <div className="absolute inset-0 rounded-full bg-gradient-to-r from-pink-500 to-purple-600 opacity-20 animate-ping"></div>
                <div
                  className="absolute inset-0 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 opacity-20 animate-ping"
                  style={{ animationDelay: "1s" }}
                ></div>
              </>
            )}

            <button
              onClick={toggleConnection}
              className={`relative w-40 h-40 rounded-full border-4 transition-all duration-500 transform hover:scale-110 shadow-2xl ${
                isConnected
                  ? isThinking
                    ? "bg-gradient-to-br from-amber-400 via-orange-500 to-red-600 border-orange-300 animate-pulse"
                    : "bg-gradient-to-br from-pink-500 via-purple-600 to-indigo-700 border-pink-300"
                  : "bg-gradient-to-br from-slate-700 to-slate-900 border-slate-500 hover:border-blue-400"
              }`}
            >
              {isConnected ? (
                <Square className="w-16 h-16 mx-auto text-white" />
              ) : (
                <Mic className="w-16 h-16 mx-auto text-white" />
              )}
            </button>
          </div>

          <div
            className={`px-6 py-3 rounded-2xl font-bold text-sm shadow-lg backdrop-blur-md border transition-all duration-300 ${
              status.type === "ready"
                ? "bg-emerald-500/30 border-emerald-400/50 text-emerald-100"
                : status.type === "connecting"
                  ? "bg-amber-500/30 border-amber-400/50 text-amber-100 animate-pulse"
                  : status.type === "listening"
                    ? "bg-blue-500/30 border-blue-400/50 text-blue-100"
                    : status.type === "thinking"
                      ? "bg-purple-500/30 border-purple-400/50 text-purple-100 animate-pulse"
                      : status.type === "speaking"
                        ? "bg-pink-500/30 border-pink-400/50 text-pink-100"
                        : "bg-red-500/30 border-red-400/50 text-red-100"
            }`}
          >
            {status.message}
          </div>
        </div>

        {/* Audio Visualizer */}
        <div className="h-20 backdrop-blur-md bg-white/5 rounded-2xl overflow-hidden relative border border-white/10 mb-8 shadow-inner">
          <div className="absolute inset-0 flex items-end justify-around px-2 gap-1">
            {Array.from({ length: 50 }).map((_, i) => (
              <div
                key={i}
                className="flex-1 bg-gradient-to-t from-cyan-500 via-blue-500 to-purple-600 rounded-t-full transition-all duration-100"
                style={{
                  height: `${Math.max(5, audioLevel * (0.5 + Math.random() * 0.8))}%`,
                  opacity: isConnected ? 0.8 : 0.2,
                }}
              ></div>
            ))}
          </div>
        </div>

        {/* Input Controls */}
        <div className="flex gap-3 mb-6">
          <input
            value={typedText}
            onChange={(e) => setTypedText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") sendTypedMessage();
            }}
            placeholder="Type a message..."
            className="flex-1 backdrop-blur-md bg-white/10 border-2 border-white/20 rounded-2xl px-5 py-4 outline-none focus:border-blue-400 focus:bg-white/20 transition-all text-white placeholder-white/50"
          />

          <button
            onClick={sendTypedMessage}
            className="bg-gradient-to-r from-blue-500 to-cyan-600 text-white px-6 py-4 rounded-2xl font-bold hover:from-blue-600 hover:to-cyan-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center gap-2"
          >
            <Send size={18} />
            Send
          </button>

          <button
            onClick={clearHistory}
            className="bg-gradient-to-r from-red-500 to-pink-600 text-white px-6 py-4 rounded-2xl font-bold hover:from-red-600 hover:to-pink-700 transition-all shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center gap-2"
          >
            <Trash2 size={18} />
          </button>
        </div>

        {/* Conversation */}
        <div
          ref={conversationBoxRef}
          className="backdrop-blur-md bg-white/5 border-2 border-white/10 rounded-2xl p-6 min-h-[320px] max-h-[450px] overflow-y-auto shadow-inner"
        >
          {messages.length === 0 && !interimMessage ? (
            <div className="text-center text-white/50 py-20">
              <div className="text-7xl mb-4">🎙️</div>
              <div className="text-lg font-semibold">
                Start your conversation
              </div>
              <div className="text-sm mt-2">
                Speak or type to begin chatting with AI agents
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, idx) => (
                <div
                  key={msg.id}
                  className={`mb-4 transform transition-all duration-300 ${
                    idx === messages.length - 1 ? "animate-slideIn" : ""
                  }`}
                  style={{
                    animation:
                      idx === messages.length - 1
                        ? "slideIn 0.3s ease-out"
                        : "none",
                  }}
                >
                  <div
                    className={`p-5 rounded-2xl leading-relaxed shadow-lg ${
                      msg.role === "user"
                        ? "bg-gradient-to-br from-blue-500/30 to-cyan-500/30 border border-blue-400/30 ml-[15%] backdrop-blur-sm"
                        : msg.role === "system"
                          ? "bg-gradient-to-br from-amber-500/20 to-orange-500/20 border border-amber-400/30 mx-[10%] text-center backdrop-blur-sm"
                          : "bg-gradient-to-br from-purple-500/30 to-pink-500/30 border border-purple-400/30 mr-[15%] backdrop-blur-sm"
                    }`}
                  >
                    <div className="text-xs font-bold uppercase mb-2 text-white/70 tracking-wider">
                      {msg.role === "user"
                        ? "You"
                        : msg.role === "system"
                          ? "System"
                          : msg.agent_name || "AI"}
                    </div>

                    <div className="text-white whitespace-pre-wrap font-medium">
                      {msg.text}
                    </div>
                  </div>
                </div>
              ))}

              {interimMessage && (
                <div className="mb-4 transform transition-all duration-200">
                  <div className="p-5 rounded-2xl leading-relaxed opacity-70 italic bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-400/20 ml-[15%] backdrop-blur-sm">
                    <div className="text-xs font-bold uppercase mb-2 text-white/60 tracking-wider">
                      You (speaking)
                    </div>
                    <div className="text-white/90 font-medium">
                      {interimMessage.text}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Debug */}
        <div className="mt-6 backdrop-blur-md bg-black/20 border border-white/10 rounded-xl p-4 text-xs text-white/60 font-mono">
          <span className="text-emerald-400 font-semibold">System:</span>{" "}
          {debugLog}
        </div>
      </div>

      <style jsx>{`
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </div>
  );
}
