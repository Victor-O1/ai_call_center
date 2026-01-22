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
  CheckCircle,
} from "lucide-react";

export default function VoiceAIAssistant() {
  const WS_URL = "ws://localhost:8003/ws/voice";

  // connection + status
  const [isConnected, setIsConnected] = useState(false);
  const [status, setStatus] = useState({
    message: "Ready to connect",
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
  const [debugLog, setDebugLog] = useState("System initialized");
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

  const isSendingAudioRef = useRef(true);

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

  const visualizeAudio = () => {
    if (!isConnected || !analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;

    setAudioLevel(Math.min(100, avg * 1.4));
    animationFrameRef.current = requestAnimationFrame(visualizeAudio);
  };

  const pushMp3Chunk = (b64Data: string) => {
    try {
      const binary = atob(b64Data);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }

      mp3ChunksRef.current.push(bytes);

      if (!isAIAudioActiveRef.current) {
        isAIAudioActiveRef.current = true;
        isSendingAudioRef.current = false;
        updateStatus("AI responding", "speaking");
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
        updateStatus("Listening", "listening");
        return;
      }

      const blob = new Blob(chunks, { type: "audio/mpeg" });
      mp3ChunksRef.current = [];

      const url = URL.createObjectURL(blob);

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
        updateStatus("Listening", "listening");
      };

      audio.onerror = (err) => {
        console.error("Audio playback error:", err);
        URL.revokeObjectURL(url);
        aiAudioRef.current = null;

        isAIAudioActiveRef.current = false;
        isSendingAudioRef.current = true;
        updateStatus("Listening", "listening");
      };

      await audio.play();
    } catch (e) {
      console.error("playBufferedMp3() failed:", e);
      isAIAudioActiveRef.current = false;
      isSendingAudioRef.current = true;
      updateStatus("Listening", "listening");
    }
  };

  const handleMessage = async (data: any) => {
    switch (data.type) {
      case "ready":
        log("Backend connected");
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
        updateStatus("Processing", "thinking");
        setIsThinking(true);
        isSendingAudioRef.current = false;
        break;

      case "agent_switch":
        setActiveAgent(data.agent_name);
        addMessage(
          "system",
          `Transferred to ${data.agent_name}`,
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
        pushMp3Chunk(data.data);
        break;

      case "audio_complete":
        await playBufferedMp3();
        break;

      case "listening":
        updateStatus("Listening", "listening");
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

        updateStatus("Listening", "listening");
        log("Conversation cleared");
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

  const connect = async () => {
    try {
      log("Connecting to backend...");
      updateStatus("Connecting", "connecting");

      const ws = new WebSocket(WS_URL);
      websocketRef.current = ws;

      ws.onopen = async () => {
        log("Connection established");

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
          updateStatus("Listening", "listening");

          log("Microphone active");
        } catch (err: any) {
          console.error("Mic error:", err);
          alert("Microphone access denied: " + err.message);
          disconnect();
        }
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
      };

      ws.onerror = () => {
        log("Connection error");
      };

      ws.onclose = () => {
        log("Connection closed");
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

    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      try {
        websocketRef.current.close();
      } catch {}
    }
    websocketRef.current = null;

    if (audioContextRef.current && audioContextRef.current.state !== "closed") {
      audioContextRef.current.close();
    }
    audioContextRef.current = null;

    if (aiAudioRef.current) {
      try {
        aiAudioRef.current.pause();
      } catch {}
      aiAudioRef.current = null;
    }
    mp3ChunksRef.current = [];
    isAIAudioActiveRef.current = false;

    setIsThinking(false);
    updateStatus("Ready to connect", "ready");
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
      alert("Not connected");
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
      happy: "text-emerald-600",
      angry: "text-red-600",
      neutral: "text-slate-600",
      frustrated: "text-amber-600",
      confused: "text-yellow-600",
      calm: "text-blue-600",
    };
    return colors[(emotion || "").toLowerCase()] || "text-slate-600";
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <div className="w-full max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-slate-900">
                AI Call Center
              </h1>
              <p className="text-sm text-slate-500">
                Multi-agent voice intelligence platform
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Analytics */}
          <div className="lg:col-span-1 space-y-6">
            {/* Session Metrics */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h2 className="text-sm font-semibold text-slate-900 mb-4">
                Session Metrics
              </h2>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      Active Agent
                    </span>
                    <Zap className="w-3.5 h-3.5 text-indigo-600" />
                  </div>
                  <p className="text-sm font-medium text-slate-900 truncate">
                    {activeAgent}
                  </p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      Customer Intent
                    </span>
                    <TrendingUp className="w-3.5 h-3.5 text-slate-400" />
                  </div>
                  <p className="text-sm font-medium text-slate-900">{intent}</p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      Emotion
                    </span>
                    <Activity className="w-3.5 h-3.5 text-slate-400" />
                  </div>
                  <p
                    className={`text-sm font-medium ${getEmotionColor(emotion)}`}
                  >
                    {emotion}
                  </p>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      Confidence
                    </span>
                    <span className="text-xs font-medium text-slate-900">
                      {confidence}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-indigo-600 transition-all duration-500 rounded-full"
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Quality Metrics */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h2 className="text-sm font-semibold text-slate-900 mb-4">
                Quality Metrics
              </h2>
              <div className="space-y-4">
                <div>
                  <span className="text-xs text-slate-500 uppercase tracking-wide block mb-2">
                    CSAT Score
                  </span>
                  <div className="flex items-center gap-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <span
                        key={star}
                        className={`text-lg ${star <= csat ? "text-amber-400" : "text-slate-200"}`}
                      >
                        ★
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-slate-500 uppercase tracking-wide">
                      Resolution Time
                    </span>
                    <Clock className="w-3.5 h-3.5 text-slate-400" />
                  </div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-slate-900">
                      {resolutionTime}s
                    </p>
                    {resolved && (
                      <CheckCircle className="w-4 h-4 text-emerald-600" />
                    )}
                  </div>
                </div>

                <div>
                  <span className="text-xs text-slate-500 uppercase tracking-wide block mb-2">
                    Call Summary
                  </span>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {summary}
                  </p>
                </div>
              </div>
            </div>

            {/* System Status */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6">
              <h2 className="text-sm font-semibold text-slate-900 mb-3">
                System Status
              </h2>
              <div className="flex items-center gap-2 text-xs">
                <div
                  className={`w-2 h-2 rounded-full ${
                    isConnected ? "bg-emerald-500" : "bg-slate-300"
                  }`}
                />
                <span className="text-slate-600 font-mono">{debugLog}</span>
              </div>
            </div>
          </div>

          {/* Right Column - Conversation & Controls */}
          <div className="lg:col-span-2 space-y-6">
            {/* Voice Control */}
            <div className="bg-white rounded-2xl border border-slate-200 p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-sm font-semibold text-slate-900 mb-1">
                    Voice Interface
                  </h2>
                  <p
                    className={`text-xs font-medium ${
                      status.type === "ready"
                        ? "text-slate-500"
                        : status.type === "connecting"
                          ? "text-amber-600"
                          : status.type === "listening"
                            ? "text-indigo-600"
                            : status.type === "thinking"
                              ? "text-violet-600"
                              : status.type === "speaking"
                                ? "text-blue-600"
                                : "text-red-600"
                    }`}
                  >
                    {status.message}
                  </p>
                </div>
                <button
                  onClick={toggleConnection}
                  className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                    isConnected
                      ? isThinking
                        ? "bg-violet-100 text-violet-600 hover:bg-violet-200"
                        : "bg-indigo-600 text-white hover:bg-indigo-700"
                      : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                  }`}
                >
                  {isConnected ? (
                    <Square className="w-6 h-6" />
                  ) : (
                    <Mic className="w-6 h-6" />
                  )}
                </button>
              </div>

              {/* Audio Visualizer */}
              <div className="h-16 bg-slate-50 rounded-xl overflow-hidden relative border border-slate-100">
                <div className="absolute inset-0 flex items-end justify-around px-1 gap-0.5">
                  {Array.from({ length: 60 }).map((_, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-indigo-600 rounded-t transition-all duration-75"
                      style={{
                        height: `${Math.max(2, audioLevel * (0.4 + Math.random() * 0.6))}%`,
                        opacity: isConnected ? 0.7 : 0.1,
                      }}
                    ></div>
                  ))}
                </div>
              </div>
            </div>

            {/* Text Input */}
            <div className="bg-white rounded-2xl border border-slate-200 p-4">
              <div className="flex gap-3">
                <input
                  value={typedText}
                  onChange={(e) => setTypedText(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") sendTypedMessage();
                  }}
                  placeholder="Type a message..."
                  className="flex-1 bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-sm outline-none focus:border-indigo-600 focus:ring-2 focus:ring-indigo-100 transition-all text-slate-900 placeholder-slate-400"
                />
                <button
                  onClick={sendTypedMessage}
                  className="px-4 py-2.5 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition-all flex items-center gap-2 text-sm font-medium"
                >
                  <Send size={16} />
                  Send
                </button>
                <button
                  onClick={clearHistory}
                  className="px-4 py-2.5 bg-slate-100 text-slate-600 rounded-xl hover:bg-slate-200 transition-all flex items-center gap-2 text-sm font-medium"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            </div>

            {/* Conversation */}
            <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-200">
                <h2 className="text-sm font-semibold text-slate-900">
                  Conversation
                </h2>
              </div>
              <div
                ref={conversationBoxRef}
                className="p-6 h-[420px] overflow-y-auto"
              >
                {messages.length === 0 && !interimMessage ? (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="w-16 h-16 rounded-2xl bg-slate-100 flex items-center justify-center mb-4">
                      <Mic className="w-8 h-8 text-slate-400" />
                    </div>
                    <h3 className="text-sm font-medium text-slate-900 mb-1">
                      No active conversation
                    </h3>
                    <p className="text-sm text-slate-500">
                      Connect and start speaking to begin
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {messages.map((msg) => (
                      <div key={msg.id} className="animate-fadeIn">
                        {msg.role === "system" ? (
                          <div className="flex justify-center">
                            <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-full">
                              <div className="w-1.5 h-1.5 rounded-full bg-indigo-600" />
                              <span className="text-xs text-slate-600">
                                {msg.text}
                              </span>
                            </div>
                          </div>
                        ) : (
                          <div
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                          >
                            <div
                              className={`max-w-[75%] rounded-2xl px-4 py-3 ${
                                msg.role === "user"
                                  ? "bg-indigo-600 text-white"
                                  : "bg-slate-100 text-slate-900"
                              }`}
                            >
                              {msg.role === "assistant" && msg.agent_name && (
                                <div className="text-xs font-medium text-slate-500 mb-1">
                                  {msg.agent_name}
                                </div>
                              )}
                              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                                {msg.text}
                              </p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}

                    {interimMessage && (
                      <div className="flex justify-end animate-fadeIn">
                        <div className="max-w-[75%] rounded-2xl px-4 py-3 bg-indigo-100 text-indigo-900 border border-indigo-200">
                          <p className="text-sm leading-relaxed opacity-70 italic">
                            {interimMessage.text}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(4px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.2s ease-out;
        }
      `}</style>
    </div>
  );
}
