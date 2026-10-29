server_code = r'''from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from octotetrahedral_spatial_rsi import SpatialTopologyEngine, RecursiveSelfImprovementEngine
from octotetrahedral_device_assistant import PixelHardwareBridge

app = FastAPI(title="OctoTetrahedral-AGI Dashboard")
spatial_engine = SpatialTopologyEngine(connectivity=4)
rsi_engine = RecursiveSelfImprovementEngine(population_size=6, mutation_rate=0.3)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OctoTetrahedral-AGI | Gemini Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .mono { font-family: 'JetBrains Mono', monospace; } 
    </style>
</head>
<body class="bg-[#090d16] text-slate-100 min-h-screen flex flex-col">
    <header class="border-b border-slate-800/80 bg-[#090d16]/80 backdrop-blur sticky top-0 z-50 px-6 py-3 flex justify-between items-center">
        <div class="flex items-center space-x-3">
            <div class="text-indigo-400 font-bold text-lg">✦</div>
            <h1 class="font-semibold text-sm tracking-wide text-slate-200 mono">OCTOTETRAHEDRAL-AGI</h1>
        </div>
        <div class="flex items-center space-x-2">
            <span class="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
            <span class="text-xs px-2.5 py-1 bg-slate-900 border border-slate-800 text-slate-400 rounded-full mono">Pixel / Termux Node</span>
        </div>
    </header>

    <main class="flex-1 max-w-5xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
                <div class="flex justify-between items-center mb-3">
                    <span class="text-xs font-semibold tracking-wider text-slate-400 uppercase mono">Hardware Bridge</span>
                    <span id="battery-box" class="text-xs mono text-emerald-400 bg-emerald-950/40 border border-emerald-800/50 px-2 py-0.5 rounded-md">Battery: --</span>
                </div>
                <div class="grid grid-cols-4 gap-2">
                    <button onclick="sendAction('battery')" class="bg-slate-800/80 hover:bg-slate-700 transition px-2 py-2 rounded-xl text-xs font-medium cursor-pointer">Battery</button>
                    <button onclick="sendAction('vibrate')" class="bg-slate-800/80 hover:bg-slate-700 transition px-2 py-2 rounded-xl text-xs font-medium cursor-pointer">Haptic</button>
                    <button onclick="sendAction('torch_on')" class="bg-emerald-600/80 hover:bg-emerald-500 transition px-2 py-2 rounded-xl text-xs font-medium text-slate-950 font-semibold cursor-pointer">Torch On</button>
                    <button onclick="sendAction('torch_off')" class="bg-rose-600/60 hover:bg-rose-600 transition px-2 py-2 rounded-xl text-xs font-medium cursor-pointer">Torch Off</button>
                </div>
            </div>

            <div class="bg-slate-900/50 border border-slate-800/80 rounded-2xl p-4 flex flex-col justify-between">
                <div class="flex justify-between items-center mb-2">
                    <span class="text-xs font-semibold tracking-wider text-slate-400 uppercase mono">RSI Engine</span>
                    <button onclick="sendAction('rsi')" class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-3 py-1 rounded-lg text-xs transition cursor-pointer">Run Cycle</button>
                </div>
                <div id="rsi-box" class="text-xs mono text-cyan-300 bg-slate-950 p-2.5 rounded-xl border border-slate-800/60 truncate">
                    Status: Nominal (Connectivity: 4)
                </div>
            </div>
        </div>

        <div class="flex-1 bg-slate-900/30 border border-slate-800/80 rounded-2xl flex flex-col overflow-hidden min-h-[420px]">
            <div id="chat-history" class="flex-1 p-4 md:p-6 overflow-y-auto space-y-6 text-sm">
                <div class="flex items-start space-x-4">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center text-slate-950 font-bold shrink-0">✦</div>
                    <div class="space-y-1 max-w-[85%]">
                        <div class="text-xs font-semibold text-slate-400 mono">OctoTetrahedral Node</div>
                        <div class="text-slate-200 bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 leading-relaxed">
                            Greetings. I am your local mobile AGI conversational bridge running on Pixel via Termux. Spatial topology engines active and RSI loops primed.
                        </div>
                    </div>
                </div>
            </div>

            <div class="p-4 bg-[#090d16]/80 border-t border-slate-800/80">
                <form onsubmit="handleChatSubmit(event)" class="max-w-4xl mx-auto relative flex items-center">
                    <input type="text" id="chat-input" placeholder="Ask anything or command device (e.g. 'check battery', 'run rsi')..." 
                        class="w-full bg-slate-900/90 border border-slate-700/80 rounded-2xl pl-4 pr-14 py-4 text-sm text-slate-100 focus:outline-none focus:border-indigo-500 mono">
                    <button type="submit" class="absolute right-3 bg-indigo-600 hover:bg-indigo-500 text-white p-2.5 rounded-xl transition cursor-pointer">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M12 5l7 7-7 7"/></svg>
                    </button>
                </form>
            </div>
        </div>
    </main>

    <script>
        let conversationHistory = [];

        async function sendAction(action) {
            try {
                const res = await fetch('/api/' + action, { method: 'POST' });
                const data = await res.json();
                if (action === 'battery' && data.result) {
                    document.getElementById('battery-box').innerText = "Battery: " + data.result;
                }
                if (action === 'rsi' && data.result) {
                    document.getElementById('rsi-box').innerText = data.result.replace(/\n/g, ' | ');
                }
                appendSystemNotice(data.message);
            } catch (err) {
                appendSystemNotice("Action error: " + err);
            }
        }

        async function handleChatSubmit(event) {
            event.preventDefault();
            const input = document.getElementById('chat-input');
            const userMsg = input.value.trim();
            if (!userMsg) return;

            appendMessage('user', userMsg);
            conversationHistory.push({ role: 'user', content: userMsg });
            input.value = '';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ messages: conversationHistory })
                });
                const data = await res.json();
                appendMessage('ai', data.reply);
                conversationHistory.push({ role: 'assistant', content: data.reply });

                if (data.battery) {
                    document.getElementById('battery-box').innerText = "Battery: " + data.battery;
                }
            } catch (err) {
                appendMessage('ai', 'Error connecting to local node.');
            }
        }

        function appendMessage(role, text) {
            const history = document.getElementById('chat-history');
            const isUser = role === 'user';
            
            const html = isUser ? `
                <div class="flex items-start justify-end space-x-4">
                    <div class="space-y-1 max-w-[85%] text-right">
                        <div class="text-xs font-semibold text-indigo-400 mono">You</div>
                        <div class="text-slate-100 bg-indigo-600/20 border border-indigo-500/30 rounded-2xl p-4 text-left leading-relaxed">
                            ${text}
                        </div>
                    </div>
                </div>` : `
                <div class="flex items-start space-x-4">
                    <div class="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center text-slate-950 font-bold shrink-0">✦</div>
                    <div class="space-y-1 max-w-[85%]">
                        <div class="text-xs font-semibold text-slate-400 mono">OctoTetrahedral Node</div>
                        <div class="text-slate-200 bg-slate-900/80 border border-slate-800/80 rounded-2xl p-4 leading-relaxed">
                            ${text}
                        </div>
                    </div>
                </div>`;
            
            history.innerHTML += html;
            history.scrollTop = history.scrollHeight;
        }

        function appendSystemNotice(text) {
            const history = document.getElementById('chat-history');
            history.innerHTML += `
                <div class="text-center my-2">
                    <span class="text-[11px] bg-slate-900/90 border border-slate-800 text-slate-400 px-3 py-1 rounded-full mono">⚡ [SYSTEM] ${text}</span>
                </div>`;
            history.scrollTop = history.scrollHeight;
        }

        window.onload = () => sendAction('battery');
    </script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return HTML_UI

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    latest_msg = req.messages[-1].content.lower() if req.messages else ""
    reply = ""
    battery_status = None

    if "battery" in latest_msg:
        battery_status = PixelHardwareBridge.get_battery()
        reply = f"Current device telemetry reports: {battery_status}"
    elif "torch" in latest_msg or "flashlight" in latest_msg:
        state = "off" if "off" in latest_msg else "on"
        PixelHardwareBridge.control_torch(state)
        reply = f"Hardware bridge successfully switched device torch {state}."
    elif "vibrate" in latest_msg or "pulse" in latest_msg:
        PixelHardwareBridge.vibrate(250)
        reply = "Triggered haptic motor pulse on device."
    elif "rsi" in latest_msg or "optimize" in latest_msg:
        best = rsi_engine.execute_rsi_cycle([{"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}], generations=3)
        reply = f"RSI Meta-Optimization complete. Elite Policy: {best.policy_id} | Fitness: {best.fitness_score} | Connectivity: {best.connectivity}"
    elif "hello" in latest_msg or "hi" in latest_msg:
        reply = "Hello! I am maintaining conversational context across our session. How can I help you steer or analyze the spatial network?"
    else:
        turn_count = len(req.messages)
        reply = f"Processed turn #{turn_count // 2 + 1} with context. Acknowledged: '{req.messages[-1].content}'. OctoTetrahedral spatial weights and RSI parameters are stable."

    return {"reply": reply, "battery": battery_status}

@app.post("/api/battery")
async def api_battery():
    res = PixelHardwareBridge.get_battery()
    return {"result": res, "message": "Battery status pulled successfully."}

@app.post("/api/vibrate")
async def api_vibrate():
    PixelHardwareBridge.vibrate(200)
    return {"message": "Triggered haptic vibration feedback."}

@app.post("/api/torch_on")
async def api_torch_on():
    PixelHardwareBridge.control_torch("on")
    return {"message": "Turned device flashlight ON."}

@app.post("/api/torch_off")
async def api_torch_off():
    PixelHardwareBridge.control_torch("off")
    return {"message": "Turned device flashlight OFF."}

@app.post("/api/analyze")
async def api_analyze():
    nodes = spatial_engine.extract_spatial_graph([[0, 1, 1], [0, 1, 0], [0, 0, 0]])
    return {"message": f"Extracted {len(nodes)} spatial nodes successfully."}

@app.post("/api/rsi")
async def api_rsi():
    best = rsi_engine.execute_rsi_cycle([{"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}], generations=3)
    res = f"Elite Policy: {best.policy_id}\nFitness: {best.fitness_score}\nConnectivity: {best.connectivity}"
    return {"result": res, "message": "RSI Meta-Optimization cycle completed."}
'''

with open("server.py", "w") as f:
    f.write(server_code)
print("[OK] Generated server.py successfully!")
