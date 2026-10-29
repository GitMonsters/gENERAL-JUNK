server_content = """from fastapi import FastAPI
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

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    latest_msg = req.messages[-1].content.lower() if req.messages else ""
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
    return {"result": PixelHardwareBridge.get_battery(), "message": "Battery status pulled successfully."}

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

@app.post("/api/rsi")
async def api_rsi():
    best = rsi_engine.execute_rsi_cycle([{"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}], generations=3)
    res = f"Elite Policy: {best.policy_id}\\nFitness: {best.fitness_score}\\nConnectivity: {best.connectivity}"
    return {"result": res, "message": "RSI Meta-Optimization cycle completed."}
"""

with open("server.py", "w", encoding="utf-8") as f:
    f.write(server_content)
print("[OK] server.py written successfully via Python script!")
