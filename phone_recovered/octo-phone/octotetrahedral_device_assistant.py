"""
octotetrahedral_device_assistant.py
===================================
Local Mobile Assistant & Hardware Bridge for Pixel / Termux environment.
Integrates spatial RSI optimization, Termux API hardware controls, and Voice-to-Text.
"""

import os
import sys
import json
import subprocess
from typing import Dict, Any

# Safe NumPy import check
try:
    import numpy as np
    HAS_NUMPY = True
except (ImportError, ModuleNotFoundError):
    HAS_NUMPY = False

from octotetrahedral_spatial_rsi import (
    SpatialTopologyEngine,
    RecursiveSelfImprovementEngine,
    SolverHeuristicPolicy
)


class PixelHardwareBridge:
    """Interfaces with Termux API tools to control local Pixel device hardware."""
    
    @staticmethod
    def run_cmd(cmd: list) -> str:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return result.stdout.strip() or result.stderr.strip()
        except Exception as e:
            return f"Hardware Error: {str(e)}"

    @classmethod
    def get_battery(cls) -> str:
        res = cls.run_cmd(["termux-battery-status"])
        try:
            data = json.loads(res)
            return f"Battery: {data.get('percentage', 'Unknown')}% | Status: {data.get('status', 'Unknown')} | Health: {data.get('health', 'Unknown')}"
        except Exception:
            return f"Raw Battery Status: {res}"

    @classmethod
    def control_torch(cls, state: str) -> str:
        return cls.run_cmd(["termux-torch", state])

    @classmethod
    def vibrate(cls, duration_ms: int = 500) -> str:
        return cls.run_cmd(["termux-vibrate", "-d", str(duration_ms)])

    @classmethod
    def speak(cls, text: str) -> str:
        return cls.run_cmd(["termux-tts-speak", text])

    @classmethod
    def listen(cls) -> str:
        print("[*] Launching speech recognition (speak into your phone)...")
        return cls.run_cmd(["termux-speech-to-text"])


class OctoTetrahedralAssistant:
    """Interactive command-line assistant running local RSI and hardware control."""
    def __init__(self):
        self.rsi_engine = RecursiveSelfImprovementEngine(population_size=6, mutation_rate=0.3)
        self.spatial_engine = SpatialTopologyEngine(connectivity=4)
        print("\n[OK] OctoTetrahedral AGI & Pixel Device Assistant Initialized.")
        print("Type 'help' for available commands or 'exit' to quit.\n")

    def run_repl(self):
        while True:
            try:
                command = input("OctoTetrahedral-AGI ~ $ ").strip()
                if not command:
                    continue
                
                parts = command.split(" ", 1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("exit", "quit"):
                    print("Shutting down assistant. Goodbye!")
                    break
                elif cmd == "help":
                    print("\nAvailable Commands:")
                    print("  listen               - Listen for voice input via Android speech-to-text")
                    print("  speak <text>         - Use device Text-To-Speech (TTS)")
                    print("  battery              - Check Pixel battery status via Termux API")
                    print("  torch [on/off]       - Turn device flashlight on or off")
                    print("  vibrate              - Trigger device haptic vibration")
                    print("  rsi <generations>    - Run Recursive Self-Improvement optimization cycle")
                    print("  analyze              - Run a test spatial graph extraction on sample grid")
                    print("  exit / quit          - Exit the assistant\n")
                elif cmd == "listen":
                    spoken = PixelHardwareBridge.listen()
                    print(f"Heard Voice Input: \"{spoken}\"")
                elif cmd == "speak":
                    text_to_say = arg if arg else "OctoTetrahedral spatial engine online."
                    print(f"Speaking: {PixelHardwareBridge.speak(text_to_say)}")
                elif cmd == "battery":
                    print(PixelHardwareBridge.get_battery())
                elif cmd == "torch":
                    state = arg.lower() if arg in ("on", "off") else "on"
                    print(f"Setting torch {state}: {PixelHardwareBridge.control_torch(state)}")
                elif cmd == "vibrate":
                    print(f"Triggering haptic feedback: {PixelHardwareBridge.vibrate(400)}")
                elif cmd == "rsi":
                    gens = int(arg) if arg.isdigit() else 3
                    print(f"Executing RSI Meta-Optimization loop for {gens} generations...")
                    dummy_tasks = [
                        {"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]},
                        {"input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}
                    ]
                    best_policy = self.rsi_engine.execute_rsi_cycle(dummy_tasks, generations=gens)
                    print(f"RSI Complete! Elite Policy Found: {best_policy.policy_id} | Fitness: {best_policy.fitness_score} | Connectivity: {best_policy.connectivity}")
                elif cmd == "analyze":
                    sample_grid = [[0, 1, 1], [0, 1, 0], [0, 0, 0]]
                    nodes = self.spatial_engine.extract_spatial_graph(sample_grid)
                    print(f"Extracted {len(nodes)} spatial graph nodes from sample grid.")
                    for node in nodes:
                        print(f"  - Node ID: {node.node_id} | Color: {node.color} | Area: {node.rsi.area} | Centroid: {node.rsi.rsi_centroid}")
                else:
                    print(f"Unknown command: '{cmd}'. Type 'help' for available actions.")

            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print(f"[ERROR] {str(e)}")


if __name__ == "__main__":
    assistant = OctoTetrahedralAssistant()
    assistant.run_repl()
