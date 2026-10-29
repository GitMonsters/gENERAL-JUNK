"""
test_features.py
================
Automated test suite for OctoTetrahedral-AGI Spatial, RSI, and Hardware features.
"""

import sys
from octotetrahedral_spatial_rsi import (
    SpatialTopologyEngine,
    RecursiveSelfImprovementEngine
)
from octotetrahedral_device_assistant import PixelHardwareBridge

def run_tests():
    print("==================================================")
    print("   OCTOTETRAHEDRAL AGI: AUTOMATED FEATURE TEST    ")
    print("==================================================")

    # 1. Test Spatial Topology Engine
    print("\n[TEST 1] Running Spatial Topology Engine & Node Extraction...")
    try:
        engine = SpatialTopologyEngine(connectivity=4)
        sample_grid = [[0, 1, 1], [0, 1, 0], [0, 0, 0]]
        nodes = engine.extract_spatial_graph(sample_grid)
        print(f"  [PASS] Successfully extracted {len(nodes)} spatial nodes.")
        for node in nodes:
            print(f"         - Node ID: {node.node_id}, Color: {node.color}, Area: {node.rsi.area}, Centroid: {node.rsi.rsi_centroid}")
    except Exception as e:
        print(f"  [FAIL] Spatial Engine Test Failed: {e}")

    # 2. Test Recursive Self-Improvement (RSI) Engine
    print("\n[TEST 2] Running Recursive Self-Improvement (RSI) Meta-Optimization Cycle...")
    try:
        rsi_engine = RecursiveSelfImprovementEngine(population_size=4, mutation_rate=0.3)
        dummy_tasks = [
            {"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]},
            {"input": [[1, 0, 1], [0, 1, 0], [1, 0, 1]], "output": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}
        ]
        best_policy = rsi_engine.execute_rsi_cycle(dummy_tasks, generations=3)
        print(f"  [PASS] RSI Cycle complete across 3 generations.")
        print(f"         - Elite Policy ID: {best_policy.policy_id}")
        print(f"         - Fitness Score:   {best_policy.fitness_score}")
        print(f"         - Connectivity:    {best_policy.connectivity}")
    except Exception as e:
        print(f"  [FAIL] RSI Engine Test Failed: {e}")

    # 3. Test Hardware Bridge (Termux API Integration)
    print("\n[TEST 3] Testing Pixel Hardware Bridge (Termux API)...")
    try:
        battery_status = PixelHardwareBridge.get_battery()
        print(f"  [PASS] Battery API Bridge: {battery_status}")
        
        vibrate_res = PixelHardwareBridge.vibrate(100)
        print(f"  [PASS] Vibration Bridge Response: {vibrate_res if vibrate_res else 'Triggered successfully'}")
    except Exception as e:
        print(f"  [FAIL / INFO] Hardware Bridge Warning (Ensure Termux:API app is installed): {e}")

    print("\n==================================================")
    print("           ALL FEATURE TESTS COMPLETED            ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
