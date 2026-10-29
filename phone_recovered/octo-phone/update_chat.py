with open("server.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace the generic else block with spatial topology analysis
old_else = """    else:
        turn_count = len(req.messages)
        reply = f"Processed turn #{turn_count // 2 + 1} with context. Acknowledged: '{req.messages[-1].content}'. OctoTetrahedral spatial weights and RSI parameters are stable." """

new_else = """    else:
        # Convert user message into a dynamic spatial matrix based on character codes
        msg_bytes = [ord(c) % 2 for c in latest_msg[:9]]
        while len(msg_bytes) < 9:
            msg_bytes.append(0)
        grid = [msg_bytes[i:i+3] for i in range(0, 9, 3)]
        
        nodes = spatial_engine.extract_spatial_graph(grid)
        best_rsi = rsi_engine.execute_rsi_cycle([{"input": grid, "output": [[1]*3 for _ in range(3)]}], generations=2)
        
        reply = f"Spatial Analysis of '{req.messages[-1].content}': Extracted {len(nodes)} active topology nodes across 3x3 manifold. RSI Elite Policy [{best_rsi.policy_id}] optimized with fitness score {best_rsi.fitness_score} at connectivity {best_rsi.connectivity}." """

if old_else in code:
    code = code.replace(old_else, new_else)
    with open("server.py", "w", encoding="utf-8") as f:
        f.write(code)
    print("[OK] server.py upgraded with spatial reasoning chat!")
else:
    print("[INFO] Target pattern already updated or structured differently.")
