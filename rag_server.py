#!/usr/bin/env python3
"""Demo RAG server + web UI (stdlib only, like chat_server.py).

    python rag_server.py --corpus "%TEMP%/opencode/physics_corpus_enwiki.jsonl" \
        --port 8766 --limit-docs 4000 [--generator ollama --model llama3.1:8b]

Endpoints
    GET  /                 the UI
    GET  /api/status       corpus size, backend, whether an LLM is attached
    POST /api/ask          {"question": "...", "k": 4}  -> answer + sources
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from rag import RAG

STATE: dict = {}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>GM RAG assistant</title><style>
 body{font-family:system-ui,Segoe UI,sans-serif;max-width:840px;margin:24px auto;padding:0 16px;color:#111}
 h1{font-size:20px} .pill{display:inline-block;background:#eef;border:1px solid #ccd;border-radius:999px;
 padding:2px 10px;font-size:12px;margin-left:6px}
 #q{width:100%;padding:10px;font-size:15px;border:1px solid #ccc;border-radius:8px}
 button{padding:9px 16px;margin-top:8px;border-radius:8px;border:1px solid #888;background:#f6f6f6;cursor:pointer}
 .ans{margin-top:16px;padding:12px;background:#f2f7f2;border-left:4px solid #4a4;border-radius:6px;white-space:pre-wrap}
 .src{margin-top:8px;padding:8px;background:#fafafa;border:1px solid #eee;border-radius:6px;font-size:13px;color:#333}
 .n{color:#888;font-size:12px} #busy{color:#888;margin-left:8px}
</style></head><body>
<h1>GM-GPT RAG assistant<span class="pill" id="pill">…</span></h1>
<p class="n">Answers come from your documents via retrieval. Sources are shown;
if the corpus lacks the answer the assistant should refuse.</p>
<input id="q" placeholder="Ask about the documents…" autocomplete="off">
<button id="go">Ask</button><span id="busy"></span>
<div id="out"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q) return;
  const out=document.getElementById('out'), busy=document.getElementById('busy');
  busy.textContent='searching…'; out.innerHTML='';
  try{
    const r=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,k:4})});
    const d=await r.json();
    let h='<div class="ans">'+esc(d.answer)+'</div>';
    if(d.sources && d.sources.length){
      h+='<div class="n" style="margin-top:10px">sources</div>';
      for(const s of d.sources){
        h+='<div class="src"><b>['+s.n+'] '+esc(s.title)+'</b> <span class="n">score '+s.score+'</span><br>'+esc(s.text)+'</div>';
      }
    }
    out.innerHTML=h;
  }catch(e){ out.innerHTML='<div class="ans">error: '+e+'</div>'; }
  busy.textContent='';
}
function esc(s){return (s+'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
document.getElementById('go').onclick=ask;
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
fetch('/api/status').then(r=>r.json()).then(d=>{
  document.getElementById('pill').textContent=d.chunks+' chunks · '+d.docs+' docs · '+d.backend;
});
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quieter
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            r: RAG = STATE["rag"]
            body = json.dumps({
                "ok": True,
                "chunks": len(r.retriever.chunks),
                "docs": STATE["docs"],
                "backend": r.generator.backend,
                "k": r.k,
            }).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            q = (req.get("question") or "").strip()
            k = int(req.get("k") or 4)
            if not q:
                self._send(400, b'{"error":"empty question"}', "application/json")
                return
            r: RAG = STATE["rag"]
            t0 = time.time()
            a = r.ask(q, k=k)
            body = json.dumps({"answer": a.text, "grounded": a.grounded,
                               "backend": a.backend, "sources": a.sources,
                               "ms": int((time.time() - t0) * 1000)},
                              ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--limit-docs", type=int, default=4000)
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--generator", default="extractive")
    ap.add_argument("--model", default="llama3.1:8b")
    args = ap.parse_args()

    t0 = time.time()
    rag = RAG.from_jsonl(args.corpus, generator=args.generator, k=args.k,
                         target=args.target, limit_docs=args.limit_docs)
    STATE["rag"] = rag
    STATE["docs"] = args.limit_docs
    print(f"[index] {len(rag.retriever.chunks)} chunks from {args.limit_docs} docs "
          f"in {time.time()-t0:.1f}s | generator={rag.generator.backend}", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] http://127.0.0.1:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
