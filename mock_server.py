"""Mock SGLang-style OpenAI-compatible endpoint for local validation ONLY.

Simulates:
  * SSE streaming chat completions
  * usage.prompt_tokens + prompt_tokens_details.cached_tokens (fake prefix cache)
  * load-dependent TTFT: cheap until in-flight concurrency exceeds a threshold,
    then grows -> produces a knee so we can validate knee detection.

Run:  python mock_server.py 30000
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_inflight = 0
_lock = threading.Lock()

# Tunables for the simulated server.
TTFT_BASE_MS = 35.0
KNEE_CONCURRENCY = 8          # below this, flat; above, TTFT climbs
TTFT_SLOPE_MS = 22.0         # ms per excess concurrent request
ITL_MS = 9.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        global _inflight
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            req = {}
        messages = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 16))
        stream = bool(req.get("stream", False))

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = max(1, prompt_chars // 4)
        # Fake prefix cache: pretend everything but the last ~250 chars is cached
        # when there's a system message (mimics RadixAttention warm prefix).
        has_system = any(m.get("role") == "system" for m in messages)
        cached = max(0, prompt_tokens - 60) if has_system else 0

        with _lock:
            _inflight += 1
            conc = _inflight
        try:
            excess = max(0, conc - KNEE_CONCURRENCY)
            ttft_ms = TTFT_BASE_MS + TTFT_SLOPE_MS * excess
            time.sleep(ttft_ms / 1000.0)

            if not stream:
                self._send_json({
                    "id": "mock", "object": "chat.completion",
                    "choices": [{"index": 0, "message":
                                 {"role": "assistant", "content": "x"},
                                 "finish_reason": "length"}],
                    "usage": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": 1,
                              "total_tokens": prompt_tokens + 1,
                              "prompt_tokens_details": {"cached_tokens": cached}},
                })
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def sse(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            # role delta
            sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
            for i in range(max_tokens):
                if i > 0:
                    time.sleep(ITL_MS / 1000.0)
                sse({"choices": [{"index": 0, "delta": {"content": "tok "}}]})
            # final usage chunk
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                 "usage": {"prompt_tokens": prompt_tokens,
                           "completion_tokens": max_tokens,
                           "total_tokens": prompt_tokens + max_tokens,
                           "prompt_tokens_details": {"cached_tokens": cached}}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with _lock:
                _inflight -= 1

    def _send_json(self, obj):
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"mock server on :{port}")
    srv.serve_forever()
