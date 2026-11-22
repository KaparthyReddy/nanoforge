import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from nanoforge.bridge import OllamaClient, OllamaError, run_local
from nanoforge.data import CharTokenizer
from nanoforge.gpt import GPT, GPTConfig


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "mock:latest"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        if self.path == "/api/generate":
            self._send(
                {
                    "response": " world " + payload["prompt"],
                    "eval_count": 10,
                    "eval_duration": 500_000_000,
                }
            )
        else:
            self.send_error(404)


@pytest.fixture
def host():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_models_and_availability(host):
    client = OllamaClient(host)
    assert client.models() == ["mock:latest"]
    assert client.is_available()


def test_generate(host):
    result = OllamaClient(host).generate("mock:latest", "hello", max_tokens=10, seed=1)
    assert result.text == " world hello"
    assert result.tokens == 10
    assert result.tokens_per_sec == pytest.approx(20.0)


def test_unreachable_host():
    client = OllamaClient("http://127.0.0.1:9", timeout=1.0)
    assert not client.is_available()
    with pytest.raises(OllamaError):
        client.models()


def test_run_local():
    tok = CharTokenizer.from_text("abcdefg")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg, seed=0)
    result = run_local(model, tok, "abc", max_new_tokens=5, temperature=0, seed=0)
    assert result.tokens == 5
    assert len(result.text) == 5
    assert result.seconds > 0
    assert result.tokens_per_sec > 0
