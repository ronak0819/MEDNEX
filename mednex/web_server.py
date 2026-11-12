from __future__ import annotations

import json
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from mednex.agent import AgentSession, AgentTurn, MEDNEXAgent
from mednex.guidance import GuidanceReply, respond_to_message
from mednex.user_response import build_user_facing_answer

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
MAX_REQUEST_BYTES = 16_384
MAX_AGENT_SESSIONS = 128
FICTIONAL_CASE_PATTERN = re.compile(r"\b(?:fictional|synthetic)\b", re.IGNORECASE)
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
_AGENT_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_REASONING_AGENT: MEDNEXAgent | None = None
_AGENT_SESSIONS: dict[str, AgentSession] = {}


def _get_reasoning_agent() -> MEDNEXAgent:
    global _REASONING_AGENT
    with _AGENT_LOCK:
        if _REASONING_AGENT is None:
            _REASONING_AGENT = MEDNEXAgent()
        return _REASONING_AGENT


def _store_session(session_id: str, session: AgentSession) -> None:
    with _SESSION_LOCK:
        _AGENT_SESSIONS[session_id] = session
        while len(_AGENT_SESSIONS) > MAX_AGENT_SESSIONS:
            _AGENT_SESSIONS.pop(next(iter(_AGENT_SESSIONS)))


def _get_session(session_id: str) -> AgentSession | None:
    with _SESSION_LOCK:
        return _AGENT_SESSIONS.get(session_id)


class MEDNEXRequestHandler(BaseHTTPRequestHandler):
    server_version = "MEDNEXLocal/1.0"

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/health":
            self._send_json(200, {"status": "ok"})
            return
        resource = STATIC_FILES.get(route)
        if resource is None:
            self._send_json(404, {"error": "Not found."})
            return
        filename, content_type = resource
        content = (WEB_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {
            "/api/chat",
            "/api/agent/start",
            "/api/agent/answer",
        }:
            self._send_json(404, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request size."})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request must contain a message within the size limit."})
            return
        if "application/json" not in self.headers.get("Content-Type", ""):
            self._send_json(415, {"error": "Send the message as application/json."})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("A JSON object is required.")
            if route == "/api/chat":
                if not isinstance(payload.get("message"), str):
                    raise ValueError("A text message is required.")
                reply = respond_to_message(payload["message"])
                response_payload = reply.model_dump(mode="json")
            elif route == "/api/agent/start":
                description = payload.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ValueError("A fictional case description is required.")
                if len(description) > 4000:
                    raise ValueError("Case description must be 4000 characters or fewer.")
                if not FICTIONAL_CASE_PATTERN.search(description):
                    raise ValueError(
                        "For safety, Reasoning Agent accepts only explicitly fictional "
                        "or synthetic educational cases. Include that wording in the case."
                    )
                turn = _get_reasoning_agent().start(description)
                session_id = uuid.uuid4().hex
                _store_session(session_id, turn.session)
                response_payload = _agent_payload(session_id, turn)
            else:
                session_id = payload.get("session_id")
                answer = payload.get("answer")
                if not isinstance(session_id, str) or not isinstance(answer, str):
                    raise ValueError("A session_id and text answer are required.")
                session = _get_session(session_id)
                if session is None:
                    self._send_json(
                        404,
                        {"error": "This reasoning session was not found or has expired. Start a new case."},
                    )
                    return
                turn = _get_reasoning_agent().answer(session, answer)
                _store_session(session_id, turn.session)
                response_payload = _agent_payload(session_id, turn)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as error:
            self._send_json(400, {"error": str(error)})
            return
        self._send_json(200, response_payload)

    def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        return


def _agent_payload(session_id: str, turn: AgentTurn) -> dict[str, object]:
    return {
        "session_id": session_id,
        "answer": build_user_facing_answer(turn),
        "awaiting_answer": turn.status == "NEEDS_MORE_INFORMATION",
    }


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("MEDNEX_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), MEDNEXRequestHandler)
    print(f"MEDNEX local educational chat: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping MEDNEX server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
