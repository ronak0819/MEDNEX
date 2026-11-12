import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from mednex.web_server import MEDNEXRequestHandler


@pytest.fixture(scope="module")
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), MEDNEXRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(server: str, route: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        f"{server}{route}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def test_reasoning_agent_returns_one_plain_english_answer(local_server: str) -> None:
    payload = _post(
        local_server,
        "/api/agent/start",
        {
            "description": (
                "A fictional 45-year-old male has a dry cough and fever for three days. "
                "His temperature is 38.5 degrees Celsius and oxygen saturation is 96 percent."
            )
        },
    )

    assert payload["awaiting_answer"] is True
    assert payload["session_id"]
    assert set(payload) == {"session_id", "answer", "awaiting_answer"}
    answer = payload["answer"]
    assert "cough" in answer and "fever" in answer
    assert "general comfort" in answer.casefold()
    assert "One follow-up question:" in answer
    assert "diagnose" in answer.casefold()
    assert "synthetic_pattern" not in answer
    assert "posterior" not in answer.casefold()
    assert "graph" not in answer.casefold()
    assert "entropy" not in answer.casefold()
    assert "38.5" not in answer and "96" not in answer and "45" not in answer


def test_follow_up_reuses_session_and_returns_a_simple_answer(local_server: str) -> None:
    first = _post(
        local_server,
        "/api/agent/start",
        {"description": "A fictional patient has a cough."},
    )
    assert first["awaiting_answer"] is True

    second = _post(
        local_server,
        "/api/agent/answer",
        {
            "session_id": first["session_id"],
            "answer": "No",
        },
    )

    assert second["session_id"] == first["session_id"]
    assert isinstance(second["answer"], str)
    assert "updated the fictional case" in second["answer"]
    assert "synthetic_pattern" not in second["answer"]
    assert "posterior" not in second["answer"].casefold()


def test_contradictory_description_returns_plain_uncertainty(local_server: str) -> None:
    payload = _post(
        local_server,
        "/api/agent/start",
        {
            "description": (
                "A fictional patient has fever but does not have a fever."
            )
        },
    )

    assert payload["awaiting_answer"] is False
    assert "both present and absent" in payload["answer"]
    assert "No conclusion was made." in payload["answer"]
    assert "UNCERTAIN" not in payload["answer"]
    assert "symbolic" not in payload["answer"].casefold()


def test_unsupported_case_returns_plain_uncertainty(local_server: str) -> None:
    payload = _post(
        local_server,
        "/api/agent/start",
        {"description": "A fictional patient reports blurry vision."},
    )

    assert payload["awaiting_answer"] is False
    assert "couldn’t confidently understand" in payload["answer"]
    assert "No conclusion was made." in payload["answer"]
    assert "validation" not in payload["answer"].casefold()


def test_concerning_case_returns_professional_help_message(local_server: str) -> None:
    payload = _post(
        local_server,
        "/api/agent/start",
        {"description": "A fictional patient has severe chest pain."},
    )

    assert payload["awaiting_answer"] is False
    assert "qualified health professional" in payload["answer"]
    assert "educational simulation" in payload["answer"]


def test_reasoning_agent_requires_explicit_synthetic_case_scope(local_server: str) -> None:
    with pytest.raises(HTTPError) as error:
        _post(
            local_server,
            "/api/agent/start",
            {"description": "A patient has a cough."},
        )

    assert error.value.code == 400


def test_agent_answer_rejects_unknown_session(local_server: str) -> None:
    with pytest.raises(HTTPError) as error:
        _post(
            local_server,
            "/api/agent/answer",
            {"session_id": "not-a-session", "answer": "Yes"},
        )

    assert error.value.code == 404


def test_web_client_uses_only_same_origin_endpoints_and_plain_answer(local_server: str) -> None:
    with urlopen(f"{local_server}/app.js") as response:
        source = response.read().decode("utf-8")

    assert 'postJson("/api/chat"' in source
    assert 'postJson("/api/agent/start"' in source
    assert 'postJson("/api/agent/answer"' in source
    assert "payload.answer" in source
    assert "turn.graph" not in source
    assert "class_probabilities" not in source
    assert "https://" not in source
    assert "openai" not in source.casefold()
