import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from pydantic import ValidationError

from mednex.guidance import GuidanceLibrary, load_guidance_library, respond_to_message
from mednex.web_server import MEDNEXRequestHandler
from http.server import ThreadingHTTPServer


def test_cough_message_gets_natural_general_support_not_a_diagnosis() -> None:
    reply = respond_to_message("I have a cough.")

    assert reply.status == "general_information"
    assert reply.understood == ["cough"]
    assert "I’m sorry you’re dealing with cough" in reply.response
    assert any("warm drink" in item for item in reply.comfort_steps)
    assert any("worsening" in item for item in reply.when_to_seek_help)
    assert "No medicines or dosages are recommended" in reply.disclaimer
    assert not any("Paracetamol" in item or "Dolo" in item for item in reply.comfort_steps)


def test_multiple_symptoms_return_combined_readable_general_information() -> None:
    reply = respond_to_message("I have a cough and fever.")

    assert reply.status == "general_information"
    assert set(reply.understood) == {"cough", "fever"}
    assert "cough and fever" in reply.response
    assert len(reply.comfort_steps) >= 4


@pytest.mark.parametrize(
    ("message", "expected", "guidance_phrase"),
    [
        ("I have a cold.", "cold symptoms", "Rest and drink fluids"),
        ("I have stomach pain.", "stomach or abdominal pain", "small sips of fluid"),
        ("My throat hurts; I have throat pain.", "sore throat", "warm or cool drinks"),
        ("I have a headache.", "headache", "drink water"),
    ],
)
def test_common_everyday_concerns_get_matched_general_information(
    message: str,
    expected: str,
    guidance_phrase: str,
) -> None:
    reply = respond_to_message(message)

    assert reply.status == "general_information"
    assert reply.understood == [expected]
    assert any(guidance_phrase in step for step in reply.comfort_steps)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I have a caugh.", "cough"),
        ("I have a feaver.", "fever"),
        ("I have stomack pain.", "stomach or abdominal pain"),
        ("I have a sore throte.", "sore throat"),
        ("I have a headake.", "headache"),
        ("I have a colld.", "cold symptoms"),
    ],
)
def test_common_explicit_misspellings_match_canonical_guidance(
    message: str,
    expected: str,
) -> None:
    reply = respond_to_message(message)

    assert reply.status == "general_information"
    assert reply.understood == [expected]
    assert reply.comfort_steps


def test_severe_stomach_pain_routes_to_urgent_help() -> None:
    reply = respond_to_message("I have sudden severe stomach pain.")

    assert reply.status == "safety_boundary"
    assert "urgent medical help" in reply.response
    assert reply.comfort_steps == []


def test_explicitly_negated_symptom_is_not_given_present_symptom_guidance() -> None:
    reply = respond_to_message("I do not have a cough.")

    assert reply.status == "general_information"
    assert reply.understood == ["cough"]
    assert reply.comfort_steps == []
    assert "not present" in reply.response


def test_personal_wording_receives_generic_non_personalized_information() -> None:
    reply = respond_to_message("I have a cough.")

    assert reply.status == "general_information"
    assert "individual situation" in reply.response
    assert "diagnosis" in reply.disclaimer


@pytest.mark.parametrize(
    "message",
    [
        "I cannot breathe.",
        "I have severe chest pain.",
        "A fictional person has difficulty breathing.",
    ],
)
def test_urgent_language_routes_to_urgent_help_without_self_care_instructions(
    message: str,
) -> None:
    reply = respond_to_message(message)

    assert reply.status == "safety_boundary"
    assert "urgent medical help" in reply.response
    assert reply.comfort_steps == []


def test_negated_urgent_phrase_does_not_trigger_urgent_boundary() -> None:
    reply = respond_to_message("I have a cough but no difficulty breathing.")

    assert reply.status == "general_information"
    assert reply.understood == ["cough"]


def test_urgent_phrase_after_contrast_is_not_hidden_by_earlier_negation() -> None:
    reply = respond_to_message(
        "I have no difficulty breathing, but I have severe chest pain."
    )

    assert reply.status == "safety_boundary"


def test_medicine_request_is_not_answered_with_a_drug_or_dose() -> None:
    reply = respond_to_message("I have a cough. What medicine should I take?")

    assert reply.status == "safety_boundary"
    assert "can’t recommend a medicine, dose" in reply.response


def test_unsupported_symptom_asks_for_clarification_without_guessing() -> None:
    reply = respond_to_message("I feel strange and unwell.")

    assert reply.status == "needs_clarification"
    assert reply.comfort_steps == []
    assert "which symptom you mean" in reply.response


def test_guidance_dataset_is_small_explicitly_scoped_and_has_unique_topics() -> None:
    library = load_guidance_library()

    assert len(library.topics) == 10
    assert "not a clinical guideline" in library.notice.casefold()
    assert len({topic.id for topic in library.topics}) == len(library.topics)


def test_guidance_dataset_rejects_missing_educational_boundary() -> None:
    with pytest.raises(ValidationError, match="educational"):
        GuidanceLibrary.model_validate(
            {
                "notice": "A general list.",
                "topics": [
                    {
                        "id": "cough",
                        "label": "cough",
                        "aliases": ["cough"],
                        "comfort_steps": ["Rest."],
                        "seek_help": ["Ask a professional if concerned."],
                        "urgent_signs": ["Seek urgent help for severe symptoms."],
                    }
                ],
            }
        )


@pytest.fixture
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


def test_local_html_and_css_are_served_without_streamlit(local_server: str) -> None:
    with urlopen(f"{local_server}/") as response:
        page = response.read().decode("utf-8")
        assert response.headers["Content-Type"].startswith("text/html")
        assert "MEDNEX assistant" in page
        assert 'data-prompt="I have a cold."' in page
        assert 'data-prompt="I have stomach pain."' in page
        assert 'id="agent-mode"' in page
        assert "Reasoning Agent" in page
        assert "styles.css" in page
    with urlopen(f"{local_server}/styles.css") as response:
        css = response.read().decode("utf-8")
        assert "grid-template-columns" in css
        assert "fonts.googleapis.com" not in css


def test_local_chat_api_returns_conversational_json_from_guidance(local_server: str) -> None:
    request = Request(
        f"{local_server}/api/chat",
        data=json.dumps({"message": "I have a cough."}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        payload = json.loads(response.read())

    assert response.status == 200
    assert payload["status"] == "general_information"
    assert payload["understood"] == ["cough"]
    assert "warm drink" in payload["response"]


def test_local_chat_api_rejects_missing_message(local_server: str) -> None:
    request = Request(
        f"{local_server}/api/chat",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)

    assert error.value.code == 400
