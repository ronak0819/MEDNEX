from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, field_validator

from mednex.models import MEDNEXModel
from mednex.nlp import parse_patient_description
from mednex.safety.safety_checks import review_input

DEFAULT_GUIDANCE_PATH = Path(__file__).resolve().parent / "data" / "general_guidance.json"
MAX_MESSAGE_LENGTH = 4000
MEDICATION_REQUEST_PATTERN = re.compile(
    r"\b(?:what|which)\s+(?:medicine|medication|drug|tablet|pill)\b|"
    r"\b(?:recommend|suggest|prescribe)\s+(?:a\s+)?"
    r"(?:medicine|medication|drug|tablet|pill)\b",
    re.IGNORECASE,
)


class GuidanceTopic(MEDNEXModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    comfort_steps: list[str] = Field(min_length=1)
    seek_help: list[str] = Field(min_length=1)
    urgent_signs: list[str] = Field(min_length=1)


class GuidanceLibrary(MEDNEXModel):
    notice: str = Field(min_length=1)
    topics: list[GuidanceTopic] = Field(min_length=1)

    @field_validator("notice")
    @classmethod
    def require_educational_scope(cls, value: str) -> str:
        normalized = value.casefold()
        if "educational" not in normalized or "not a clinical guideline" not in normalized:
            raise ValueError("Guidance notice must identify its educational, non-guideline scope.")
        return value

    @field_validator("topics")
    @classmethod
    def require_unique_topic_ids(cls, topics: list[GuidanceTopic]) -> list[GuidanceTopic]:
        identifiers = [topic.id for topic in topics]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Guidance topic identifiers must be unique.")
        return topics


class GuidanceReply(MEDNEXModel):
    status: Literal["general_information", "safety_boundary", "needs_clarification"]
    response: str
    understood: list[str] = Field(default_factory=list)
    comfort_steps: list[str] = Field(default_factory=list)
    when_to_seek_help: list[str] = Field(default_factory=list)
    urgent_signs: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "General educational information only. Not a diagnosis or personalized "
        "medical advice. No medicines or dosages are recommended."
    )


def load_guidance_library(path: str | Path | None = None) -> GuidanceLibrary:
    source = Path(path) if path is not None else DEFAULT_GUIDANCE_PATH
    with source.open("r", encoding="utf-8") as guidance_file:
        return GuidanceLibrary.model_validate(json.load(guidance_file))


def respond_to_message(
    message: str,
    library: GuidanceLibrary | None = None,
) -> GuidanceReply:
    if not isinstance(message, str):
        raise TypeError("Message must be a string.")
    normalized_message = message.strip()
    if not normalized_message:
        raise ValueError("Enter a symptom or question to continue.")
    if len(normalized_message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message must be {MAX_MESSAGE_LENGTH} characters or fewer.")

    selected_library = library or load_guidance_library()
    safety = review_input(normalized_message)
    if safety.urgent_language_detected:
        return GuidanceReply(
            status="safety_boundary",
            response=(
                "I’m concerned by the urgent wording in your message. Please seek "
                "urgent medical help now or contact your local emergency service. "
                "I can’t assess an emergency through this chat."
            ),
        )
    if MEDICATION_REQUEST_PATTERN.search(normalized_message) or any(
        "individualized care guidance" in reason for reason in safety.reasons
    ):
        return GuidanceReply(
            status="safety_boundary",
            response=(
                "I can’t recommend a medicine, dose, or personal treatment plan. "
                "A pharmacist or qualified healthcare professional can advise you "
                "based on your circumstances. If symptoms are severe or rapidly "
                "worsening, seek urgent help."
            ),
        )

    extraction = parse_patient_description(normalized_message)
    mentioned_names = {
        symptom.name.casefold() for symptom in extraction.state.symptoms
    }
    matched_topics = [
        topic
        for topic in selected_library.topics
        if topic.id.replace("-", " ").casefold() in mentioned_names
    ]
    present_topics = [
        topic
        for topic in matched_topics
        if any(
            symptom.name.casefold() == topic.id.replace("-", " ").casefold()
            and not symptom.is_negated
            for symptom in extraction.state.symptoms
        )
    ]
    negated_labels = [
        symptom.name
        for symptom in extraction.state.symptoms
        if symptom.is_negated
    ]
    understood = [topic.label for topic in present_topics]

    if not present_topics:
        if negated_labels:
            response = (
                "Thanks for clarifying. I understood that "
                + ", ".join(negated_labels)
                + " was described as not present, so I haven’t treated it as a current symptom. "
                "Tell me what symptom or general health topic you would like information about."
            )
            status: Literal["general_information", "needs_clarification"] = "general_information"
        else:
            response = (
                "I’m not sure which symptom you mean. I can share general information "
                "for a cold, stomach pain, cough, fever, throat pain, headache, "
                "nasal congestion, nausea, diarrhea, or fatigue. Describe the concern "
                "in a few words, and I’ll show general comfort measures and when to seek help."
            )
            status = "needs_clarification"
        return GuidanceReply(status=status, response=response, understood=negated_labels)

    comfort_steps = [
        step for topic in present_topics for step in topic.comfort_steps
    ]
    seek_help = [item for topic in present_topics for item in topic.seek_help]
    urgent_signs = [item for topic in present_topics for item in topic.urgent_signs]
    symptom_phrase = (
        ", ".join(understood[:-1]) + f" and {understood[-1]}"
        if len(understood) > 1
        else understood[0]
    )
    response = (
        f"I’m sorry you’re dealing with {symptom_phrase}. I can share general "
        "comfort information, but I can’t determine the cause or assess your "
        "individual situation from a message."
    )
    if negated_labels:
        response += (
            " I also noted that "
            + ", ".join(negated_labels)
            + " was described as not present."
        )
    response += (
        "\n\nA few gentle things that may help:\n"
        + "\n".join(f"• {step}" for step in comfort_steps)
        + "\n\nWhen to contact a healthcare professional:\n"
        + "\n".join(f"• {item}" for item in seek_help)
        + "\n\nSeek urgent help if:\n"
        + "\n".join(f"• {item}" for item in urgent_signs)
        + "\n\nNo medicines or dosages are recommended here."
    )
    return GuidanceReply(
        status="general_information",
        response=response,
        understood=understood,
        comfort_steps=comfort_steps,
        when_to_seek_help=seek_help,
        urgent_signs=urgent_signs,
    )
