from __future__ import annotations

from mednex.agent import AgentTurn
from mednex.guidance import GuidanceLibrary, load_guidance_library


def build_user_facing_answer(
    turn: AgentTurn,
    library: GuidanceLibrary | None = None,
) -> str:
    """Translate an actual agent turn into one short, non-technical response."""
    if turn.safety.professional_assessment_required:
        message = turn.safety.user_message or (
            "This case needs assessment by a qualified healthcare professional."
        )
        return (
            f"{message}\n\n"
            "MEDNEX is an educational simulation, not a medical service."
        )

    extraction = turn.extraction
    if extraction is None:
        return (
            "I couldn’t prepare a clear educational summary from this case. "
            "Please revise it using a brief, explicitly fictional description."
        )

    symptom_states: dict[str, set[bool]] = {}
    for symptom in extraction.state.symptoms:
        symptom_states.setdefault(symptom.name, set()).add(symptom.is_negated)
    conflicting_symptoms = sorted(
        name for name, states in symptom_states.items() if len(states) > 1
    )
    if conflicting_symptoms or extraction.contradictions:
        if conflicting_symptoms:
            names = ", ".join(conflicting_symptoms)
            detail = f"The case describes {names} as both present and absent."
        else:
            detail = "Some details in the case conflict with one another."
        return (
            f"I couldn’t make a clear summary because {detail} "
            "Please correct the fictional description and try again. "
            "No conclusion was made."
        )

    if extraction.ambiguous_statements or extraction.unsupported_statements:
        return (
            "I couldn’t confidently understand part of this fictional description, "
            "so I won’t guess. Please rephrase it using a supported symptom or "
            "remove the unclear detail. No conclusion was made."
        )

    present_symptoms = list(
        dict.fromkeys(
            symptom.name
            for symptom in extraction.state.symptoms
            if not symptom.is_negated
        )
    )
    if not present_symptoms:
        answer = (
            "I couldn’t identify a current supported symptom in this fictional "
            "case, so I can’t offer a useful symptom summary. Please describe "
            "the concern in a few words. No conclusion was made."
        )
        return _append_follow_up(answer, turn)

    selected_library = library or load_guidance_library()
    topics_by_name = {
        topic.id.replace("-", " ").casefold(): topic
        for topic in selected_library.topics
    }
    topics = [
        topics_by_name[name.casefold()]
        for name in present_symptoms
        if name.casefold() in topics_by_name
    ]
    recognized_names = {topic.id.replace("-", " ").casefold() for topic in topics}
    unlisted_symptoms = [
        name for name in present_symptoms if name.casefold() not in recognized_names
    ]
    if not topics:
        answer = (
            "I recognized some details, but I don’t have matching general "
            "information for this symptom. I won’t guess or infer a cause. "
            "Please consult a qualified healthcare professional if you’re concerned."
        )
        return _append_follow_up(answer, turn)

    labels = [topic.label for topic in topics]
    symptom_phrase = (
        ", ".join(labels[:-1]) + f" and {labels[-1]}"
        if len(labels) > 1
        else labels[0]
    )
    comfort_steps = list(
        dict.fromkeys(step for topic in topics for step in topic.comfort_steps)
    )
    seek_help = list(
        dict.fromkeys(
            _remove_leading_phrase(
                item,
                (
                    "Contact a healthcare professional if ",
                    "Seek professional advice promptly for ",
                ),
            )
            for topic in topics
            for item in topic.seek_help
        )
    )
    urgent_signs = list(
        dict.fromkeys(
            _remove_leading_phrase(item, ("Get urgent medical help for ",))
            for topic in topics
            for item in topic.urgent_signs
        )
    )

    sections = [
        f"For this fictional case, you described {symptom_phrase}. "
        "I can’t determine the cause or diagnose from this description.",
    ]
    if turn.session.answers:
        sections.insert(0, "Thanks, I’ve updated the fictional case with your answer.")
    if unlisted_symptoms:
        sections.append(
            "I don’t have matching general information for "
            + ", ".join(unlisted_symptoms)
            + ", so I won’t guess about it."
        )
    sections.append(
        "For general comfort:\n"
        + "\n".join(f"• {step}" for step in comfort_steps)
    )
    sections.append(
        "Contact a healthcare professional if:\n"
        + "\n".join(f"• {item}" for item in seek_help)
    )
    sections.append(
        "Seek urgent help for:\n"
        + "\n".join(f"• {item}" for item in urgent_signs)
    )
    sections.append(
        "This is general educational information, not a diagnosis or treatment "
        "plan. No medicines or dosages are recommended."
    )
    return _append_follow_up("\n\n".join(sections), turn)


def _remove_leading_phrase(text: str, phrases: tuple[str, ...]) -> str:
    for phrase in phrases:
        if text.casefold().startswith(phrase.casefold()):
            return text[len(phrase) :].strip()
    return text


def _append_follow_up(answer: str, turn: AgentTurn) -> str:
    if turn.question_to_user:
        return f"{answer}\n\nOne follow-up question: {turn.question_to_user}"
    return answer
