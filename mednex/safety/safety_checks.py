import re

from pydantic import Field

from mednex.models import MEDNEXModel

PERSONAL_HEALTH_PATTERN = re.compile(
    r"\b(?:i have|i am experiencing|my symptoms|my condition|for myself|"
    r"this is about me|my child|my family member)\b",
    re.IGNORECASE,
)
DIRECT_CARE_REQUEST_PATTERN = re.compile(
    r"\b(?:what (?:should|can) i (?:take|do)|prescribe|prescription|"
    r"dosage|dose of|treatment for me|should i stop taking)\b",
    re.IGNORECASE,
)
URGENT_LANGUAGE_PATTERN = re.compile(
    r"\b(?:emergency|can't breathe|cannot breathe|difficulty breathing|"
    r"shortness of breath|breathlessness|not breathing|unconscious|overdose|"
    r"severe chest pain|chest pain|sudden severe (?:stomach|abdominal) pain|"
    r"severe (?:stomach|abdominal) pain|coughing up blood|blue or grey lips|"
    r"immediate danger)\b",
    re.IGNORECASE,
)
URGENT_NEGATION_PATTERN = re.compile(
    r"\b(?:no|not|without|denies?|don't have|do not have|doesn't have|"
    r"does not have)\b[^.!?;]{0,45}$",
    re.IGNORECASE,
)
CLAUSE_CONTRAST_PATTERN = re.compile(r"\b(?:but|however|although)\b", re.IGNORECASE)


def _is_negated_urgent_match(text: str, match_start: int) -> bool:
    preceding = text[max(0, match_start - 50) : match_start]
    contrast_matches = list(CLAUSE_CONTRAST_PATTERN.finditer(preceding))
    if contrast_matches:
        preceding = preceding[contrast_matches[-1].end() :]
    return URGENT_NEGATION_PATTERN.search(preceding) is not None


class SafetyReview(MEDNEXModel):
    professional_assessment_required: bool
    urgent_language_detected: bool
    reasons: list[str] = Field(default_factory=list)
    user_message: str | None = None


def review_input(text: str) -> SafetyReview:
    personal = PERSONAL_HEALTH_PATTERN.search(text) is not None
    direct_care_request = DIRECT_CARE_REQUEST_PATTERN.search(text) is not None
    urgent = any(
        not _is_negated_urgent_match(text, match.start())
        for match in URGENT_LANGUAGE_PATTERN.finditer(text)
    )
    reasons: list[str] = []
    if personal:
        reasons.append("Input appears to describe a real person's health.")
    if direct_care_request:
        reasons.append("The input requests individualized care guidance.")
    if urgent:
        reasons.append("The input contains explicitly urgent language.")
    required = personal or direct_care_request or urgent
    message = None
    if required:
        message = (
            "MEDNEX is an educational simulation and cannot evaluate real or urgent "
            "health concerns or provide individualized treatment. Please contact a "
            "qualified health professional; if there may be immediate danger, contact "
            "local emergency services."
        )
    return SafetyReview(
        professional_assessment_required=required,
        urgent_language_detected=urgent,
        reasons=reasons,
        user_message=message,
    )
