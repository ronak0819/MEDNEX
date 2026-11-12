from pydantic import Field

from mednex.models import MEDNEXModel
from mednex.state import ExtractionResult

MAX_DESCRIPTION_LENGTH = 4000


class InputValidationResult(MEDNEXModel):
    valid: bool
    issues: list[str] = Field(default_factory=list)


def validate_case_input(text: str, extraction: ExtractionResult) -> InputValidationResult:
    issues: list[str] = []
    if not text.strip():
        issues.append("Description is empty.")
    if len(text) > MAX_DESCRIPTION_LENGTH:
        issues.append(f"Description exceeds the {MAX_DESCRIPTION_LENGTH}-character limit.")
    if extraction.ambiguous_statements:
        issues.append("Some statements could not be interpreted without guessing.")
    if extraction.unsupported_statements:
        issues.append("Some statements use concepts outside the supported vocabulary.")
    if extraction.contradictions:
        issues.append("The description contains contradictory statements.")
    if not extraction.state.symptoms and not extraction.state.extracted_observations:
        issues.append("No supported observations were extracted.")
    has_supported_facts = bool(
        extraction.state.symptoms or extraction.state.extracted_observations
    )
    return InputValidationResult(
        valid=bool(text.strip()) and len(text) <= MAX_DESCRIPTION_LENGTH and has_supported_facts,
        issues=issues,
    )
