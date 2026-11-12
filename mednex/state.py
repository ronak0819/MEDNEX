from pydantic import Field

from mednex.models import (
    ExtractedObservation,
    FinalAssessmentReport,
    InvestigationSuggestion,
    MEDNEXModel,
    MissingInformation,
    PatientInformation,
    ReasoningEvidence,
    Symptom,
    VitalSigns,
)


class ReasoningState(MEDNEXModel):
    """Typed data shared between MEDNEX components."""

    original_description: str = ""
    patient_information: PatientInformation = Field(default_factory=PatientInformation)
    symptoms: list[Symptom] = Field(default_factory=list)
    vital_signs: VitalSigns | None = None
    extracted_observations: list[ExtractedObservation] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    reasoning_evidence: list[ReasoningEvidence] = Field(default_factory=list)
    investigation_suggestions: list[InvestigationSuggestion] = Field(default_factory=list)
    final_report: FinalAssessmentReport | None = None


class ExtractionResult(MEDNEXModel):
    state: ReasoningState
    extraction_confidence: float = Field(
        ge=0,
        le=1,
        description="Deterministic extraction clarity indicator, not clinical confidence.",
    )
    ambiguous_statements: list[str] = Field(default_factory=list)
    unsupported_statements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
