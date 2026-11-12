from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import CONFIG

NonEmptyText = Annotated[str, Field(min_length=1)]


class MEDNEXModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PatientInformation(MEDNEXModel):
    patient_id: NonEmptyText | None = None
    age_years: int | None = Field(default=None, ge=0, le=125)
    sex: NonEmptyText | None = None
    relevant_history: list[NonEmptyText] = Field(default_factory=list)


class SymptomDuration(MEDNEXModel):
    amount: float = Field(gt=0)
    unit: Literal["minutes", "hours", "days", "weeks", "months", "years"]


class Symptom(MEDNEXModel):
    name: NonEmptyText
    description: NonEmptyText | None = None
    location: NonEmptyText | None = None
    duration: SymptomDuration | None = None
    is_negated: bool = False


class VitalSigns(MEDNEXModel):
    temperature_celsius: float | None = Field(default=None, allow_inf_nan=False)
    heart_rate_bpm: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    respiratory_rate_per_minute: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    systolic_blood_pressure_mmhg: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    diastolic_blood_pressure_mmhg: float | None = Field(
        default=None, gt=0, allow_inf_nan=False
    )
    oxygen_saturation_percent: float | None = Field(
        default=None, ge=0, le=100, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def validate_blood_pressure_pair(self) -> VitalSigns:
        has_systolic = self.systolic_blood_pressure_mmhg is not None
        has_diastolic = self.diastolic_blood_pressure_mmhg is not None
        if has_systolic != has_diastolic:
            raise ValueError("Systolic and diastolic blood pressure must be provided together.")
        return self


class ExtractedObservation(MEDNEXModel):
    name: NonEmptyText
    value: NonEmptyText
    source_text: NonEmptyText
    source_type: Literal["patient_description", "structured_input"]
    is_negated: bool = False


class MissingInformation(MEDNEXModel):
    field: NonEmptyText
    reason: NonEmptyText


class ReasoningEvidence(MEDNEXModel):
    statement: NonEmptyText
    source_observation: NonEmptyText
    relationship: Literal["supporting", "contradicting", "contextual"]


class InvestigationSuggestion(MEDNEXModel):
    name: NonEmptyText
    educational_purpose: NonEmptyText


class FinalAssessmentReport(MEDNEXModel):
    summary: NonEmptyText
    evidence: list[ReasoningEvidence] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    investigation_suggestions: list[InvestigationSuggestion] = Field(default_factory=list)
    educational_notice: NonEmptyText = CONFIG.educational_notice
