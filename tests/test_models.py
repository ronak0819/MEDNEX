import pytest
from pydantic import ValidationError

from mednex.models import (
    ExtractedObservation,
    FinalAssessmentReport,
    InvestigationSuggestion,
    MissingInformation,
    PatientInformation,
    ReasoningEvidence,
    Symptom,
    SymptomDuration,
    VitalSigns,
)
from mednex.state import ReasoningState


def test_accepts_synthetic_patient_and_symptom_data() -> None:
    patient = PatientInformation(
        patient_id="synthetic-001",
        age_years=28,
        relevant_history=["Synthetic history for validation."],
    )
    symptom = Symptom(
        name="Headache",
        description="A fictional example symptom.",
        duration=SymptomDuration(amount=2, unit="hours"),
    )

    assert patient.patient_id == "synthetic-001"
    assert symptom.duration is not None
    assert symptom.duration.unit == "hours"


@pytest.mark.parametrize("age", [-1, 126, 150.5])
def test_rejects_invalid_age(age: int | float) -> None:
    with pytest.raises(ValidationError):
        PatientInformation(age_years=age)


def test_rejects_empty_symptom_name() -> None:
    with pytest.raises(ValidationError):
        Symptom(name="   ")


def test_rejects_non_positive_duration() -> None:
    with pytest.raises(ValidationError):
        SymptomDuration(amount=0, unit="days")


def test_rejects_out_of_range_oxygen_saturation() -> None:
    with pytest.raises(ValidationError):
        VitalSigns(oxygen_saturation_percent=101)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_vital_signs(value: float) -> None:
    with pytest.raises(ValidationError):
        VitalSigns(temperature_celsius=value)


def test_requires_both_blood_pressure_values() -> None:
    with pytest.raises(ValidationError, match="must be provided together"):
        VitalSigns(systolic_blood_pressure_mmhg=118)


def test_state_holds_typed_component_data() -> None:
    state = ReasoningState(
        original_description="Synthetic patient description.",
        patient_information=PatientInformation(age_years=34),
        symptoms=[Symptom(name="Cough")],
        vital_signs=VitalSigns(heart_rate_bpm=72),
        extracted_observations=[
            ExtractedObservation(
                name="symptom",
                value="Cough",
                source_text="The fictional patient reports a cough.",
                source_type="patient_description",
            )
        ],
        missing_information=[MissingInformation(field="duration", reason="Not stated.")],
        reasoning_evidence=[
            ReasoningEvidence(
                statement="Synthetic evidence item.",
                source_observation="symptom",
                relationship="contextual",
            )
        ],
        investigation_suggestions=[
            InvestigationSuggestion(
                name="Example investigation",
                educational_purpose="Demonstrate the report data shape.",
            )
        ],
    )

    assert state.patient_information.age_years == 34
    assert state.symptoms[0].name == "Cough"
    assert state.extracted_observations[0].source_type == "patient_description"
    assert state.reasoning_evidence[0].relationship == "contextual"


def test_final_report_uses_typed_nested_entries() -> None:
    report = FinalAssessmentReport(
        summary="Synthetic educational summary.",
        missing_information=[MissingInformation(field="age", reason="Not supplied.")],
        investigation_suggestions=[
            InvestigationSuggestion(
                name="Example investigation",
                educational_purpose="Illustrate a structured suggestion.",
            )
        ],
    )

    assert report.missing_information[0].field == "age"
    assert "Educational simulation only" in report.educational_notice
