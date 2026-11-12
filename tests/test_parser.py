from mednex.nlp import parse_patient_description
from mednex.nlp.normalizer import normalize_text, tokenize


def test_extracts_demographics_symptoms_duration_and_vitals() -> None:
    result = parse_patient_description(
        "A 45-year-old male has dry cough and fever for three days. "
        "His temperature is 38.5 degrees Celsius and oxygen saturation is 96 percent."
    )

    assert result.state.patient_information.age_years == 45
    assert result.state.patient_information.sex == "male"
    assert [symptom.name for symptom in result.state.symptoms] == ["cough", "fever"]
    assert all(symptom.duration is not None for symptom in result.state.symptoms)
    assert result.state.symptoms[0].duration.amount == 3
    assert result.state.vital_signs.temperature_celsius == 38.5
    assert result.state.vital_signs.oxygen_saturation_percent == 96
    assert result.extraction_confidence == 1


def test_handles_incomplete_description_without_filling_missing_facts() -> None:
    result = parse_patient_description("The patient reports a headache.")

    assert result.state.patient_information.age_years is None
    assert result.state.patient_information.sex is None
    assert result.state.vital_signs.temperature_celsius is None
    assert len(result.state.symptoms) == 1
    assert result.state.symptoms[0].duration is None


def test_marks_negated_symptoms_as_absent() -> None:
    result = parse_patient_description(
        "The patient has no fever and does not have a cough."
    )

    assert [(symptom.name, symptom.is_negated) for symptom in result.state.symptoms] == [
        ("fever", True),
        ("cough", True),
    ]


def test_reports_contradictory_symptoms_and_vitals() -> None:
    result = parse_patient_description(
        "The patient has fever but no fever. Temperature is 38 C and temperature is 40 C."
    )

    assert result.contradictions
    assert any("fever" in item for item in result.contradictions)
    assert any("temperature_celsius" in item for item in result.contradictions)
    assert result.state.vital_signs.temperature_celsius is None


def test_does_not_assert_hedged_or_unsupported_symptoms() -> None:
    result = parse_patient_description(
        "The patient may have a cough and reports blurry vision."
    )

    assert result.state.symptoms == []
    assert result.ambiguous_statements
    assert result.unsupported_statements


def test_normalizes_aliases_units_and_additional_supported_vitals() -> None:
    result = parse_patient_description(
        "High temperature and difficulty breathing for 2 hrs. "
        "Temp 101.3 F, pulse 72 bpm, respiratory rate 18 breaths per minute, "
        "O2 sat 96%, blood pressure 120/80 mmHg."
    )

    assert [symptom.name for symptom in result.state.symptoms] == [
        "fever",
        "shortness of breath",
    ]
    assert all(symptom.duration.unit == "hours" for symptom in result.state.symptoms)
    assert round(result.state.vital_signs.temperature_celsius, 1) == 38.5
    assert result.state.vital_signs.heart_rate_bpm == 72
    assert result.state.vital_signs.respiratory_rate_per_minute == 18
    assert result.state.vital_signs.oxygen_saturation_percent == 96
    assert result.state.vital_signs.systolic_blood_pressure_mmhg == 120
    assert not result.ambiguous_statements


def test_extracts_labeled_age() -> None:
    result = parse_patient_description("Age: 42. The patient reports nausea.")

    assert result.state.patient_information.age_years == 42


def test_does_not_extract_value_with_unrecognized_or_missing_unit() -> None:
    result = parse_patient_description("Temperature is 38.5 and oxygen saturation is 96.")

    assert result.state.vital_signs.temperature_celsius is None
    assert result.state.vital_signs.oxygen_saturation_percent is None
    assert result.ambiguous_statements


def test_marks_numeric_overflow_ambiguous_without_raising() -> None:
    result = parse_patient_description(
        f"Temperature is {'9' * 400} C."
    )

    assert result.state.vital_signs.temperature_celsius is None
    assert result.ambiguous_statements


def test_identifies_disagreement_in_explicit_demographics() -> None:
    result = parse_patient_description("A 32-year-old male, female patient.")

    assert result.state.patient_information.sex is None
    assert any("sex" in item for item in result.contradictions)


def test_normalizer_and_tokenizer_are_deterministic() -> None:
    assert normalize_text("High temperature, difficulty breathing; pulse") == (
        "fever, shortness of breath; heart rate"
    )
    assert tokenize("Pulse is 72 bpm") == ["heart", "rate", "is", "72", "bpm"]
