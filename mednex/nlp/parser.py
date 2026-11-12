from __future__ import annotations

import math
import re
from collections import defaultdict

from mednex.models import (
    ExtractedObservation,
    PatientInformation,
    Symptom,
    SymptomDuration,
    VitalSigns,
)
from mednex.nlp.normalizer import parse_number, tokenize
from mednex.nlp.patterns import (
    BLOOD_PRESSURE_PATTERN,
    DURATION_PATTERN,
    HEDGE_PATTERN,
    HEART_RATE_PATTERN,
    NEGATION_PATTERN,
    OXYGEN_PATTERN,
    RESPIRATORY_RATE_PATTERN,
    SYMPTOM_PATTERNS,
    TEMPERATURE_PATTERN,
    VITAL_LABELS,
)
from mednex.state import ExtractionResult, ReasoningState

AGE_PATTERN = re.compile(
    r"(?:\b(?P<age>\d{1,3})\s*(?:-|\s)?\s*(?:year|yr)s?\s*[- ]?old\b|"
    r"\bage\s*(?:is|of|:)?\s*(?P<labeled_age>\d{1,3})\b)",
    re.IGNORECASE,
)
SEX_PATTERN = re.compile(r"\b(?P<sex>female|male|woman|man|girl|boy)\b", re.IGNORECASE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"[!?]+|\.(?!\d)(?=\s|$)|\n+")
CLAUSE_CUE_PATTERN = re.compile(
    r"\b(?:has|have|reports?|experiences?|complains of|with|feels?)\b",
    re.IGNORECASE,
)
NEGATION_BREAK_PATTERN = re.compile(r"\b(?:but|however|although)\b", re.IGNORECASE)

VITAL_PATTERNS = {
    "temperature_celsius": TEMPERATURE_PATTERN,
    "heart_rate_bpm": HEART_RATE_PATTERN,
    "respiratory_rate_per_minute": RESPIRATORY_RATE_PATTERN,
    "oxygen_saturation_percent": OXYGEN_PATTERN,
}
VITAL_LABEL_FIELDS = {
    "temperature": "temperature_celsius",
    "heart_rate": "heart_rate_bpm",
    "respiratory_rate": "respiratory_rate_per_minute",
    "oxygen_saturation": "oxygen_saturation_percent",
}


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for boundary in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        if text[start : boundary.start()].strip():
            spans.append((start, boundary.end(), text[start : boundary.end()]))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text), text[start:]))
    return spans


def _duration(sentence: str) -> tuple[SymptomDuration | None, int]:
    matches = list(DURATION_PATTERN.finditer(sentence))
    if len(matches) != 1:
        return None, len(matches)
    match = matches[0]
    amount = parse_number(match.group("amount"))
    if amount is None or amount <= 0:
        return None, 1
    raw_unit = match.group("unit").lower()
    if raw_unit.startswith("min"):
        unit = "minutes"
    elif raw_unit.startswith("h") or raw_unit.startswith("hr"):
        unit = "hours"
    elif raw_unit.startswith("d"):
        unit = "days"
    elif raw_unit.startswith("w"):
        unit = "weeks"
    elif raw_unit.startswith("m"):
        unit = "months"
    else:
        unit = "years"
    return SymptomDuration(amount=amount, unit=unit), 1


def _is_negated(sentence: str, start: int) -> bool:
    preceding = sentence[max(0, start - 60) : start]
    last_break = list(NEGATION_BREAK_PATTERN.finditer(preceding))
    if last_break:
        preceding = preceding[last_break[-1].end() :]
    return NEGATION_PATTERN.search(preceding) is not None


def _is_hedged(sentence: str, start: int) -> bool:
    preceding = sentence[max(0, start - 45) : start]
    return HEDGE_PATTERN.search(preceding) is not None


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _check_unknown_symptom_clauses(sentence: str, matched_spans: list[tuple[int, int]]) -> list[str]:
    unsupported: list[str] = []
    pieces = re.split(r"\b(?:and|but|or)\b|[,;]", sentence, flags=re.IGNORECASE)
    offset = 0
    for piece in pieces:
        piece_start = sentence.find(piece, offset)
        if piece_start < 0:
            piece_start = offset
        piece_end = piece_start + len(piece)
        offset = piece_end
        if not CLAUSE_CUE_PATTERN.search(piece):
            continue
        if not tokenize(piece):
            continue
        if any(start < piece_end and end > piece_start for start, end in matched_spans):
            continue
        if re.search(r"\b(?:temperature|temp|heart rate|pulse|respiratory rate|"
                     r"breathing rate|oxygen saturation|spo2|o2 sat|blood pressure|bp)\b",
                     piece, re.IGNORECASE):
            continue
        candidate = piece.strip()
        if candidate and candidate not in unsupported:
            unsupported.append(candidate)
    return unsupported


def parse_patient_description(text: str) -> ExtractionResult:
    """Extract explicitly stated facts with deterministic patterns only.

    The confidence value describes extraction clarity only. It is not a
    probability, diagnosis, or measure of clinical correctness.
    """
    if not isinstance(text, str):
        raise TypeError("Patient description must be a string.")

    patient_facts: dict[str, set[object]] = defaultdict(set)
    symptom_entries: list[Symptom] = []
    observation_entries: list[ExtractedObservation] = []
    value_mentions: dict[str, list[tuple[float, str]]] = defaultdict(list)
    pressure_mentions: list[tuple[float, float, str]] = []
    ambiguities: list[str] = []
    unsupported: list[str] = []
    contradictions: list[str] = []

    def add_unique(target: list[str], statement: str) -> None:
        statement = statement.strip()
        if statement and statement not in target:
            target.append(statement)

    for _sentence_start, _sentence_end, sentence in _sentence_spans(text):
        age_matches = list(AGE_PATTERN.finditer(sentence))
        for match in age_matches:
            age = int(match.group("age") or match.group("labeled_age"))
            if age > 125:
                add_unique(ambiguities, sentence)
            else:
                patient_facts["age"].add(age)
                observation_entries.append(
                    ExtractedObservation(
                        name="age_years",
                        value=str(age),
                        source_text=match.group(),
                        source_type="patient_description",
                    )
                )

        for match in SEX_PATTERN.finditer(sentence):
            raw_sex = match.group("sex").lower()
            sex = "female" if raw_sex in {"female", "woman", "girl"} else "male"
            patient_facts["sex"].add(sex)
            observation_entries.append(
                ExtractedObservation(
                    name="sex",
                    value=sex,
                    source_text=match.group(),
                    source_type="patient_description",
                )
            )

        duration, duration_count = _duration(sentence)
        if duration_count > 1:
            add_unique(ambiguities, sentence)

        symptom_spans: list[tuple[int, int]] = []
        sentence_symptoms: list[tuple[int, Symptom, str]] = []
        for canonical, pattern in SYMPTOM_PATTERNS:
            for match in pattern.finditer(sentence):
                if any(start < match.end() and end > match.start() for start, end in symptom_spans):
                    continue
                symptom_spans.append((match.start(), match.end()))
                if _is_hedged(sentence, match.start()):
                    add_unique(ambiguities, sentence)
                    continue
                is_negated = _is_negated(sentence, match.start())
                symptom = Symptom(
                    name=canonical,
                    duration=duration,
                    is_negated=is_negated,
                )
                sentence_symptoms.append((match.start(), symptom, match.group()))

        for _start, symptom, source in sorted(sentence_symptoms):
            symptom_entries.append(symptom)
            observation_entries.append(
                ExtractedObservation(
                    name="symptom",
                    value=f"{'negated' if symptom.is_negated else 'present'}: {symptom.name}",
                    source_text=source,
                    source_type="patient_description",
                    is_negated=symptom.is_negated,
                )
            )

        for clause in _check_unknown_symptom_clauses(sentence, symptom_spans):
            add_unique(unsupported, clause)

        for label, pattern in VITAL_LABELS.items():
            if pattern.search(sentence):
                if label == "temperature":
                    has_adjacent_numeric_value = re.search(
                        r"\b(?:temperature|temp)\s*(?:is|of|at|was|=|:)?\s*\d",
                        sentence,
                        re.IGNORECASE,
                    )
                    if not has_adjacent_numeric_value:
                        continue
                if label == "blood_pressure":
                    has_value = BLOOD_PRESSURE_PATTERN.search(sentence) is not None
                else:
                    has_value = VITAL_PATTERNS[VITAL_LABEL_FIELDS[label]].search(sentence) is not None
                if not has_value:
                    add_unique(ambiguities, sentence)

        for field, pattern in VITAL_PATTERNS.items():
            for match in pattern.finditer(sentence):
                value = float(match.group("value"))
                source = match.group()
                if not math.isfinite(value):
                    add_unique(ambiguities, source)
                    continue
                if field == "oxygen_saturation_percent" and not 0 <= value <= 100:
                    add_unique(ambiguities, source)
                    continue
                if field in {"heart_rate_bpm", "respiratory_rate_per_minute"} and value <= 0:
                    add_unique(ambiguities, source)
                    continue
                if field == "temperature_celsius":
                    unit = match.group("unit").lower().replace(" ", "")
                    if "f" in unit or "fahrenheit" in unit:
                        value = (value - 32) * 5 / 9
                    value_mentions[field].append((value, source))
                else:
                    value_mentions[field].append((value, source))

        for match in BLOOD_PRESSURE_PATTERN.finditer(sentence):
            systolic = float(match.group("systolic"))
            diastolic = float(match.group("diastolic"))
            pressure_mentions.append((systolic, diastolic, match.group()))

        if duration is not None and not symptom_spans:
            observation_entries.append(
                ExtractedObservation(
                    name="symptom_duration",
                    value=f"{_format_number(duration.amount)} {duration.unit}",
                    source_text=sentence.strip(),
                    source_type="patient_description",
                )
            )

    for field in ("age", "sex"):
        if len(patient_facts[field]) > 1:
            contradictions.append(f"Conflicting explicitly stated patient {field}: {sorted(patient_facts[field])}.")
    if len(patient_facts["age"]) > 1:
        patient_facts["age"].clear()
    if len(patient_facts["sex"]) > 1:
        patient_facts["sex"].clear()

    symptom_polarities: dict[str, set[bool]] = defaultdict(set)
    for symptom in symptom_entries:
        symptom_polarities[symptom.name].add(symptom.is_negated)
    for name, polarities in symptom_polarities.items():
        if len(polarities) > 1:
            contradictions.append(f"Both presence and absence of '{name}' are stated.")

    vital_values: dict[str, float | None] = {}
    for field, mentions in value_mentions.items():
        distinct = {round(value, 6) for value, _source in mentions}
        if len(distinct) > 1:
            contradictions.append(f"Conflicting values stated for {field}: {sorted(distinct)}.")
            vital_values[field] = None
            for value, source in mentions:
                observation_entries.append(
                    ExtractedObservation(
                        name=field,
                        value=_format_number(value),
                        source_text=source,
                        source_type="patient_description",
                    )
                )
        elif mentions:
            value = mentions[0][0]
            vital_values[field] = value
            observation_entries.append(
                ExtractedObservation(
                    name=field,
                    value=_format_number(value),
                    source_text=mentions[0][1],
                    source_type="patient_description",
                )
            )

    distinct_pressures = {(systolic, diastolic) for systolic, diastolic, _ in pressure_mentions}
    valid_pressure_mentions = []
    for systolic, diastolic, source in pressure_mentions:
        if systolic <= 0 or diastolic <= 0:
            add_unique(ambiguities, source)
        else:
            valid_pressure_mentions.append((systolic, diastolic, source))
    distinct_pressures = {
        (systolic, diastolic)
        for systolic, diastolic, _source in valid_pressure_mentions
    }
    if len(distinct_pressures) > 1:
        contradictions.append(f"Conflicting blood pressure values stated: {sorted(distinct_pressures)}.")
        pressure = None
        for systolic, diastolic, source in valid_pressure_mentions:
            observation_entries.append(
                ExtractedObservation(
                    name="blood_pressure_mmhg",
                    value=f"{_format_number(systolic)}/{_format_number(diastolic)}",
                    source_text=source,
                    source_type="patient_description",
                )
            )
    elif valid_pressure_mentions:
        systolic, diastolic, source = valid_pressure_mentions[0]
        pressure = (systolic, diastolic)
        observation_entries.append(
            ExtractedObservation(
                name="blood_pressure_mmhg",
                value=f"{_format_number(systolic)}/{_format_number(diastolic)}",
                source_text=source,
                source_type="patient_description",
            )
        )
    else:
        pressure = None

    vital_signs = VitalSigns(
        temperature_celsius=vital_values.get("temperature_celsius"),
        heart_rate_bpm=vital_values.get("heart_rate_bpm"),
        respiratory_rate_per_minute=vital_values.get("respiratory_rate_per_minute"),
        oxygen_saturation_percent=vital_values.get("oxygen_saturation_percent"),
        systolic_blood_pressure_mmhg=pressure[0] if pressure else None,
        diastolic_blood_pressure_mmhg=pressure[1] if pressure else None,
    )

    state = ReasoningState(
        original_description=text,
        patient_information=PatientInformation(
            age_years=next(iter(patient_facts["age"]), None),
            sex=next(iter(patient_facts["sex"]), None),
        ),
        symptoms=symptom_entries,
        vital_signs=vital_signs,
        extracted_observations=observation_entries,
    )
    extracted_count = (
        len(symptom_entries)
        + len(patient_facts["age"])
        + len(patient_facts["sex"])
        + sum(value is not None for value in vital_values.values())
        + (1 if pressure else 0)
    )
    issue_count = len(ambiguities) + len(unsupported) + len(contradictions)
    confidence = (
        extracted_count / (extracted_count + 2 * issue_count)
        if extracted_count + issue_count
        else 0.0
    )
    return ExtractionResult(
        state=state,
        extraction_confidence=confidence,
        ambiguous_statements=ambiguities,
        unsupported_statements=unsupported,
        contradictions=contradictions,
    )
