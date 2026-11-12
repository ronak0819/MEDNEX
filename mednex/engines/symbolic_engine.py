from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from mednex.models import ExtractedObservation, MEDNEXModel, Symptom
from mednex.state import ExtractionResult

FactType = Literal["symptom", "observation", "evidence_flag"]
ConditionOperator = Literal[
    "present",
    "negated",
    "equals",
    "greater_than",
    "at_least",
    "less_than",
    "at_most",
]
EvidencePolarity = Literal["supporting", "conflicting", "contextual"]
RuleStatus = Literal["fired", "blocked_missing_information", "conditions_not_met"]
DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "data" / "symbolic_rules.json"


class RuleCondition(MEDNEXModel):
    fact_type: FactType
    key: str = Field(min_length=1)
    operator: ConditionOperator
    value: str | float | None = None

    @model_validator(mode="after")
    def validate_condition_shape(self) -> RuleCondition:
        if self.fact_type == "evidence_flag":
            if self.operator != "present" or self.value is not None:
                raise ValueError("Evidence-flag conditions only support 'present' without a value.")
        elif self.fact_type == "symptom":
            if self.operator not in {"present", "negated"} or self.value is not None:
                raise ValueError("Symptom conditions support 'present' or 'negated' only.")
        elif self.operator in {
            "equals",
            "greater_than",
            "at_least",
            "less_than",
            "at_most",
        } and self.value is None:
            raise ValueError(f"Observation operator '{self.operator}' requires a value.")
        elif self.operator == "present" and self.value is not None:
            raise ValueError("Observation 'present' conditions do not take a value.")
        elif self.operator in {
            "greater_than",
            "at_least",
            "less_than",
            "at_most",
        }:
            try:
                float(self.value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Observation operator '{self.operator}' requires a numeric value."
                ) from error
        return self


class ProductionRule(MEDNEXModel):
    id: str = Field(min_length=1)
    priority: int
    conditions: list[RuleCondition] = Field(min_length=1)
    assessment_signal: str = Field(min_length=1)
    polarity: EvidencePolarity
    evidence_flag: str = Field(min_length=1)
    explanation: str = Field(min_length=1)


class RuleSet(MEDNEXModel):
    description: str = Field(min_length=1)
    rules: list[ProductionRule]

    @model_validator(mode="after")
    def validate_unique_rule_ids(self) -> RuleSet:
        rule_ids = [rule.id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Production rule ids must be unique.")
        return self


class ConditionTrace(MEDNEXModel):
    fact_type: FactType
    key: str
    operator: ConditionOperator
    expected_value: str | float | None = None
    status: Literal["matched", "missing", "not_matched"]
    explanation: str


class RuleTrace(MEDNEXModel):
    rule_id: str
    priority: int
    status: RuleStatus
    conditions: list[ConditionTrace]
    explanation: str


class FiredRule(MEDNEXModel):
    rule_id: str
    priority: int
    assessment_signal: str
    polarity: EvidencePolarity
    evidence_flag: str
    explanation: str
    matched_conditions: list[ConditionTrace]


class CandidateAssessmentSignal(MEDNEXModel):
    assessment_signal: str
    supporting_evidence: list[FiredRule] = Field(default_factory=list)
    conflicting_evidence: list[FiredRule] = Field(default_factory=list)
    contextual_evidence: list[FiredRule] = Field(default_factory=list)
    has_conflicting_evidence: bool = False


class RuleConflict(MEDNEXModel):
    assessment_signal: str
    rule_ids: list[str]
    explanation: str


class SymbolicReasoningResult(MEDNEXModel):
    fired_rules: list[FiredRule] = Field(default_factory=list)
    candidate_signals: list[CandidateAssessmentSignal] = Field(default_factory=list)
    rule_trace: list[RuleTrace] = Field(default_factory=list)
    conflicts: list[RuleConflict] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    explanation: str = (
        "Signals and flags are outputs of synthetic educational production rules. "
        "They are not diagnoses or clinically validated assessments."
    )


def _normalize_value(value: str | float) -> str:
    return str(value).strip().casefold()


def _matches_observation_value(
    observation: ExtractedObservation,
    condition: RuleCondition,
) -> bool:
    if condition.operator == "negated":
        if not observation.is_negated:
            return False
        return (
            condition.value is None
            or _normalize_value(condition.value) in _normalize_value(observation.value)
        )
    if observation.is_negated:
        return False
    if condition.operator == "present":
        return True
    if condition.operator == "equals":
        return _normalize_value(observation.value) == _normalize_value(condition.value or "")
    try:
        observed_number = float(observation.value)
        expected_number = float(condition.value)  # validated when rules load
    except (TypeError, ValueError):
        return False
    if condition.operator == "greater_than":
        return observed_number > expected_number
    if condition.operator == "at_least":
        return observed_number >= expected_number
    if condition.operator == "less_than":
        return observed_number < expected_number
    if condition.operator == "at_most":
        return observed_number <= expected_number
    return False


class SymbolicEngine:
    """Deterministic forward-chaining engine for explicitly synthetic rules."""

    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset

    @classmethod
    def load(cls, path: str | Path | None = None) -> SymbolicEngine:
        rules_path = Path(path) if path is not None else DEFAULT_RULES_PATH
        with rules_path.open("r", encoding="utf-8") as rules_file:
            raw_ruleset = json.load(rules_file)
        return cls(RuleSet.model_validate(raw_ruleset))

    def _evaluate_condition(
        self,
        condition: RuleCondition,
        extraction: ExtractionResult,
        derived_flags: set[str],
    ) -> ConditionTrace:
        if condition.fact_type == "symptom":
            matching_symptoms = [
                symptom
                for symptom in extraction.state.symptoms
                if symptom.name.casefold() == condition.key.casefold()
            ]
            if not matching_symptoms:
                return ConditionTrace(
                    fact_type=condition.fact_type,
                    key=condition.key,
                    operator=condition.operator,
                    status="missing",
                    explanation=f"Symptom {condition.key!r} was not mentioned.",
                )
            expected_negated = condition.operator == "negated"
            matched = any(
                symptom.is_negated == expected_negated
                for symptom in matching_symptoms
            )
            return ConditionTrace(
                fact_type=condition.fact_type,
                key=condition.key,
                operator=condition.operator,
                status="matched" if matched else "not_matched",
                explanation=(
                    f"Explicit symptom record for {condition.key!r} "
                    f"{'matches' if matched else 'does not match'} the requested polarity."
                ),
            )

        if condition.fact_type == "evidence_flag":
            matched = condition.key in derived_flags
            return ConditionTrace(
                fact_type=condition.fact_type,
                key=condition.key,
                operator=condition.operator,
                status="matched" if matched else "missing",
                explanation=(
                    f"Derived evidence flag {condition.key!r} "
                    f"{'is available' if matched else 'has not been derived yet'}."
                ),
            )

        matching_observations = [
            observation
            for observation in extraction.state.extracted_observations
            if observation.name.casefold() == condition.key.casefold()
        ]
        if condition.operator == "negated" and condition.value is not None:
            matching_observations = [
                observation
                for observation in matching_observations
                if _normalize_value(condition.value)
                in _normalize_value(observation.value)
            ]
        if not matching_observations and condition.key in {
            "temperature_celsius",
            "heart_rate_bpm",
            "respiratory_rate_per_minute",
            "oxygen_saturation_percent",
            "systolic_blood_pressure_mmhg",
            "diastolic_blood_pressure_mmhg",
        } and extraction.state.vital_signs is not None:
            value = getattr(extraction.state.vital_signs, condition.key)
            if value is not None:
                matching_observations = [
                    ExtractedObservation(
                        name=condition.key,
                        value=str(value),
                        source_text="Structured vital-sign state",
                        source_type="structured_input",
                    )
                ]
        if not matching_observations:
            return ConditionTrace(
                fact_type=condition.fact_type,
                key=condition.key,
                operator=condition.operator,
                expected_value=condition.value,
                status="missing",
                explanation=f"Observation {condition.key!r} was not supplied.",
            )
        matched = any(
            _matches_observation_value(observation, condition)
            for observation in matching_observations
        )
        return ConditionTrace(
            fact_type=condition.fact_type,
            key=condition.key,
            operator=condition.operator,
            expected_value=condition.value,
            status="matched" if matched else "not_matched",
            explanation=(
                f"Observation {condition.key!r} "
                f"{'matches' if matched else 'does not match'} the rule condition."
            ),
        )

    def reason(self, extraction: ExtractionResult) -> SymbolicReasoningResult:
        ordered_rules = sorted(
            self.ruleset.rules, key=lambda rule: (-rule.priority, rule.id)
        )
        derived_flags: set[str] = set()
        fired_rule_ids: set[str] = set()
        fired_rules: list[FiredRule] = []
        latest_traces: dict[str, RuleTrace] = {}

        changed = True
        while changed:
            changed = False
            for rule in ordered_rules:
                if rule.id in fired_rule_ids:
                    continue
                condition_traces = [
                    self._evaluate_condition(condition, extraction, derived_flags)
                    for condition in rule.conditions
                ]
                missing = [
                    item for item in condition_traces if item.status == "missing"
                ]
                unmet = [
                    item for item in condition_traces if item.status == "not_matched"
                ]
                if missing:
                    status: RuleStatus = "blocked_missing_information"
                elif unmet:
                    status = "conditions_not_met"
                else:
                    status = "fired"
                    fired_rule_ids.add(rule.id)
                    derived_flags.add(rule.evidence_flag)
                    fired_rules.append(
                        FiredRule(
                            rule_id=rule.id,
                            priority=rule.priority,
                            assessment_signal=rule.assessment_signal,
                            polarity=rule.polarity,
                            evidence_flag=rule.evidence_flag,
                            explanation=rule.explanation,
                            matched_conditions=condition_traces,
                        )
                    )
                    changed = True

                latest_traces[rule.id] = RuleTrace(
                    rule_id=rule.id,
                    priority=rule.priority,
                    status=status,
                    conditions=condition_traces,
                    explanation=rule.explanation,
                )

        fired_rules.sort(key=lambda item: (-item.priority, item.rule_id))
        signals: dict[str, list[FiredRule]] = defaultdict(list)
        for fired_rule in fired_rules:
            signals[fired_rule.assessment_signal].append(fired_rule)

        candidates: list[CandidateAssessmentSignal] = []
        conflicts: list[RuleConflict] = []
        for signal_name in sorted(signals):
            signal_rules = signals[signal_name]
            supporting = [rule for rule in signal_rules if rule.polarity == "supporting"]
            conflicting = [rule for rule in signal_rules if rule.polarity == "conflicting"]
            contextual = [rule for rule in signal_rules if rule.polarity == "contextual"]
            has_conflict = bool(supporting and conflicting)
            candidates.append(
                CandidateAssessmentSignal(
                    assessment_signal=signal_name,
                    supporting_evidence=supporting,
                    conflicting_evidence=conflicting,
                    contextual_evidence=contextual,
                    has_conflicting_evidence=has_conflict,
                )
            )
            if has_conflict:
                conflicts.append(
                    RuleConflict(
                        assessment_signal=signal_name,
                        rule_ids=sorted(
                            rule.rule_id for rule in supporting + conflicting
                        ),
                        explanation=(
                            "Fired synthetic rules provide both supporting and "
                            "conflicting evidence for the same assessment signal."
                        ),
                    )
                )

        traces = [
            latest_traces[rule.id]
            for rule in ordered_rules
            if rule.id in latest_traces
        ]
        missing_information = sorted(
            {
                condition.explanation
                for trace in traces
                if trace.status == "blocked_missing_information"
                for condition in trace.conditions
                if condition.status == "missing"
            }
        )
        return SymbolicReasoningResult(
            fired_rules=fired_rules,
            candidate_signals=candidates,
            rule_trace=traces,
            conflicts=conflicts,
            missing_information=missing_information,
        )
