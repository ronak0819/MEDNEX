import pytest
from pydantic import ValidationError

from mednex.engines.symbolic_engine import (
    ProductionRule,
    RuleCondition,
    RuleSet,
    SymbolicEngine,
)
from mednex.models import ExtractedObservation, Symptom, VitalSigns
from mednex.nlp import parse_patient_description
from mednex.state import ExtractionResult, ReasoningState


def test_fires_rule_when_all_explicit_conditions_match() -> None:
    extraction = parse_patient_description("A fictional patient has cough and fever.")

    result = SymbolicEngine.load().reason(extraction)

    assert "synthetic-cough-fever-pattern" in {
        rule.rule_id for rule in result.fired_rules
    }
    alpha = next(
        signal for signal in result.candidate_signals
        if signal.assessment_signal == "synthetic_pattern_alpha"
    )
    assert alpha.supporting_evidence
    assert all(
        condition.status == "matched"
        for rule in alpha.supporting_evidence
        for condition in rule.matched_conditions
    )


def test_rule_does_not_fire_when_one_condition_is_not_met() -> None:
    extraction = parse_patient_description("A fictional patient has cough.")

    result = SymbolicEngine.load().reason(extraction)

    assert result.fired_rules == []
    trace = next(
        trace for trace in result.rule_trace
        if trace.rule_id == "synthetic-cough-fever-pattern"
    )
    assert trace.status == "blocked_missing_information"
    assert any("fever" in item for item in result.missing_information)


def test_multiple_rules_fire_with_forward_chaining() -> None:
    extraction = parse_patient_description(
        "A fictional patient has cough, fever, and headache."
    )

    result = SymbolicEngine.load().reason(extraction)

    assert {rule.rule_id for rule in result.fired_rules} >= {
        "synthetic-cough-fever-pattern",
        "synthetic-forward-chain-to-beta",
    }
    chained = next(
        rule for rule in result.fired_rules
        if rule.rule_id == "synthetic-forward-chain-to-beta"
    )
    assert any(
        condition.fact_type == "evidence_flag"
        and condition.key == "alpha_pattern_supported"
        and condition.status == "matched"
        for condition in chained.matched_conditions
    )
    assert result.fired_rules == sorted(
        result.fired_rules, key=lambda rule: (-rule.priority, rule.rule_id)
    )


def test_reports_missing_observation_without_firing_rule() -> None:
    extraction = ExtractionResult(
        state=ReasoningState(
            symptoms=[Symptom(name="headache")],
            vital_signs=VitalSigns(),
        ),
        extraction_confidence=1,
    )

    result = SymbolicEngine.load().reason(extraction)

    assert "synthetic-oxygen-observation-conflict" not in {
        rule.rule_id for rule in result.fired_rules
    }
    trace = next(
        trace for trace in result.rule_trace
        if trace.rule_id == "synthetic-oxygen-observation-conflict"
    )
    assert trace.status == "blocked_missing_information"
    assert any("oxygen_saturation_percent" in item for item in result.missing_information)


def test_detects_conflicting_fired_rules_for_same_signal() -> None:
    extraction = parse_patient_description(
        "A fictional patient has cough and fever but no fever."
    )

    result = SymbolicEngine.load().reason(extraction)

    assert {rule.rule_id for rule in result.fired_rules} >= {
        "synthetic-cough-fever-pattern",
        "synthetic-cough-with-negated-fever",
    }
    alpha = next(
        signal for signal in result.candidate_signals
        if signal.assessment_signal == "synthetic_pattern_alpha"
    )
    assert alpha.has_conflicting_evidence
    assert alpha.supporting_evidence and alpha.conflicting_evidence
    assert any(
        conflict.assessment_signal == "synthetic_pattern_alpha"
        for conflict in result.conflicts
    )


def test_negated_observation_condition_is_supported() -> None:
    extraction = parse_patient_description(
        "A fictional patient has cough and does not have a fever."
    )

    result = SymbolicEngine.load().reason(extraction)

    rule = next(
        rule for rule in result.fired_rules
        if rule.rule_id == "synthetic-cough-with-negated-fever"
    )
    assert any(
        condition.fact_type == "observation"
        and condition.operator == "negated"
        and condition.status == "matched"
        for condition in rule.matched_conditions
    )


def test_numeric_observation_condition_uses_structured_observation() -> None:
    extraction = ExtractionResult(
        state=ReasoningState(
            symptoms=[Symptom(name="headache")],
            extracted_observations=[
                ExtractedObservation(
                    name="oxygen_saturation_percent",
                    value="88",
                    source_text="synthetic structured input",
                    source_type="structured_input",
                )
            ],
        ),
        extraction_confidence=1,
    )

    result = SymbolicEngine.load().reason(extraction)

    assert "synthetic-oxygen-observation-conflict" in {
        rule.rule_id for rule in result.fired_rules
    }


def test_rules_reject_numeric_comparison_without_numeric_threshold() -> None:
    with pytest.raises(ValidationError, match="requires a numeric value"):
        RuleCondition(
            fact_type="observation",
            key="measurement",
            operator="less_than",
            value="not a number",
        )


def test_result_labels_outputs_as_non_diagnostic() -> None:
    result = SymbolicEngine.load().reason(
        parse_patient_description("A fictional patient has cough and fever.")
    )

    assert "not diagnoses" in result.explanation
    assert all(signal.assessment_signal.startswith("synthetic_") for signal in result.candidate_signals)


def test_rule_set_rejects_duplicate_rule_ids() -> None:
    rule = ProductionRule(
        id="duplicate",
        priority=1,
        conditions=[
            RuleCondition(fact_type="symptom", key="cough", operator="present")
        ],
        assessment_signal="synthetic_signal",
        polarity="supporting",
        evidence_flag="synthetic_flag",
        explanation="Synthetic test rule.",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        RuleSet(description="synthetic", rules=[rule, rule])
