import json

import pytest
from pydantic import ValidationError

from mednex.engines.bayesian_engine import BayesianEngine, BayesianParameters
from mednex.nlp import parse_patient_description


def _candidate_map(result):
    return {candidate.candidate_label: candidate for candidate in result.candidates}


def test_hand_calculated_single_evidence_bayes_update() -> None:
    # alpha: (.4 * .8) / (.4*.8 + .35*.3 + .25*.2)
    result = BayesianEngine.load().infer(
        parse_patient_description("A fictional patient has cough.")
    )
    candidates = _candidate_map(result)

    assert candidates["synthetic_pattern_alpha"].prior_probability == pytest.approx(0.4)
    assert candidates["synthetic_pattern_alpha"].unnormalized_joint == pytest.approx(0.32)
    assert result.normalization_constant == pytest.approx(0.475)
    assert candidates["synthetic_pattern_alpha"].posterior_probability == pytest.approx(
        0.32 / 0.475
    )
    assert candidates["synthetic_pattern_beta"].posterior_probability == pytest.approx(
        0.105 / 0.475
    )
    assert candidates["synthetic_other"].posterior_probability == pytest.approx(
        0.05 / 0.475
    )
    assert sum(item.posterior_probability for item in result.candidates) == pytest.approx(1)


def test_hand_calculated_multiple_evidence_update_and_trace() -> None:
    # alpha: (.4 * .8 * .75) / (.24 + .02625 + .015)
    result = BayesianEngine.load().infer(
        parse_patient_description(
            "A fictional patient has cough and fever."
        )
    )
    candidates = _candidate_map(result)
    alpha = candidates["synthetic_pattern_alpha"]

    assert alpha.unnormalized_joint == pytest.approx(0.24)
    assert result.normalization_constant == pytest.approx(0.28125)
    assert alpha.posterior_probability == pytest.approx(0.24 / 0.28125)
    assert [item.evidence_key for item in alpha.evidence_calculations] == [
        "symptom:cough:present",
        "symptom:fever:present",
    ]
    assert [item.joint_after_evidence for item in alpha.evidence_calculations] == pytest.approx(
        [0.32, 0.24]
    )


def test_missing_observations_do_not_count_as_negative_evidence() -> None:
    result = BayesianEngine.load().infer(
        parse_patient_description("An unspecified fictional patient.")
    )
    candidates = _candidate_map(result)

    assert not result.evidence_used
    assert candidates["synthetic_pattern_alpha"].posterior_probability == pytest.approx(0.4)
    assert candidates["synthetic_pattern_beta"].posterior_probability == pytest.approx(0.35)
    assert "symptom:fever" in result.missing_observations
    assert "symptom:cough" in result.missing_observations


def test_negated_symptom_uses_explicit_absence_likelihood() -> None:
    result = BayesianEngine.load().infer(
        parse_patient_description("A fictional patient has no fever.")
    )

    assert [item.key for item in result.evidence_used] == ["symptom:fever:absent"]
    alpha = _candidate_map(result)["synthetic_pattern_alpha"]
    assert alpha.evidence_calculations[0].likelihood == pytest.approx(0.25)
    assert alpha.unnormalized_joint == pytest.approx(0.1)


def test_unmodeled_observations_are_reported_but_not_calculated() -> None:
    result = BayesianEngine.load().infer(
        parse_patient_description("A fictional patient has dizziness.")
    )

    assert "symptom:dizziness" in result.unmodeled_observations
    assert not result.evidence_used
    assert all(not candidate.evidence_calculations for candidate in result.candidates)


def test_contradictory_symptom_is_excluded_and_reported() -> None:
    result = BayesianEngine.load().infer(
        parse_patient_description("A fictional patient has fever but no fever.")
    )

    assert "fever" in result.contradictory_observations
    assert all(
        evidence.key not in {"symptom:fever:present", "symptom:fever:absent"}
        for evidence in result.evidence_used
    )


def test_contradictory_measurements_are_not_counted_twice() -> None:
    extraction = parse_patient_description(
        "Temperature is 37 C and temperature is 39 C."
    )

    result = BayesianEngine.load().infer(extraction)

    assert "observation:temperature_celsius" in result.contradictory_observations
    assert all(
        evidence.key != "observation:temperature_celsius:reported"
        for evidence in result.evidence_used
    )


@pytest.mark.parametrize(
    ("priors", "message"),
    [
        (
            {"a": 0.6, "b": 0.6},
            "must sum to 1",
        ),
        (
            {"a": -0.1, "b": 1.1},
            "greater than or equal to 0",
        ),
        (
            {"a": 0.4, "b": 0.6},
            "must define every prior candidate",
        ),
    ],
)
def test_rejects_invalid_probability_parameters(priors, message) -> None:
    raw = json.loads(BayesianEngine.load().parameters.model_dump_json())
    raw["priors"] = priors
    raw["likelihoods"] = {
        key: {"a": 0.5, "b": 0.5}
        for key in raw["likelihoods"]
    }
    if "must define every prior candidate" in message:
        raw["likelihoods"]["symptom:cough:present"] = {"a": 0.5}

    with pytest.raises(ValidationError, match=message):
        BayesianParameters.model_validate(raw)


def test_valid_zero_likelihood_is_supported_and_normalized() -> None:
    raw = json.loads(BayesianEngine.load().parameters.model_dump_json())
    raw["likelihoods"]["symptom:cough:present"] = {
        "synthetic_pattern_alpha": 0.0,
        "synthetic_pattern_beta": 0.5,
        "synthetic_other": 0.5,
    }
    engine = BayesianEngine(BayesianParameters.model_validate(raw))

    result = engine.infer(parse_patient_description("A fictional patient has cough."))

    candidates = _candidate_map(result)
    assert candidates["synthetic_pattern_alpha"].posterior_probability == 0
    assert sum(item.posterior_probability for item in result.candidates) == pytest.approx(1)


def test_rejects_likelihood_outside_probability_range() -> None:
    raw = json.loads(BayesianEngine.load().parameters.model_dump_json())
    raw["likelihoods"]["symptom:cough:present"]["synthetic_pattern_alpha"] = 1.1

    with pytest.raises(ValidationError, match="less than or equal to 1"):
        BayesianParameters.model_validate(raw)


def test_empty_evidence_preserves_priors_and_exposes_assumptions() -> None:
    result = BayesianEngine.load().infer(
        parse_patient_description("A fictional patient.")
    )
    candidates = _candidate_map(result)

    assert all(
        candidates[label].posterior_probability == pytest.approx(prior)
        for label, prior in BayesianEngine.load().parameters.priors.items()
    )
    assert "conditionally independent" in result.conditional_independence_assumption
    assert "synthetic" in result.parameter_origin
    assert "not certainty" in result.conclusion_notice
