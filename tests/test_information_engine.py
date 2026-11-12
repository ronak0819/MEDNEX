import json

import pytest
from pydantic import ValidationError

from mednex.engines.bayesian_engine import BayesianEngine
from mednex.engines.information_engine import (
    InformationEngine,
    InformationQuestionSet,
    entropy,
)
from mednex.nlp import parse_patient_description


def test_entropy_known_uniform_binary_distribution() -> None:
    assert entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)


def test_entropy_known_uniform_four_outcome_distribution() -> None:
    assert entropy({str(index): 0.25 for index in range(4)}) == pytest.approx(2.0)


def test_entropy_of_deterministic_distribution_is_zero() -> None:
    assert entropy({"certain": 1.0, "impossible": 0.0}) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "distribution",
    [
        {},
        {"a": 0.4, "b": 0.4},
        {"a": -0.1, "b": 1.1},
        {"a": float("nan"), "b": float("nan")},
    ],
)
def test_entropy_rejects_invalid_distributions(distribution) -> None:
    with pytest.raises(ValueError):
        entropy(distribution)


def test_expected_information_gain_for_perfect_binary_question() -> None:
    questions = InformationQuestionSet.model_validate(
        {
            "description": "Synthetic exact test.",
            "parameter_origin": "Hand-calculated synthetic example.",
            "conditional_independence_assumption": "Test-only.",
            "questions": [
                {
                    "id": "perfect",
                    "text": "Which binary state is true?",
                    "evidence_key": "symptom:binary",
                    "answers": {
                        "yes": {"a": 1.0, "b": 0.0},
                        "no": {"a": 0.0, "b": 1.0},
                    },
                }
            ],
        }
    )
    result = InformationEngine(questions).rank_questions({"a": 0.5, "b": 0.5})

    question = result.ranked_questions[0]
    assert result.current_entropy_bits == pytest.approx(1.0)
    assert question.expected_posterior_entropy_bits == pytest.approx(0.0)
    assert question.information_gain_bits == pytest.approx(1.0)
    assert sum(answer.probability for answer in question.possible_answers) == pytest.approx(1)


def test_expected_information_gain_for_uninformative_question_is_zero() -> None:
    questions = InformationQuestionSet.model_validate(
        {
            "description": "Synthetic exact test.",
            "parameter_origin": "Hand-calculated synthetic example.",
            "conditional_independence_assumption": "Test-only.",
            "questions": [
                {
                    "id": "uninformative",
                    "text": "A mathematically uninformative question?",
                    "evidence_key": "symptom:irrelevant",
                    "answers": {
                        "yes": {"a": 0.5, "b": 0.5},
                        "no": {"a": 0.5, "b": 0.5},
                    },
                }
            ],
        }
    )
    result = InformationEngine(questions).rank_questions({"a": 0.5, "b": 0.5})

    assert result.ranked_questions == []
    assert result.excluded_questions["uninformative"].startswith(
        "The modeled answer likelihoods do not change"
    )


def test_bundled_questions_are_ranked_by_gain_and_explainable() -> None:
    extraction = parse_patient_description("A fictional patient has cough.")
    posterior = BayesianEngine.load().infer(extraction)

    result = InformationEngine.load().rank_questions(posterior, extraction)

    gains = [question.information_gain_bits for question in result.ranked_questions]
    assert gains == sorted(gains, reverse=True)
    assert all(question.explanation for question in result.ranked_questions)
    assert "ask_cough" in result.excluded_questions
    assert all(question.evidence_key != "symptom:cough" for question in result.ranked_questions)
    assert "mathematical" in result.safety_notice
    assert "independently perform" in result.safety_notice


def test_bundled_question_answers_are_already_normalized_likelihoods() -> None:
    question_set = InformationEngine.load().question_set

    for question in question_set.questions:
        for label in next(iter(question.answers.values())):
            assert sum(
                answer[label] for answer in question.answers.values()
            ) == pytest.approx(1)


def test_excludes_question_when_observation_was_already_reported() -> None:
    extraction = parse_patient_description(
        "Temperature is 37 degrees Celsius."
    )
    result = InformationEngine.load().rank_questions(
        BayesianEngine.load().infer(extraction), extraction
    )

    assert result.excluded_questions["ask_temperature"] == (
        "The corresponding observation is already reported."
    )


def test_unanswered_relevant_questions_can_be_ranked() -> None:
    extraction = parse_patient_description("A fictional patient has cough.")
    result = InformationEngine.load().rank_questions(
        BayesianEngine.load().infer(extraction), extraction
    )

    ids = {question.question_id for question in result.ranked_questions}
    assert "ask_fever" in ids
    assert "ask_headache" in ids


def test_question_likelihoods_must_sum_to_one_per_candidate() -> None:
    raw = json.loads(InformationEngine.load().question_set.model_dump_json())
    raw["questions"][0]["answers"]["present"]["synthetic_pattern_alpha"] = 0.9

    with pytest.raises(ValidationError, match="must sum to 1"):
        InformationQuestionSet.model_validate(raw)


def test_question_candidate_labels_must_match_posterior_labels() -> None:
    with pytest.raises(ValueError, match="exactly the Bayesian candidate labels"):
        InformationEngine.load().rank_questions({"other": 1.0})
