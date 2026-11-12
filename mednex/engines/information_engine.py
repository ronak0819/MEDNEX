from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import Field, model_validator

from mednex.models import MEDNEXModel
from mednex.engines.bayesian_engine import BayesianInferenceResult
from mednex.state import ExtractionResult

DEFAULT_QUESTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "information_questions.json"
)


class EducationalQuestion(MEDNEXModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_key: str = Field(min_length=1)
    answers: dict[str, dict[str, float]] = Field(min_length=2)


class InformationQuestionSet(MEDNEXModel):
    description: str = Field(min_length=1)
    parameter_origin: str = Field(min_length=1)
    conditional_independence_assumption: str = Field(min_length=1)
    questions: list[EducationalQuestion]

    @model_validator(mode="after")
    def validate_unique_questions_and_answers(self) -> InformationQuestionSet:
        ids = [question.id for question in self.questions]
        evidence_keys = [question.evidence_key for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Information question ids must be unique.")
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("Question evidence keys must be unique.")
        for question in self.questions:
            labels = set(next(iter(question.answers.values())))
            for answer, row in question.answers.items():
                if set(row) != labels:
                    raise ValueError(
                        f"Answer likelihoods for {question.id!r} must use the same candidates."
                    )
                for label, probability in row.items():
                    if not math.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError(
                            f"Answer likelihood for {label!r} must be finite and in [0, 1]."
                        )
            for label in labels:
                total = sum(row[label] for row in question.answers.values())
                if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(
                        f"Answer likelihoods for question {question.id!r} and "
                        f"candidate {label!r} must sum to 1."
                    )
        return self


class AnswerInformationGain(MEDNEXModel):
    answer: str
    probability: float = Field(ge=0, le=1)
    posterior_entropy_bits: float = Field(ge=0)


class RankedQuestion(MEDNEXModel):
    question_id: str
    question: str
    evidence_key: str
    information_gain_bits: float = Field(ge=0)
    current_entropy_bits: float = Field(ge=0)
    expected_posterior_entropy_bits: float = Field(ge=0)
    possible_answers: list[AnswerInformationGain]
    explanation: str


class InformationGainResult(MEDNEXModel):
    candidate_labels: list[str]
    prior_distribution: dict[str, float]
    current_entropy_bits: float = Field(ge=0)
    ranked_questions: list[RankedQuestion]
    excluded_questions: dict[str, str]
    calculation_basis: str
    parameter_origin: str
    conditional_independence_assumption: str
    limitations: list[str]
    safety_notice: str = (
        "Information gain measures expected mathematical uncertainty reduction "
        "only. It does not establish clinical importance or safety and does not "
        "recommend that a patient independently perform any investigation."
    )


def entropy(distribution: dict[str, float]) -> float:
    """Return Shannon entropy in bits for a valid discrete probability distribution."""
    if not distribution:
        raise ValueError("A probability distribution must contain at least one outcome.")
    if any(not math.isfinite(probability) or probability < 0 for probability in distribution.values()):
        raise ValueError("Probabilities must be finite and non-negative.")
    if not math.isclose(sum(distribution.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Probabilities must sum to 1.")
    return -sum(
        probability * math.log2(probability)
        for probability in distribution.values()
        if probability > 0
    )


def _observation_evidence_keys(extraction: ExtractionResult) -> set[str]:
    keys: set[str] = set()
    symptom_states: dict[str, set[bool]] = {}
    for symptom in extraction.state.symptoms:
        symptom_states.setdefault(symptom.name.casefold(), set()).add(symptom.is_negated)
    for name, states in symptom_states.items():
        if len(states) == 1:
            state = "absent" if True in states else "present"
            keys.add(f"symptom:{name}:{state}")
    observation_aliases = {
        "temperature_celsius": "observation:temperature_celsius:reported",
        "heart_rate_bpm": "observation:heart_rate_bpm:reported",
        "respiratory_rate_per_minute": "observation:respiratory_rate_per_minute:reported",
        "oxygen_saturation_percent": "observation:oxygen_saturation_percent:reported",
    }
    observed_names: set[str] = set()
    contradictory_names: set[str] = set()
    for observation in extraction.state.extracted_observations:
        if observation.name == "symptom":
            continue
        observed_names.add(observation.name)
    observation_values: dict[str, set[str]] = {}
    for observation in extraction.state.extracted_observations:
        if observation.name != "symptom":
            observation_values.setdefault(observation.name, set()).add(observation.value)
    contradictory_names = {
        name for name, values in observation_values.items() if len(values) > 1
    }
    if extraction.state.vital_signs is not None:
        for field in observation_aliases:
            if getattr(extraction.state.vital_signs, field) is not None:
                observed_names.add(field)
    for name in observed_names - contradictory_names:
        if name in observation_aliases:
            keys.add(observation_aliases[name])
    return keys


def _question_already_answered(
    evidence_key: str,
    observed_keys: set[str],
) -> bool:
    if evidence_key.startswith("symptom:"):
        symptom_name = evidence_key.removeprefix("symptom:")
        return any(
            key in {
                f"symptom:{symptom_name}:present",
                f"symptom:{symptom_name}:absent",
            }
            for key in observed_keys
        )
    if evidence_key.startswith("observation:"):
        observation_name = evidence_key.removeprefix("observation:")
        return f"observation:{observation_name}:reported" in observed_keys
    return False


def _distribution_from_bayesian_result(
    result: BayesianInferenceResult,
) -> dict[str, float]:
    distribution = {
        candidate.candidate_label: candidate.posterior_probability
        for candidate in result.candidates
    }
    entropy(distribution)
    return distribution


class InformationEngine:
    """Ranks synthetic educational questions by expected entropy reduction."""

    def __init__(self, question_set: InformationQuestionSet) -> None:
        self.question_set = question_set

    @classmethod
    def load(cls, path: str | Path | None = None) -> InformationEngine:
        questions_path = Path(path) if path is not None else DEFAULT_QUESTIONS_PATH
        with questions_path.open("r", encoding="utf-8") as questions_file:
            raw_question_set = json.load(questions_file)
        return cls(InformationQuestionSet.model_validate(raw_question_set))

    def _likelihoods_for_labels(
        self,
        question: EducationalQuestion,
        candidate_labels: set[str],
    ) -> None:
        for answer_name, likelihoods in question.answers.items():
            if set(likelihoods) != candidate_labels:
                raise ValueError(
                    f"Answer {answer_name!r} in question {question.id!r} must "
                    "define exactly the Bayesian candidate labels."
                )

    def rank_questions(
        self,
        posterior: BayesianInferenceResult | dict[str, float],
        extraction: ExtractionResult | None = None,
        answered_question_ids: set[str] | None = None,
    ) -> InformationGainResult:
        distribution = (
            _distribution_from_bayesian_result(posterior)
            if isinstance(posterior, BayesianInferenceResult)
            else dict(posterior)
        )
        current_entropy = entropy(distribution)
        labels = set(distribution)
        observed_keys = _observation_evidence_keys(extraction) if extraction else set()
        answered = answered_question_ids or set()
        ranked: list[RankedQuestion] = []
        excluded: dict[str, str] = {}

        for question in self.question_set.questions:
            self._likelihoods_for_labels(question, labels)
            if question.id in answered:
                excluded[question.id] = "This question was already answered in the conversation."
                continue
            if _question_already_answered(question.evidence_key, observed_keys):
                excluded[question.id] = "The corresponding observation is already reported."
                continue

            answer_details: list[AnswerInformationGain] = []
            expected_entropy = 0.0
            for answer, answer_likelihoods in question.answers.items():
                joint = {
                    label: distribution[label] * answer_likelihoods[label]
                    for label in distribution
                }
                answer_probability = sum(joint.values())
                if answer_probability == 0:
                    conditional_entropy = 0.0
                else:
                    conditional_distribution = {
                        label: value / answer_probability
                        for label, value in joint.items()
                    }
                    conditional_entropy = entropy(conditional_distribution)
                    expected_entropy += answer_probability * conditional_entropy
                answer_details.append(
                    AnswerInformationGain(
                        answer=answer,
                        probability=answer_probability,
                        posterior_entropy_bits=conditional_entropy,
                    )
                )

            information_gain = current_entropy - expected_entropy
            if information_gain < -1e-9:
                raise ValueError(
                    f"Expected information gain for {question.id!r} became negative."
                )
            information_gain = max(0.0, information_gain)
            if information_gain <= 1e-12:
                excluded[question.id] = (
                    "The modeled answer likelihoods do not change the candidate distribution."
                )
                continue
            ranked.append(
                RankedQuestion(
                    question_id=question.id,
                    question=question.text,
                    evidence_key=question.evidence_key,
                    information_gain_bits=information_gain,
                    current_entropy_bits=current_entropy,
                    expected_posterior_entropy_bits=expected_entropy,
                    possible_answers=answer_details,
                    explanation=(
                        f"Under the synthetic answer likelihoods, this question "
                        f"is expected to reduce candidate entropy from "
                        f"{current_entropy:.4f} to {expected_entropy:.4f} bits "
                        f"({information_gain:.4f} bits of expected information gain)."
                    ),
                )
            )

        ranked.sort(key=lambda question: (-question.information_gain_bits, question.question_id))
        return InformationGainResult(
            candidate_labels=sorted(labels),
            prior_distribution=distribution,
            current_entropy_bits=current_entropy,
            ranked_questions=ranked,
            excluded_questions=excluded,
            calculation_basis=(
                "H(X) = -sum p(x) log2 p(x); "
                "P(answer) = sum_x P(x)P(answer|x); "
                "P(x|answer) = P(x)P(answer|x)/P(answer); "
                "IG(question) = H(X) - sum_answer P(answer)H(X|answer)."
            ),
            parameter_origin=self.question_set.parameter_origin,
            conditional_independence_assumption=(
                self.question_set.conditional_independence_assumption
            ),
            limitations=[
                self.question_set.description,
                "Information gain is expected mathematical uncertainty reduction, not clinical importance or safety.",
                "Questions and likelihoods are synthetic educational examples and are not validated clinical guidance.",
                "This engine does not recommend that a patient independently perform any investigation.",
                "If extraction is omitted, no questions can be excluded as already answered.",
            ],
        )
