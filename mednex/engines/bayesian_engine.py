from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator

from mednex.models import MEDNEXModel
from mednex.state import ExtractionResult

Probability = Annotated[float, Field(ge=0, le=1)]
PositiveProbability = Annotated[float, Field(gt=0, le=1)]
DEFAULT_PARAMETERS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "bayesian_parameters.json"
)


class BayesianParameters(MEDNEXModel):
    description: str = Field(min_length=1)
    parameter_origin: str = Field(min_length=1)
    conditional_independence_assumption: str = Field(min_length=1)
    priors: dict[str, Probability] = Field(min_length=2)
    likelihoods: dict[str, dict[str, Probability]] = Field(min_length=1)
    limitations: list[str] = Field(min_length=1)

    @field_validator("priors")
    @classmethod
    def validate_priors_sum_to_one(
        cls, priors: dict[str, float]
    ) -> dict[str, float]:
        if not math.isclose(sum(priors.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Candidate priors must sum to 1.")
        return priors

    @model_validator(mode="after")
    def validate_likelihood_tables(self) -> BayesianParameters:
        candidate_labels = set(self.priors)
        for evidence_key, table in self.likelihoods.items():
            if not evidence_key.strip():
                raise ValueError("Likelihood evidence keys must not be empty.")
            if set(table) != candidate_labels:
                raise ValueError(
                    f"Likelihoods for {evidence_key!r} must define every prior candidate."
                )
        return self


class ObservedEvidence(MEDNEXModel):
    key: str
    description: str
    source_text: str


class EvidenceCalculation(MEDNEXModel):
    evidence_key: str
    evidence_description: str
    source_text: str
    likelihood: Probability
    joint_after_evidence: Probability


class BayesianCandidateResult(MEDNEXModel):
    candidate_label: str
    prior_probability: Probability
    unnormalized_joint: Probability
    posterior_probability: Probability
    evidence_calculations: list[EvidenceCalculation]


class BayesianInferenceResult(MEDNEXModel):
    candidates: list[BayesianCandidateResult]
    evidence_used: list[ObservedEvidence]
    missing_observations: list[str]
    unmodeled_observations: list[str]
    contradictory_observations: list[str]
    normalization_constant: PositiveProbability
    parameter_origin: str
    conditional_independence_assumption: str
    limitations: list[str]
    conclusion_notice: str = (
        "Posterior probabilities are conditional on synthetic teaching parameters. "
        "They are not certainty, a diagnosis, or clinical risk estimates."
    )

    @model_validator(mode="after")
    def validate_posterior_normalization(self) -> BayesianInferenceResult:
        if not math.isclose(
            sum(candidate.posterior_probability for candidate in self.candidates),
            1.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("Posterior probabilities must sum to 1.")
        return self


def _extract_observed_evidence(
    extraction: ExtractionResult,
    parameters: BayesianParameters,
) -> tuple[list[ObservedEvidence], list[str], list[str], list[str]]:
    symptom_states: dict[str, set[bool]] = {}
    for symptom in extraction.state.symptoms:
        name = symptom.name.casefold()
        symptom_states.setdefault(name, set()).add(symptom.is_negated)

    contradictory = sorted(
        name for name, states in symptom_states.items() if len(states) > 1
    )
    observation_values: dict[str, set[str]] = {}
    for observation in extraction.state.extracted_observations:
        if observation.name != "symptom":
            observation_values.setdefault(observation.name, set()).add(
                observation.value
            )
    contradictory.extend(
        f"observation:{name}"
        for name, values in observation_values.items()
        if len(values) > 1
    )
    contradictory_observation_names = {
        item.removeprefix("observation:")
        for item in contradictory
        if item.startswith("observation:")
    }
    evidence: list[ObservedEvidence] = []
    seen_keys: set[str] = set()
    modeled_symptoms: set[str] = set()
    for symptom in extraction.state.symptoms:
        if symptom.name.casefold() in contradictory:
            continue
        evidence_key = (
            f"symptom:{symptom.name.casefold()}:"
            f"{'absent' if symptom.is_negated else 'present'}"
        )
        if evidence_key not in parameters.likelihoods or evidence_key in seen_keys:
            continue
        modeled_symptoms.add(symptom.name.casefold())
        seen_keys.add(evidence_key)
        evidence.append(
            ObservedEvidence(
                key=evidence_key,
                description=f"{symptom.name} explicitly "
                f"{'negated' if symptom.is_negated else 'reported'}",
                source_text=symptom.name,
            )
        )

    observation_aliases = {
        "temperature_celsius": "observation:temperature_celsius:reported",
        "heart_rate_bpm": "observation:heart_rate_bpm:reported",
        "respiratory_rate_per_minute": "observation:respiratory_rate_per_minute:reported",
        "oxygen_saturation_percent": "observation:oxygen_saturation_percent:reported",
    }
    modeled_observations: set[str] = set()
    for observation in extraction.state.extracted_observations:
        if observation.name == "symptom":
            continue
        if observation.name in contradictory_observation_names:
            continue
        evidence_key = observation_aliases.get(observation.name)
        if evidence_key is None:
            continue
        modeled_observations.add(observation.name)
        if evidence_key not in parameters.likelihoods or evidence_key in seen_keys:
            continue
        seen_keys.add(evidence_key)
        evidence.append(
            ObservedEvidence(
                key=evidence_key,
                description=f"{observation.name} was explicitly reported",
                source_text=observation.source_text,
            )
        )

    if extraction.state.vital_signs is not None:
        for field_name, evidence_key in observation_aliases.items():
            if (
                getattr(extraction.state.vital_signs, field_name) is not None
                and field_name not in modeled_observations
                and f"observation:{field_name}" not in contradictory
                and evidence_key in parameters.likelihoods
                and evidence_key not in seen_keys
            ):
                seen_keys.add(evidence_key)
                evidence.append(
                    ObservedEvidence(
                        key=evidence_key,
                        description=f"{field_name} was supplied in structured vital signs",
                        source_text="Structured vital-sign state",
                    )
                )

    unmodeled: list[str] = []
    for symptom_name, states in symptom_states.items():
        if symptom_name in contradictory or symptom_name in modeled_symptoms:
            continue
        unmodeled.append(f"symptom:{symptom_name}")
    for observation in extraction.state.extracted_observations:
        if (
            observation.name != "symptom"
            and observation.name not in observation_aliases
            and observation.name not in contradictory_observation_names
        ):
            unmodeled.append(f"observation:{observation.name}")

    observed_variables = {
        ":".join(item.key.split(":")[:2]) for item in evidence
    }
    modeled_variables = {
        ":".join(key.split(":")[:2]) for key in parameters.likelihoods
    }
    missing = sorted(
        variable
        for variable in modeled_variables - observed_variables
        if not (
            variable.startswith("symptom:")
            and variable.split(":")[1] in contradictory
        )
        and not (
            variable.startswith("observation:")
            and variable.split(":")[1] in contradictory_observation_names
        )
    )
    return evidence, missing, sorted(set(unmodeled)), contradictory


class BayesianEngine:
    """Transparent naive-Bayes demonstration with synthetic parameters only."""

    def __init__(self, parameters: BayesianParameters) -> None:
        self.parameters = parameters

    @classmethod
    def load(cls, path: str | Path | None = None) -> BayesianEngine:
        parameters_path = Path(path) if path is not None else DEFAULT_PARAMETERS_PATH
        with parameters_path.open("r", encoding="utf-8") as parameter_file:
            raw_parameters = json.load(parameter_file)
        return cls(BayesianParameters.model_validate(raw_parameters))

    def infer(self, extraction: ExtractionResult) -> BayesianInferenceResult:
        evidence, missing, unmodeled, contradictory = _extract_observed_evidence(
            extraction, self.parameters
        )
        candidate_calculations: dict[str, list[EvidenceCalculation]] = {}
        unnormalized: dict[str, float] = {}

        for candidate_label, prior in self.parameters.priors.items():
            joint = prior
            calculations: list[EvidenceCalculation] = []
            for item in evidence:
                likelihood = self.parameters.likelihoods[item.key][candidate_label]
                joint *= likelihood
                calculations.append(
                    EvidenceCalculation(
                        evidence_key=item.key,
                        evidence_description=item.description,
                        source_text=item.source_text,
                        likelihood=likelihood,
                        joint_after_evidence=joint,
                    )
                )
            unnormalized[candidate_label] = joint
            candidate_calculations[candidate_label] = calculations

        normalization_constant = sum(unnormalized.values())
        if not math.isfinite(normalization_constant) or normalization_constant <= 0:
            raise ValueError(
                "Bayesian normalization failed: evidence has zero or invalid total likelihood."
            )

        candidates = [
            BayesianCandidateResult(
                candidate_label=candidate_label,
                prior_probability=prior,
                unnormalized_joint=unnormalized[candidate_label],
                posterior_probability=unnormalized[candidate_label]
                / normalization_constant,
                evidence_calculations=candidate_calculations[candidate_label],
            )
            for candidate_label, prior in self.parameters.priors.items()
        ]
        return BayesianInferenceResult(
            candidates=candidates,
            evidence_used=evidence,
            missing_observations=missing,
            unmodeled_observations=unmodeled,
            contradictory_observations=sorted(set(contradictory)),
            normalization_constant=normalization_constant,
            parameter_origin=self.parameters.parameter_origin,
            conditional_independence_assumption=(
                self.parameters.conditional_independence_assumption
            ),
            limitations=self.parameters.limitations,
        )
