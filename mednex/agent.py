from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from mednex.engines import (
    BayesianEngine,
    BayesianInferenceResult,
    ClassicalMLEngine,
    GraphInvestigationResult,
    InformationEngine,
    InformationGainResult,
    KnowledgeGraph,
    MLPredictionResult,
    ModelEvaluationReport,
    SymbolicEngine,
    SymbolicReasoningResult,
)
from mednex.engines.ml_engine import SUPPORTED_SYMPTOMS
from mednex.engines.explainability import ExplainableReport, build_explainable_report
from mednex.engines.information_engine import RankedQuestion
from mednex.engines.planner import AgentStatus, DecisionPlan, create_decision_plan
from mednex.models import (
    ExtractedObservation,
    MEDNEXModel,
    PatientInformation,
    Symptom,
    VitalSigns,
)
from mednex.nlp import parse_patient_description
from mednex.safety.safety_checks import SafetyReview, review_input
from mednex.safety.validation import InputValidationResult, validate_case_input
from mednex.state import ExtractionResult, ReasoningState

AnswerPolarity = Literal["present", "absent", "reported", "not_reported", "unknown"]
SymptomCorrection = Literal["present", "absent", "not_reported"]
FIELD_CORRECTION_KEYS = {
    "age_years",
    "sex",
    "temperature_celsius",
    "heart_rate_bpm",
    "respiratory_rate_per_minute",
    "systolic_blood_pressure_mmhg",
    "diastolic_blood_pressure_mmhg",
    "oxygen_saturation_percent",
}
YES_PATTERN = re.compile(
    r"\b(?:yes|present|reported|measured|has|have|reports?|experiences?)\b",
    re.I,
)
NO_PATTERN = re.compile(
    r"\b(?:no|not present|absent|not reported|was not measured|"
    r"don't have|do not have|doesn't have|does not have)\b",
    re.I,
)
UNKNOWN_PATTERN = re.compile(r"\b(?:unknown|unsure|not sure|don't know|do not know|uncertain)\b", re.I)
MAX_FOLLOW_UP_QUESTIONS = 3


class AgentSession(MEDNEXModel):
    case_description: str
    answers: dict[str, str] = Field(default_factory=dict)
    symptom_corrections: dict[str, SymptomCorrection] = Field(default_factory=dict)
    field_corrections: dict[str, str | None] = Field(default_factory=dict)
    answered_question_ids: list[str] = Field(default_factory=list)
    follow_up_count: int = 0
    current_question_id: str | None = None
    previous_posteriors: dict[str, float] = Field(default_factory=dict)
    previous_fired_rule_ids: list[str] = Field(default_factory=list)
    previous_graph_edges: list[str] = Field(default_factory=list)
    previous_ml_label: str | None = None
    closed: bool = False


class AgentTurn(MEDNEXModel):
    session: AgentSession
    status: AgentStatus
    question_to_user: str | None = None
    validation: InputValidationResult | None = None
    safety: SafetyReview
    extraction: ExtractionResult | None = None
    graph: GraphInvestigationResult | None = None
    symbolic: SymbolicReasoningResult | None = None
    bayesian: BayesianInferenceResult | None = None
    ml_prediction: MLPredictionResult | None = None
    ml_evaluation: ModelEvaluationReport | None = None
    information_gain: InformationGainResult | None = None
    decision: DecisionPlan
    report: ExplainableReport
    change_explanation: str


def _answer_polarity(question: RankedQuestion, answer: str) -> AnswerPolarity:
    stripped = answer.strip()
    if UNKNOWN_PATTERN.search(stripped):
        return "unknown"
    if re.match(
        r"^(?:no\b|not\b|absent\b|don't\b|do not\b|doesn't\b|does not\b)",
        stripped,
        re.IGNORECASE,
    ):
        return "absent" if question.evidence_key.startswith("symptom:") else "not_reported"
    negative = NO_PATTERN.search(stripped) is not None
    positive = YES_PATTERN.search(stripped) is not None
    evidence_key = question.evidence_key
    if evidence_key.startswith("symptom:"):
        if re.search(r"\d", stripped):
            return "reported"
        if negative and not positive:
            return "absent"
        if positive and not negative:
            return "present"
        if re.fullmatch(r"no\b.*", stripped, re.IGNORECASE):
            return "absent"
        if re.fullmatch(r"yes\b.*", stripped, re.IGNORECASE):
            return "present"
        return "unknown"
    if negative and not positive:
        return "not_reported"
    if positive and not negative:
        return "reported"
    if re.fullmatch(r"no\b.*", stripped, re.IGNORECASE):
        return "not_reported"
    if re.fullmatch(r"yes\b.*", stripped, re.IGNORECASE):
        return "reported"
    return "unknown"


def _apply_answers(
    extraction: ExtractionResult,
    session: AgentSession,
    information_engine: InformationEngine,
) -> ExtractionResult:
    questions = {
        question.id: question
        for question in information_engine.question_set.questions
    }
    symptoms = list(extraction.state.symptoms)
    observations = list(extraction.state.extracted_observations)
    answer_conflicts = list(extraction.contradictions)
    for question_id, answer in session.answers.items():
        question = questions.get(question_id)
        if question is None:
            continue
        polarity = _answer_polarity(
            RankedQuestion(
                question_id=question.id,
                question=question.text,
                evidence_key=question.evidence_key,
                information_gain_bits=0,
                current_entropy_bits=0,
                expected_posterior_entropy_bits=0,
                possible_answers=[],
                explanation="",
            ),
            answer,
        )
        if polarity == "unknown":
            continue
        if question.evidence_key.startswith("symptom:"):
            symptom_name = question.evidence_key.removeprefix("symptom:")
            existing = [
                symptom for symptom in symptoms
                if symptom.name.casefold() == symptom_name.casefold()
            ]
            answer_is_negated = polarity == "absent"
            if existing and any(
                symptom.is_negated != answer_is_negated for symptom in existing
            ):
                message = (
                    f"Follow-up answer conflicts with the original statement for "
                    f"symptom {symptom_name!r}."
                )
                if message not in answer_conflicts:
                    answer_conflicts.append(message)
                continue
            if not existing:
                symptoms.append(
                    Symptom(name=symptom_name, is_negated=answer_is_negated)
                )
                observations.append(
                    ExtractedObservation(
                        name="symptom",
                        value=f"{'negated' if answer_is_negated else 'present'}: {symptom_name}",
                        source_text=answer,
                        source_type="structured_input",
                        is_negated=answer_is_negated,
                    )
                )
        elif question.evidence_key.startswith("observation:"):
            observation_name = question.evidence_key.removeprefix("observation:")
            if polarity == "reported" and not any(
                observation.name == observation_name for observation in observations
            ):
                observations.append(
                    ExtractedObservation(
                        name=observation_name,
                        value="reported",
                        source_text=answer,
                        source_type="structured_input",
                    )
                )

    for symptom_name, correction in session.symptom_corrections.items():
        symptoms = [
            symptom
            for symptom in symptoms
            if symptom.name.casefold() != symptom_name.casefold()
        ]
        observations = [
            observation
            for observation in observations
            if not (
                observation.name == "symptom"
                and symptom_name.casefold() in observation.value.casefold()
            )
        ]
        if correction == "not_reported":
            continue
        is_negated = correction == "absent"
        symptoms.append(Symptom(name=symptom_name, is_negated=is_negated))
        observations.append(
            ExtractedObservation(
                name="symptom",
                value=f"{'negated' if is_negated else 'present'}: {symptom_name}",
                source_text="User-corrected extracted symptom",
                source_type="structured_input",
                is_negated=is_negated,
            )
        )

    patient_fields = extraction.state.patient_information.model_dump()
    vital_fields = (
        extraction.state.vital_signs.model_dump()
        if extraction.state.vital_signs is not None
        else {key: None for key in FIELD_CORRECTION_KEYS if key != "age_years" and key != "sex"}
    )
    for key, raw_value in session.field_corrections.items():
        if key not in FIELD_CORRECTION_KEYS:
            continue
        if raw_value is None or not raw_value.strip():
            value = None
        elif key == "age_years":
            value = int(raw_value)
        elif key == "sex":
            value = raw_value.strip()
        else:
            value = float(raw_value)
        if key in patient_fields:
            patient_fields[key] = value
        else:
            vital_fields[key] = value
        observations = [item for item in observations if item.name != key]
        if value is not None:
            observations.append(
                ExtractedObservation(
                    name=key,
                    value=str(value),
                    source_text="User-corrected extracted information",
                    source_type="structured_input",
                )
            )

    patient_information = PatientInformation.model_validate(patient_fields)
    vital_signs = VitalSigns.model_validate(vital_fields)
    updated_state = ReasoningState(
        original_description="\n".join(
            [
                extraction.state.original_description,
                *[
                    f"[Follow-up {question_id}: {answer}]"
                    for question_id, answer in session.answers.items()
                ],
            ]
        ).strip(),
        patient_information=patient_information,
        symptoms=symptoms,
        vital_signs=(
            vital_signs
            if any(value is not None for value in vital_fields.values())
            else None
        ),
        extracted_observations=observations,
        missing_information=extraction.state.missing_information,
        reasoning_evidence=extraction.state.reasoning_evidence,
        investigation_suggestions=extraction.state.investigation_suggestions,
        final_report=extraction.state.final_report,
    )
    return ExtractionResult(
        state=updated_state,
        extraction_confidence=extraction.extraction_confidence,
        ambiguous_statements=extraction.ambiguous_statements,
        unsupported_statements=extraction.unsupported_statements,
        contradictions=answer_conflicts,
    )


class MEDNEXAgent:
    """Zero-LLM orchestrator for fictional, synthetic educational cases."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph | None = None,
        symbolic_engine: SymbolicEngine | None = None,
        bayesian_engine: BayesianEngine | None = None,
        ml_engine: ClassicalMLEngine | None = None,
        information_engine: InformationEngine | None = None,
        max_follow_up_questions: int = MAX_FOLLOW_UP_QUESTIONS,
    ) -> None:
        if max_follow_up_questions < 0:
            raise ValueError("max_follow_up_questions cannot be negative.")
        self.knowledge_graph = knowledge_graph or KnowledgeGraph.load()
        self.symbolic_engine = symbolic_engine or SymbolicEngine.load()
        self.bayesian_engine = bayesian_engine or BayesianEngine.load()
        self.ml_engine = ml_engine or ClassicalMLEngine.train()
        self.information_engine = information_engine or InformationEngine.load()
        self.max_follow_up_questions = max_follow_up_questions

    def start(self, description: str) -> AgentTurn:
        if not isinstance(description, str):
            raise TypeError("Case description must be a string.")
        session = AgentSession(case_description=description)
        return self._run_cycle(session)

    def answer(self, session: AgentSession, answer: str) -> AgentTurn:
        if session.closed:
            raise ValueError("This MEDNEX session is closed.")
        if session.current_question_id is None:
            raise ValueError("There is no pending MEDNEX follow-up question.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("A non-empty answer is required.")
        safety = review_input(f"{session.case_description}\n{answer}")
        if safety.professional_assessment_required:
            closed_session = AgentSession(
                **{
                    **session.model_dump(),
                    "closed": True,
                    "current_question_id": None,
                }
            )
            return self._safety_turn(closed_session, safety)
        answers = dict(session.answers)
        answers[session.current_question_id] = answer
        answered_ids = list(session.answered_question_ids)
        if session.current_question_id not in answered_ids:
            answered_ids.append(session.current_question_id)
        next_session = AgentSession(
            case_description=session.case_description,
            answers=answers,
            symptom_corrections=session.symptom_corrections,
            field_corrections=session.field_corrections,
            answered_question_ids=answered_ids,
            follow_up_count=session.follow_up_count + 1,
            previous_posteriors=session.previous_posteriors,
            previous_fired_rule_ids=session.previous_fired_rule_ids,
            previous_graph_edges=session.previous_graph_edges,
            previous_ml_label=session.previous_ml_label,
        )
        return self._run_cycle(next_session)

    def correct_symptoms(
        self,
        session: AgentSession,
        corrections: dict[str, SymptomCorrection],
    ) -> AgentTurn:
        if session.closed:
            raise ValueError("This MEDNEX session is closed.")
        if not corrections:
            raise ValueError("At least one symptom correction is required.")
        safety = review_input(session.case_description)
        if safety.professional_assessment_required:
            closed_session = AgentSession(
                **{
                    **session.model_dump(),
                    "closed": True,
                    "current_question_id": None,
                }
            )
            return self._safety_turn(closed_session, safety)
        merged_corrections = dict(session.symptom_corrections)
        merged_corrections.update(
            {name.strip().casefold(): correction for name, correction in corrections.items()}
        )
        corrected_session = AgentSession(
            case_description=session.case_description,
            answers=session.answers,
            symptom_corrections=merged_corrections,
            field_corrections=session.field_corrections,
            answered_question_ids=session.answered_question_ids,
            follow_up_count=session.follow_up_count,
            previous_posteriors=session.previous_posteriors,
            previous_fired_rule_ids=session.previous_fired_rule_ids,
            previous_graph_edges=session.previous_graph_edges,
            previous_ml_label=session.previous_ml_label,
        )
        return self._run_cycle(corrected_session)

    def correct_extracted_fields(
        self,
        session: AgentSession,
        corrections: dict[str, str | None],
    ) -> AgentTurn:
        if session.closed:
            raise ValueError("This MEDNEX session is closed.")
        if not corrections:
            raise ValueError("At least one extracted-field correction is required.")
        unsupported_fields = set(corrections) - FIELD_CORRECTION_KEYS
        if unsupported_fields:
            raise ValueError(
                f"Unsupported extracted-field corrections: {sorted(unsupported_fields)}"
            )
        safety = review_input(session.case_description)
        if safety.professional_assessment_required:
            closed_session = AgentSession(
                **{
                    **session.model_dump(),
                    "closed": True,
                    "current_question_id": None,
                }
            )
            return self._safety_turn(closed_session, safety)
        merged_corrections = dict(session.field_corrections)
        merged_corrections.update(corrections)
        corrected_session = AgentSession(
            case_description=session.case_description,
            answers=session.answers,
            symptom_corrections=session.symptom_corrections,
            field_corrections=merged_corrections,
            answered_question_ids=session.answered_question_ids,
            follow_up_count=session.follow_up_count,
            previous_posteriors=session.previous_posteriors,
            previous_fired_rule_ids=session.previous_fired_rule_ids,
            previous_graph_edges=session.previous_graph_edges,
            previous_ml_label=session.previous_ml_label,
        )
        return self._run_cycle(corrected_session)

    def _safety_turn(self, session: AgentSession, safety: SafetyReview) -> AgentTurn:
        decision = create_decision_plan(
            extraction=None,
            validation=None,
            safety=safety,
            follow_up_count=session.follow_up_count,
            max_follow_up_questions=self.max_follow_up_questions,
        )
        if (
            decision.status == "EVALUATION_COMPLETE"
            and any(
                _answer_polarity_for_question_id(
                    self.information_engine, question_id, answer
                )
                == "unknown"
                for question_id, answer in session.answers.items()
            )
        ):
            decision = DecisionPlan(
                status="UNCERTAIN",
                rationale=(
                    "One or more follow-up responses did not resolve the requested "
                    "information; no candidate is forced."
                ),
                next_action="Stop and retain an uncertain educational result.",
                completed=True,
            )
        report = build_explainable_report(
            plan=decision,
            extraction=None,
            graph=None,
            symbolic=None,
            bayesian=None,
            ml=None,
            information=None,
            safety_message=safety.user_message,
        )
        return AgentTurn(
            session=session,
            status=decision.status,
            safety=safety,
            decision=decision,
            report=report,
            change_explanation="Evaluation stopped at the safety boundary.",
        )

    def _run_cycle(self, session: AgentSession) -> AgentTurn:
        safety = review_input(session.case_description)
        if safety.professional_assessment_required:
            closed_session = AgentSession(
                **{
                    **session.model_dump(),
                    "closed": True,
                    "current_question_id": None,
                }
            )
            return self._safety_turn(closed_session, safety)

        extraction = parse_patient_description(session.case_description)
        extraction = _apply_answers(extraction, session, self.information_engine)
        validation = validate_case_input(session.case_description, extraction)
        if not validation.valid:
            decision = create_decision_plan(
                extraction,
                validation,
                safety,
                follow_up_count=session.follow_up_count,
                max_follow_up_questions=self.max_follow_up_questions,
            )
            report = build_explainable_report(
                decision, extraction, None, None, None, None, None
            )
            closed_session = AgentSession(
                **{**session.model_dump(), "closed": True, "current_question_id": None}
            )
            return AgentTurn(
                session=closed_session,
                status=decision.status,
                validation=validation,
                safety=safety,
                extraction=extraction,
                decision=decision,
                report=report,
                change_explanation="Input validation prevented reasoning.",
            )

        graph_result = self.knowledge_graph.investigate(extraction)
        symbolic_result = self.symbolic_engine.reason(extraction)
        bayesian_result = self.bayesian_engine.infer(extraction)
        ml_feature_states: dict[str, set[bool]] = {}
        for symptom in extraction.state.symptoms:
            if symptom.name in SUPPORTED_SYMPTOMS:
                ml_feature_states.setdefault(symptom.name, set()).add(
                    symptom.is_negated
                )
        has_supported_ml_evidence = any(
            len(polarities) == 1 for polarities in ml_feature_states.values()
        )
        ml_prediction = (
            self.ml_engine.predict(extraction)
            if has_supported_ml_evidence
            else None
        )
        information_result = self.information_engine.rank_questions(
            bayesian_result,
            extraction,
            answered_question_ids=set(session.answered_question_ids),
        )
        decision = create_decision_plan(
            extraction=extraction,
            validation=validation,
            safety=safety,
            symbolic=symbolic_result,
            bayesian=bayesian_result,
            information=information_result,
            follow_up_count=session.follow_up_count,
            max_follow_up_questions=self.max_follow_up_questions,
        )
        current_posteriors = {
            candidate.candidate_label: candidate.posterior_probability
            for candidate in bayesian_result.candidates
        }
        current_fired_rules = [rule.rule_id for rule in symbolic_result.fired_rules]
        current_graph_edges = [
            f"{edge.source}->{edge.target}:{edge.relation}"
            for edge in graph_result.visited_edges
        ]
        current_ml_label = ml_prediction.predicted_label if ml_prediction else None
        updated_session = AgentSession(
            case_description=session.case_description,
            answers=session.answers,
            symptom_corrections=session.symptom_corrections,
            field_corrections=session.field_corrections,
            answered_question_ids=session.answered_question_ids,
            follow_up_count=session.follow_up_count,
            current_question_id=decision.next_question_id,
            previous_posteriors=current_posteriors,
            previous_fired_rule_ids=current_fired_rules,
            previous_graph_edges=current_graph_edges,
            previous_ml_label=current_ml_label,
            closed=decision.completed,
        )
        change_explanation = self._explain_changes(
            session,
            current_posteriors,
            current_fired_rules,
            current_graph_edges,
            current_ml_label,
        )
        report = build_explainable_report(
            decision,
            extraction,
            graph_result,
            symbolic_result,
            bayesian_result,
            ml_prediction,
            information_result,
        )
        return AgentTurn(
            session=updated_session,
            status=decision.status,
            question_to_user=decision.next_question,
            validation=validation,
            safety=safety,
            extraction=extraction,
            graph=graph_result,
            symbolic=symbolic_result,
            bayesian=bayesian_result,
            ml_prediction=ml_prediction,
            ml_evaluation=self.ml_engine.report,
            information_gain=information_result,
            decision=decision,
            report=report,
            change_explanation=change_explanation,
        )

    @staticmethod
    def _answer_polarity_for_question_id(
        information_engine: InformationEngine,
        question_id: str,
        answer: str,
    ) -> AnswerPolarity:
        question = next(
            (
                item
                for item in information_engine.question_set.questions
                if item.id == question_id
            ),
            None,
        )
        if question is None:
            return "unknown"
        return _answer_polarity(
            RankedQuestion(
                question_id=question.id,
                question=question.text,
                evidence_key=question.evidence_key,
                information_gain_bits=0,
                current_entropy_bits=0,
                expected_posterior_entropy_bits=0,
                possible_answers=[],
                explanation="",
            ),
            answer,
        )

    @staticmethod
    def _explain_changes(
        previous_session: AgentSession,
        posteriors: dict[str, float],
        fired_rule_ids: list[str],
        graph_edges: list[str],
        ml_label: str | None,
    ) -> str:
        if not previous_session.previous_posteriors:
            return "Initial educational evaluation; no prior result exists for comparison."
        changed_probabilities = [
            f"{label}: {previous_session.previous_posteriors.get(label, 0):.3f} "
            f"to {probability:.3f}"
            for label, probability in posteriors.items()
            if abs(previous_session.previous_posteriors.get(label, 0) - probability) > 1e-9
        ]
        new_rules = sorted(set(fired_rule_ids) - set(previous_session.previous_fired_rule_ids))
        removed_rules = sorted(set(previous_session.previous_fired_rule_ids) - set(fired_rule_ids))
        new_edges = sorted(set(graph_edges) - set(previous_session.previous_graph_edges))
        removed_edges = sorted(set(previous_session.previous_graph_edges) - set(graph_edges))
        details: list[str] = []
        if previous_session.answered_question_ids:
            question_id = previous_session.answered_question_ids[-1]
            answer = previous_session.answers.get(question_id)
            if answer is not None:
                details.append(
                    f"Follow-up response recorded for {question_id}: {answer!r}."
                )
        if changed_probabilities:
            details.append(
                "Synthetic Bayesian values changed (" + "; ".join(changed_probabilities) + ")."
            )
        if new_rules:
            details.append("New synthetic rules fired: " + ", ".join(new_rules) + ".")
        if removed_rules:
            details.append("Previously fired synthetic rules no longer fired: " + ", ".join(removed_rules) + ".")
        if new_edges:
            details.append("New graph links visited: " + ", ".join(new_edges) + ".")
        if removed_edges:
            details.append("Graph links no longer visited: " + ", ".join(removed_edges) + ".")
        if ml_label != previous_session.previous_ml_label:
            details.append(
                "Synthetic ML label changed from "
                f"{previous_session.previous_ml_label or 'not available'} to "
                f"{ml_label or 'not available'}."
            )
        if len(details) == 1 and previous_session.answered_question_ids:
            details.append(
                "No modeled graph, symbolic, Bayesian, or ML output changed."
            )
        elif not details:
            details.append(
                "No modeled graph, symbolic, Bayesian, or ML output changed."
            )
        return " ".join(details)
