from typing import Literal

from pydantic import Field

from mednex.engines.bayesian_engine import BayesianInferenceResult
from mednex.engines.information_engine import InformationGainResult
from mednex.engines.symbolic_engine import SymbolicReasoningResult
from mednex.models import MEDNEXModel
from mednex.safety.safety_checks import SafetyReview
from mednex.safety.validation import InputValidationResult
from mednex.state import ExtractionResult

AgentStatus = Literal[
    "NEEDS_MORE_INFORMATION",
    "EVALUATION_COMPLETE",
    "UNCERTAIN",
    "PROFESSIONAL_ASSESSMENT_REQUIRED",
]


class DecisionPlan(MEDNEXModel):
    status: AgentStatus
    next_question_id: str | None = None
    next_question: str | None = None
    rationale: str
    next_action: str
    completed: bool


def create_decision_plan(
    extraction: ExtractionResult | None,
    validation: InputValidationResult | None,
    safety: SafetyReview,
    symbolic: SymbolicReasoningResult | None = None,
    bayesian: BayesianInferenceResult | None = None,
    information: InformationGainResult | None = None,
    follow_up_count: int = 0,
    max_follow_up_questions: int = 3,
) -> DecisionPlan:
    if safety.professional_assessment_required:
        return DecisionPlan(
            status="PROFESSIONAL_ASSESSMENT_REQUIRED",
            rationale="A safety boundary was triggered; the simulation must not proceed.",
            next_action=safety.user_message or "Stop the educational evaluation.",
            completed=True,
        )
    if extraction is None or validation is None or not validation.valid:
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="Input validation did not permit a safe educational evaluation.",
            next_action="Revise the description to a concise fictional educational case.",
            completed=True,
        )
    if extraction.contradictions:
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="Contradictory input prevents unambiguous interpretation.",
            next_action="Clarify the conflicting statements before continuing.",
            completed=True,
        )
    if extraction.ambiguous_statements:
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="Ambiguous statements cannot safely be treated as facts.",
            next_action="Clarify or omit the ambiguous statements.",
            completed=True,
        )
    if extraction.unsupported_statements:
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="Unsupported statements cannot be evaluated by the modeled components.",
            next_action="Clarify the unsupported wording or use only supported fictional observations.",
            completed=True,
        )
    if information and information.ranked_questions:
        if follow_up_count < max_follow_up_questions:
            question = information.ranked_questions[0]
            return DecisionPlan(
                status="NEEDS_MORE_INFORMATION",
                next_question_id=question.question_id,
                next_question=question.question,
                rationale=(
                    "A remaining synthetic educational question has positive "
                    "expected information gain; asking one question preserves "
                    "the one-at-a-time interaction."
                ),
                next_action="Ask the highest-ranked educational follow-up question.",
                completed=False,
            )
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="The follow-up limit was reached while useful information remains.",
            next_action="Stop and retain the result as uncertain.",
            completed=True,
        )
    if bayesian is None or symbolic is None:
        return DecisionPlan(
            status="UNCERTAIN",
            rationale="The required reasoning outputs are unavailable.",
            next_action="Stop without producing an assessment signal.",
            completed=True,
        )
    return DecisionPlan(
        status="EVALUATION_COMPLETE",
        rationale=(
            "No remaining modeled educational follow-up question offers positive "
            "expected information gain. Residual candidate uncertainty is retained."
        ),
        next_action=(
            "Present the evidence summary and remaining uncertainty without "
            "forcing a candidate assessment."
        ),
        completed=True,
    )
