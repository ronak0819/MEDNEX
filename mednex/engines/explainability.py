from pydantic import Field

from mednex.engines.bayesian_engine import BayesianInferenceResult
from mednex.engines.information_engine import InformationGainResult
from mednex.engines.knowledge_graph import GraphInvestigationResult
from mednex.engines.ml_engine import MLPredictionResult
from mednex.engines.planner import DecisionPlan
from mednex.engines.symbolic_engine import SymbolicReasoningResult
from mednex.models import MEDNEXModel
from mednex.state import ExtractionResult


class ExplainableReport(MEDNEXModel):
    status: str
    summary: str
    extracted_facts: list[str] = Field(default_factory=list)
    missing_or_unmodeled: list[str] = Field(default_factory=list)
    graph_evidence: list[str] = Field(default_factory=list)
    symbolic_evidence: list[str] = Field(default_factory=list)
    bayesian_summary: list[str] = Field(default_factory=list)
    ml_summary: str | None = None
    uncertainty_summary: str
    next_action: str
    limitations: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "MEDNEX is an educational simulation only. This report is not a diagnosis "
        "or a substitute for professional care and contains no treatment instructions."
    )


def build_explainable_report(
    plan: DecisionPlan,
    extraction: ExtractionResult | None,
    graph: GraphInvestigationResult | None,
    symbolic: SymbolicReasoningResult | None,
    bayesian: BayesianInferenceResult | None,
    ml: MLPredictionResult | None,
    information: InformationGainResult | None,
    safety_message: str | None = None,
) -> ExplainableReport:
    facts: list[str] = []
    if extraction:
        facts.extend(
            f"Reported {'absence of ' if symptom.is_negated else ''}{symptom.name}"
            for symptom in extraction.state.symptoms
        )
        if extraction.state.patient_information.age_years is not None:
            facts.append(
                f"Explicitly stated age: {extraction.state.patient_information.age_years}"
            )
        if extraction.state.patient_information.sex:
            facts.append(
                f"Explicitly stated sex: {extraction.state.patient_information.sex}"
            )
        facts.extend(
            f"Explicit observation {observation.name}: {observation.value}"
            for observation in extraction.state.extracted_observations
            if observation.name != "symptom"
        )

    missing: list[str] = []
    if extraction:
        missing.extend(
            f"Ambiguous: {item}" for item in extraction.ambiguous_statements
        )
        missing.extend(
            f"Unsupported: {item}" for item in extraction.unsupported_statements
        )
        missing.extend(
            f"Contradictory: {item}" for item in extraction.contradictions
        )
    if bayesian:
        missing.extend(f"Not reported: {item}" for item in bayesian.missing_observations)
        missing.extend(f"Not modeled: {item}" for item in bayesian.unmodeled_observations)
        missing.extend(
            f"Conflicting: {item}" for item in bayesian.contradictory_observations
        )
    if graph:
        missing.extend(
            f"Missing knowledge-graph node: {node_id}"
            for node_id in graph.missing_graph_nodes
        )
    if symbolic:
        missing.extend(
            f"Required rule information missing: {item}"
            for item in symbolic.missing_information
        )

    graph_evidence = [
        f"{edge.source} -> {edge.target} ({edge.polarity}): {edge.rationale}"
        for edge in (graph.visited_edges if graph else [])
    ]
    symbolic_evidence = [
        f"{rule.rule_id} fired ({rule.polarity}) for {rule.assessment_signal}: "
        f"{rule.explanation}"
        for rule in (symbolic.fired_rules if symbolic else [])
    ]
    bayesian_summary = [
        f"{candidate.candidate_label}: prior={candidate.prior_probability:.3f}, "
        f"posterior={candidate.posterior_probability:.3f}"
        for candidate in (bayesian.candidates if bayesian else [])
    ]
    ml_summary = None
    if ml:
        ml_summary = (
            f"{ml.model_name} predicted synthetic label {ml.predicted_label}; "
            "its class probabilities are separate and are not combined with Bayesian values."
        )
    elif extraction:
        ml_summary = (
            "Machine-learning prediction was not run because the case contains "
            "no unambiguous symptoms supported by the synthetic training features."
        )
    uncertainty_parts: list[str] = []
    if bayesian:
        uncertainty_parts.append(
            f"Bayesian candidate entropy is "
            f"{information.current_entropy_bits:.3f} bits."
            if information
            else "Bayesian probabilities are conditional on synthetic parameters."
        )
    if symbolic and symbolic.conflicts:
        uncertainty_parts.append("Symbolic rules produced conflicting evidence.")
    if extraction and (extraction.ambiguous_statements or extraction.contradictions):
        uncertainty_parts.append("Some input cannot be interpreted unambiguously.")
    if safety_message:
        uncertainty_parts.append(safety_message)
    if not uncertainty_parts:
        uncertainty_parts.append(
            "No clinical certainty or clinical accuracy is implied by any component."
        )
    limitations = [
        "All graph, symbolic, Bayesian, and machine-learning evidence is synthetic and educational.",
        "No engine output is a confirmed diagnosis, treatment, or patient-specific recommendation.",
        "Bayesian and machine-learning probabilities are kept separate.",
    ]
    if information:
        limitations.extend(information.limitations)
    return ExplainableReport(
        status=plan.status,
        summary=plan.rationale,
        extracted_facts=facts,
        missing_or_unmodeled=missing,
        graph_evidence=graph_evidence,
        symbolic_evidence=symbolic_evidence,
        bayesian_summary=bayesian_summary,
        ml_summary=ml_summary,
        uncertainty_summary=" ".join(uncertainty_parts),
        next_action=plan.next_action,
        limitations=list(dict.fromkeys(limitations)),
    )
