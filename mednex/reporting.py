from __future__ import annotations

from mednex.agent import AgentTurn


def _section(title: str, items: list[str]) -> list[str]:
    return [title, *(f"- {item}" for item in items)] if items else [title, "- None reported."]


def format_agent_report(turn: AgentTurn) -> str:
    """Build a plain-text report exclusively from the current AgentTurn outputs."""
    lines = [
        "MEDNEX | DETERMINISTIC EDUCATIONAL SIMULATION REPORT",
        "=" * 58,
        "NOT FOR REAL PATIENT DECISIONS. This is not a diagnosis, medical advice,",
        "triage, or treatment recommendation. All reasoning datasets and candidate",
        "labels are synthetic educational examples.",
        "",
        f"Simulation status: {turn.status}",
        f"Summary: {turn.report.summary}",
        f"Next action: {turn.report.next_action}",
        "",
        "Input supplied (fictional example only):",
        turn.session.case_description,
        "",
    ]

    if turn.extraction is None:
        lines.extend(
            [
                "Extraction and reasoning: Not run.",
                f"Safety boundary: {turn.safety.user_message or 'Evaluation stopped.'}",
                "",
            ]
        )
    else:
        lines.append(
            "Extraction clarity indicator: "
            f"{turn.extraction.extraction_confidence:.0%} "
            "(text extraction clarity only; not clinical confidence)"
        )
        lines.extend(_section("Extracted facts:", turn.report.extracted_facts))
        lines.append("")
        lines.extend(
            _section(
                "Missing, ambiguous, unsupported, or unmodeled information:",
                turn.report.missing_or_unmodeled,
            )
        )
        lines.append("")

    if turn.graph is not None:
        graph_candidates = [
            f"{candidate.node.label}: {candidate.node.description}"
            for candidate in turn.graph.candidates
        ]
        lines.extend(
            _section("Synthetic graph associations (not diagnoses):", graph_candidates)
        )
        lines.extend(
            _section("Graph relationships considered:", turn.report.graph_evidence)
        )
        lines.append("")
    if turn.symbolic is not None:
        rule_facts = [
            f"{rule.rule_id}: {rule.polarity} evidence for "
            f"{rule.assessment_signal} — {rule.explanation}"
            for rule in turn.symbolic.fired_rules
        ]
        lines.extend(_section("Synthetic symbolic evidence flags:", rule_facts))
        lines.extend(
            _section(
                "Symbolic conflicts:",
                [conflict.explanation for conflict in turn.symbolic.conflicts],
            )
        )
        lines.append("")
    if turn.bayesian is not None:
        bayesian_rows = [
            f"{candidate.candidate_label}: prior={candidate.prior_probability:.4f}, "
            f"posterior={candidate.posterior_probability:.4f}"
            for candidate in turn.bayesian.candidates
        ]
        lines.extend(
            _section(
                "Synthetic Bayesian calculations (not clinical risk estimates):",
                bayesian_rows,
            )
        )
        lines.extend(
            _section(
                "Bayesian evidence used:",
                [
                    f"{evidence.key}: {evidence.description} "
                    f"(source: {evidence.source_text})"
                    for evidence in turn.bayesian.evidence_used
                ],
            )
        )
        lines.append(turn.bayesian.conclusion_notice)
        lines.append("")
    if turn.ml_prediction is not None:
        lines.extend(
            [
                "Separate synthetic ML output (not a clinical prediction):",
                f"- Model: {turn.ml_prediction.model_name}",
                f"- Educational label: {turn.ml_prediction.predicted_label}",
                "- Class probabilities: "
                + ", ".join(
                    f"{label}={probability:.4f}"
                    for label, probability in turn.ml_prediction.class_probabilities.items()
                ),
                turn.ml_prediction.separation_notice,
                "",
            ]
        )
    if turn.ml_evaluation is not None:
        metrics = [
            f"{metric.model_name}: accuracy={metric.accuracy:.3f}, "
            f"precision={metric.precision_macro:.3f}, recall={metric.recall_macro:.3f}, "
            f"F1={metric.f1_macro:.3f}"
            for metric in turn.ml_evaluation.metrics
        ]
        lines.extend(
            _section("Synthetic dataset holdout metrics (not clinical performance):", metrics)
        )
        lines.append("")
    if turn.information_gain is not None:
        ranked_questions = [
            f"{index}. {question.question} "
            f"(expected information gain {question.information_gain_bits:.4f} bits): "
            f"{question.explanation}"
            for index, question in enumerate(
                turn.information_gain.ranked_questions, start=1
            )
        ]
        lines.extend(
            _section("Ranked educational follow-up questions:", ranked_questions)
        )
        lines.append("")

    lines.extend(
        [
            f"Uncertainty: {turn.report.uncertainty_summary}",
            f"Change explanation: {turn.change_explanation}",
            "",
            "Medication and treatment:",
            "MEDNEX does not recommend medicines, doses, or patient-specific treatment.",
            "",
        ]
    )
    lines.extend(_section("Limitations:", turn.report.limitations))
    lines.extend(["", f"Disclaimer: {turn.report.disclaimer}"])
    return "\n".join(lines)
