import json

import pytest
from pydantic import ValidationError

from mednex.engines.knowledge_graph import KnowledgeBaseDocument, KnowledgeGraph
from mednex.state import ExtractionResult, ReasoningState
from mednex.models import Symptom
from mednex.nlp import parse_patient_description


def test_loads_and_validates_synthetic_graph() -> None:
    knowledge_graph = KnowledgeGraph.load()

    assert knowledge_graph.graph.number_of_nodes() > 0
    assert knowledge_graph.graph.number_of_edges() > 0
    categories = {
        attributes["category"]
        for _, attributes in knowledge_graph.graph.nodes(data=True)
    }
    assert {
        "symptom",
        "observation",
        "candidate_condition",
        "investigation_concept",
        "risk_indicator",
    } <= categories


def test_retrieves_candidate_evidence_and_explains_visited_relationships() -> None:
    extraction = parse_patient_description(
        "A fictional patient has cough and fever. Temperature is 38.2 C."
    )
    result = KnowledgeGraph.load().investigate(extraction)

    candidate_ids = {candidate.node.id for candidate in result.candidates}
    assert "condition:synthetic_pattern_alpha" in candidate_ids
    alpha = next(
        item for item in result.candidates
        if item.node.id == "condition:synthetic_pattern_alpha"
    )
    assert {item.polarity for item in alpha.evidence} >= {"supporting"}
    assert all(item.rationale for item in alpha.evidence)
    assert all(edge.considered_because for edge in result.visited_edges)
    assert any(
        edge.relation == "investigation_relevant_to_question"
        for edge in result.visited_edges
    )
    assert "not a diagnosis" in result.explanation


def test_negated_symptom_is_not_treated_as_positive_evidence() -> None:
    extraction = parse_patient_description("The fictional patient has no cough.")
    result = KnowledgeGraph.load().investigate(extraction)

    cough_evidence = [
        evidence
        for candidate in result.candidates
        for evidence in candidate.evidence
        if evidence.source_node == "symptom:cough"
    ]
    assert cough_evidence
    assert all(item.negated_input for item in cough_evidence)
    assert all(item.polarity == "contextual" for item in cough_evidence)


def test_flags_input_quality_without_clinical_risk_interpretation() -> None:
    extraction = parse_patient_description(
        "The fictional patient has fever but no fever."
    )
    result = KnowledgeGraph.load().investigate(extraction)

    assert "risk:conflicting_input" in {
        node.id for node in result.visited_nodes
    }
    assert any("not a clinical risk" in node.description for node in result.visited_nodes)


def test_reports_unrepresented_symptom_as_missing_graph_node() -> None:
    extraction = ExtractionResult(
        state=ReasoningState(symptoms=[Symptom(name="unlisted fictional symptom")]),
        extraction_confidence=0.5,
    )

    result = KnowledgeGraph.load().investigate(extraction)

    assert result.missing_graph_nodes == ["symptom:unlisted_fictional_symptom"]
    assert result.candidates == []


def test_rejects_edge_that_references_missing_node() -> None:
    document = json.loads(
        (KnowledgeGraph.load().document.model_dump_json())
    )
    document["edges"][0]["target"] = "condition:not_in_graph"

    with pytest.raises(ValidationError, match="missing target node"):
        KnowledgeBaseDocument.model_validate(document)


def test_rejects_relationship_with_incompatible_node_categories() -> None:
    document = json.loads(KnowledgeGraph.load().document.model_dump_json())
    document["edges"][0]["relation"] = "observation_relevant_condition"

    with pytest.raises(ValidationError, match="requires observation"):
        KnowledgeBaseDocument.model_validate(document)


def test_visualizes_the_full_or_investigated_graph() -> None:
    knowledge_graph = KnowledgeGraph.load()
    extraction = parse_patient_description("A fictional patient has cough.")

    full_figure = knowledge_graph.visualize()
    visited_figure = knowledge_graph.visualize(
        knowledge_graph.investigate(extraction)
    )

    assert full_figure is not None
    assert visited_figure is not None
    import matplotlib.pyplot as plt

    plt.close(full_figure)
    plt.close(visited_figure)
