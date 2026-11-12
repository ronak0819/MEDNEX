from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import Field, model_validator

from mednex.models import MEDNEXModel
from mednex.state import ExtractionResult

NodeCategory = Literal[
    "symptom",
    "observation",
    "candidate_condition",
    "investigation_concept",
    "risk_indicator",
    "assessment_question",
]
EvidencePolarity = Literal["supporting", "conflicting", "contextual", "none"]
EdgeRelation = Literal[
    "symptom_associated_condition",
    "observation_relevant_condition",
    "condition_has_question",
    "investigation_relevant_to_question",
    "risk_indicator_for_question",
]

RELATION_ENDPOINTS: dict[str, tuple[str, str]] = {
    "symptom_associated_condition": ("symptom", "candidate_condition"),
    "observation_relevant_condition": ("observation", "candidate_condition"),
    "condition_has_question": ("candidate_condition", "assessment_question"),
    "investigation_relevant_to_question": (
        "investigation_concept",
        "assessment_question",
    ),
    "risk_indicator_for_question": ("risk_indicator", "assessment_question"),
}
DEFAULT_KNOWLEDGE_BASE = (
    Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"
)


class KnowledgeNode(MEDNEXModel):
    id: str = Field(min_length=1)
    category: NodeCategory
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)


class KnowledgeEdge(MEDNEXModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: EdgeRelation
    polarity: EvidencePolarity
    rationale: str = Field(min_length=1)


class KnowledgeBaseDocument(MEDNEXModel):
    description: str = Field(min_length=1)
    nodes: list[KnowledgeNode]
    edges: list[KnowledgeEdge]

    @model_validator(mode="after")
    def validate_graph_references_and_types(self) -> KnowledgeBaseDocument:
        nodes_by_id: dict[str, KnowledgeNode] = {}
        for node in self.nodes:
            if node.id in nodes_by_id:
                raise ValueError(f"Duplicate knowledge graph node id: {node.id}")
            nodes_by_id[node.id] = node

        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.source not in nodes_by_id:
                raise ValueError(f"Knowledge graph edge references missing source node: {edge.source}")
            if edge.target not in nodes_by_id:
                raise ValueError(f"Knowledge graph edge references missing target node: {edge.target}")
            expected_source, expected_target = RELATION_ENDPOINTS[edge.relation]
            actual_source = nodes_by_id[edge.source].category
            actual_target = nodes_by_id[edge.target].category
            if (actual_source, actual_target) != (expected_source, expected_target):
                raise ValueError(
                    f"Edge '{edge.relation}' requires {expected_source} -> "
                    f"{expected_target}, got {actual_source} -> {actual_target}."
                )
            if edge.relation in {
                "condition_has_question",
                "investigation_relevant_to_question",
                "risk_indicator_for_question",
            } and edge.polarity != "contextual":
                raise ValueError(
                    f"Edge '{edge.relation}' must use contextual polarity."
                )
            if edge.relation in {
                "symptom_associated_condition",
                "observation_relevant_condition",
            } and edge.polarity == "none":
                raise ValueError(
                    f"Evidence edge '{edge.relation}' must define an evidence polarity."
                )
            key = (edge.source, edge.target, edge.relation)
            if key in edge_keys:
                raise ValueError(f"Duplicate knowledge graph edge: {key}")
            edge_keys.add(key)
        return self


class VisitedNode(MEDNEXModel):
    id: str
    category: NodeCategory
    label: str
    description: str


class VisitedEdge(MEDNEXModel):
    source: str
    target: str
    relation: EdgeRelation
    polarity: EvidencePolarity
    rationale: str
    considered_because: str


class GraphEvidence(MEDNEXModel):
    source_node: str
    target_condition: str
    relation: EdgeRelation
    polarity: EvidencePolarity
    rationale: str
    matched_input: str
    negated_input: bool = False


class CandidateCondition(MEDNEXModel):
    node: VisitedNode
    evidence: list[GraphEvidence] = Field(default_factory=list)


class GraphInvestigationResult(MEDNEXModel):
    candidates: list[CandidateCondition] = Field(default_factory=list)
    visited_nodes: list[VisitedNode] = Field(default_factory=list)
    visited_edges: list[VisitedEdge] = Field(default_factory=list)
    missing_graph_nodes: list[str] = Field(default_factory=list)
    explanation: str = (
        "These are graph associations for a synthetic educational exercise. "
        "Connectivity is not a diagnosis, probability, ranking, or validated guideline."
    )


def _symptom_node_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"symptom:{slug}"


def _effective_polarity(polarity: EvidencePolarity, negated: bool) -> EvidencePolarity:
    if not negated:
        return polarity
    return "contextual"


class KnowledgeGraph:
    """Validated NetworkX graph backed by the local synthetic knowledge base."""

    def __init__(self, document: KnowledgeBaseDocument) -> None:
        self.document = document
        self.graph = nx.DiGraph()
        for node in document.nodes:
            self.graph.add_node(node.id, **node.model_dump())
        for edge in document.edges:
            self.graph.add_edge(edge.source, edge.target, **edge.model_dump())

    @classmethod
    def load(cls, path: str | Path | None = None) -> KnowledgeGraph:
        knowledge_path = Path(path) if path is not None else DEFAULT_KNOWLEDGE_BASE
        with knowledge_path.open("r", encoding="utf-8") as knowledge_file:
            raw_document = json.load(knowledge_file)
        document = KnowledgeBaseDocument.model_validate(raw_document)
        return cls(document)

    def _node(self, node_id: str) -> VisitedNode:
        attributes = self.graph.nodes[node_id]
        return VisitedNode(
            id=node_id,
            category=attributes["category"],
            label=attributes["label"],
            description=attributes["description"],
        )

    def _edge(self, source: str, target: str, considered_because: str) -> VisitedEdge:
        attributes = self.graph.edges[source, target]
        return VisitedEdge(
            source=source,
            target=target,
            relation=attributes["relation"],
            polarity=attributes["polarity"],
            rationale=attributes["rationale"],
            considered_because=considered_because,
        )

    def investigate(self, extraction: ExtractionResult) -> GraphInvestigationResult:
        visited_node_ids: set[str] = set()
        visited_edge_keys: dict[tuple[str, str], VisitedEdge] = {}
        missing_nodes: list[str] = []
        evidence_by_condition: dict[str, list[GraphEvidence]] = {}
        candidate_ids: set[str] = set()

        def add_missing(node_id: str) -> None:
            if node_id not in missing_nodes:
                missing_nodes.append(node_id)

        def consider_edge(
            source: str,
            target: str,
            considered_because: str,
        ) -> None:
            visited_node_ids.update((source, target))
            visited_edge_keys[(source, target)] = self._edge(
                source, target, considered_because
            )

        def add_observation_evidence(
            node_id: str,
            matched_input: str,
        ) -> None:
            if node_id not in self.graph:
                add_missing(node_id)
                return
            visited_node_ids.add(node_id)
            for _, condition_id, attributes in self.graph.out_edges(
                node_id, data=True
            ):
                if attributes["relation"] != "observation_relevant_condition":
                    continue
                considered_because = (
                    f"Input explicitly contained {matched_input!r}, which maps to "
                    f"observation node {node_id}."
                )
                consider_edge(node_id, condition_id, considered_because)
                candidate_ids.add(condition_id)
                evidence_by_condition.setdefault(condition_id, []).append(
                    GraphEvidence(
                        source_node=node_id,
                        target_condition=condition_id,
                        relation=attributes["relation"],
                        polarity=attributes["polarity"],
                        rationale=attributes["rationale"],
                        matched_input=matched_input,
                    )
                )

        for symptom in extraction.state.symptoms:
            node_id = _symptom_node_id(symptom.name)
            if node_id not in self.graph:
                add_missing(node_id)
                continue
            visited_node_ids.add(node_id)
            for _, condition_id, attributes in self.graph.out_edges(
                node_id, data=True
            ):
                if attributes["relation"] != "symptom_associated_condition":
                    continue
                considered_because = (
                    f"Extracted symptom {symptom.name!r}"
                    f"{' was negated' if symptom.is_negated else ''} maps to "
                    f"symptom node {node_id}."
                )
                consider_edge(node_id, condition_id, considered_because)
                candidate_ids.add(condition_id)
                evidence_by_condition.setdefault(condition_id, []).append(
                    GraphEvidence(
                        source_node=node_id,
                        target_condition=condition_id,
                        relation=attributes["relation"],
                        polarity=_effective_polarity(
                            attributes["polarity"], symptom.is_negated
                        ),
                        rationale=attributes["rationale"],
                        matched_input=symptom.name,
                        negated_input=symptom.is_negated,
                    )
                )

        for observation in extraction.state.extracted_observations:
            if observation.name == "symptom":
                continue
            node_id = f"observation:{observation.name}"
            add_observation_evidence(node_id, observation.value)

        for condition_id in sorted(candidate_ids):
            for _, question_id, attributes in self.graph.out_edges(
                condition_id, data=True
            ):
                if attributes["relation"] != "condition_has_question":
                    continue
                consider_edge(
                    condition_id,
                    question_id,
                    f"Explored the educational assessment question attached to "
                    f"candidate node {condition_id}.",
                )
                for investigation_id, _, investigation_attributes in self.graph.in_edges(
                    question_id, data=True
                ):
                    if investigation_attributes["relation"] != "investigation_relevant_to_question":
                        continue
                    consider_edge(
                        investigation_id,
                        question_id,
                        f"Included this information-gathering concept because it "
                        f"addresses question node {question_id}.",
                    )

        if extraction.ambiguous_statements or extraction.unsupported_statements:
            self._visit_input_quality_flag(
                "risk:uncertain_input",
                "Input ambiguity or unsupported wording prompted a data-quality flag.",
                visited_node_ids,
                visited_edge_keys,
            )
        if extraction.contradictions:
            self._visit_input_quality_flag(
                "risk:conflicting_input",
                "Contradictory extracted statements prompted a data-quality flag.",
                visited_node_ids,
                visited_edge_keys,
            )

        candidates = [
            CandidateCondition(
                node=self._node(condition_id),
                evidence=evidence_by_condition.get(condition_id, []),
            )
            for condition_id in sorted(candidate_ids)
        ]
        return GraphInvestigationResult(
            candidates=candidates,
            visited_nodes=[
                self._node(node_id) for node_id in sorted(visited_node_ids)
            ],
            visited_edges=[
                visited_edge_keys[key] for key in sorted(visited_edge_keys)
            ],
            missing_graph_nodes=missing_nodes,
        )

    def _visit_input_quality_flag(
        self,
        risk_node_id: str,
        reason: str,
        visited_node_ids: set[str],
        visited_edge_keys: dict[tuple[str, str], VisitedEdge],
    ) -> None:
        if risk_node_id not in self.graph:
            return
        visited_node_ids.add(risk_node_id)
        for _, question_id, attributes in self.graph.out_edges(
            risk_node_id, data=True
        ):
            if attributes["relation"] != "risk_indicator_for_question":
                continue
            visited_node_ids.add(question_id)
            visited_edge_keys[(risk_node_id, question_id)] = self._edge(
                risk_node_id, question_id, reason
            )
            for investigation_id, _, investigation_attributes in self.graph.in_edges(
                question_id, data=True
            ):
                if investigation_attributes["relation"] != "investigation_relevant_to_question":
                    continue
                visited_node_ids.add(investigation_id)
                visited_edge_keys[(investigation_id, question_id)] = self._edge(
                    investigation_id,
                    question_id,
                    f"Included this clarification concept because input-quality "
                    f"question node {question_id} was flagged.",
                )

    def visualize(self, result: GraphInvestigationResult | None = None):
        """Return a Matplotlib figure of the full graph or visited investigation."""
        import matplotlib.pyplot as plt

        if result is None:
            graph = self.graph
        else:
            graph = nx.DiGraph()
            for node in result.visited_nodes:
                graph.add_node(node.id, **node.model_dump())
            for edge in result.visited_edges:
                graph.add_edge(edge.source, edge.target, **edge.model_dump())

        category_colors = {
            "symptom": "#8ecae6",
            "observation": "#bde0a8",
            "candidate_condition": "#ffd166",
            "investigation_concept": "#cdb4db",
            "risk_indicator": "#ffadad",
            "assessment_question": "#e9ecef",
        }
        colors = [
            category_colors[graph.nodes[node_id]["category"]]
            for node_id in graph.nodes
        ]
        labels = {
            node_id: graph.nodes[node_id]["label"] for node_id in graph.nodes
        }
        figure, axis = plt.subplots(figsize=(13, 9))
        positions = nx.spring_layout(graph, seed=7)
        nx.draw_networkx(
            graph,
            pos=positions,
            labels=labels,
            node_color=colors,
            node_size=2400,
            font_size=8,
            arrows=True,
            ax=axis,
        )
        edge_labels = {
            (source, target): attributes["relation"].replace("_", " ")
            for source, target, attributes in graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            graph, pos=positions, edge_labels=edge_labels, font_size=6, ax=axis
        )
        axis.set_axis_off()
        axis.set_title("MEDNEX synthetic educational knowledge graph")
        figure.tight_layout()
        return figure
