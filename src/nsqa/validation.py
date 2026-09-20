"""Garde-fou SYMBOLIQUE : les triplets extraits doivent respecter les contraintes SHACL."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pyshacl import validate
from rdflib import RDF, Graph, Namespace

from .kg import KnowledgeGraph

SH = Namespace("http://www.w3.org/ns/shacl#")


@dataclass
class Violation:
    focus: str
    message: str


@dataclass
class ValidationReport:
    conforms: bool
    violations: list[Violation] = field(default_factory=list)


def validate_kg(kg: KnowledgeGraph, shapes_path: str | Path) -> ValidationReport:
    shapes = Graph().parse(str(shapes_path), format="turtle")
    conforms, results_graph, _ = validate(
        kg.graph, shacl_graph=shapes, inference="none", abort_on_first=False
    )
    violations = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = results_graph.value(result, SH.focusNode)
        msg = results_graph.value(result, SH.resultMessage)
        violations.append(
            Violation(kg.label(focus) if focus is not None else "?", str(msg) if msg else "contrainte violée")
        )
    return ValidationReport(bool(conforms), violations)
