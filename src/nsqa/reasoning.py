"""Couche SYMBOLIQUE : chaînage avant avec traçabilité des preuves.

Chaque fait inféré mémorise la règle appliquée et ses prémisses, ce qui permet de
reconstruire une explication complète pour chaque réponse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdflib import RDF, RDFS, Graph, Variable

from .kg import NS, KnowledgeGraph, Triple

Pattern = tuple  # (s, p, o) ; chaque terme est un URIRef ou une Variable


def V(name: str) -> Variable:
    return Variable(name)


@dataclass(frozen=True)
class Rule:
    name: str
    description: str
    body: tuple[Pattern, ...]
    head: Pattern


a, b, c, x, p, d, d1, d2 = (V(n) for n in ("a", "b", "c", "x", "p", "d", "d1", "d2"))

DEFAULT_RULES: tuple[Rule, ...] = (
    Rule(
        "R1-symétrie",
        "interagitAvec est symétrique",
        ((a, NS.interagitAvec, b),),
        (b, NS.interagitAvec, a),
    ),
    Rule(
        "R2-transitivité-sous-classe",
        "subClassOf est transitive",
        ((a, RDFS.subClassOf, b), (b, RDFS.subClassOf, c)),
        (a, RDFS.subClassOf, c),
    ),
    Rule(
        "R3-héritage-de-type",
        "une instance hérite des types de ses classes",
        ((x, RDF.type, a), (a, RDFS.subClassOf, b)),
        (x, RDF.type, b),
    ),
    Rule(
        "R4-héritage-contre-indication",
        "une instance hérite des contre-indications de sa classe",
        ((x, RDF.type, a), (a, NS.contreIndiquePour, c)),
        (x, NS.contreIndiquePour, c),
    ),
    Rule(
        "R5-héritage-risque",
        "une instance hérite des risques de sa classe",
        ((x, RDF.type, a), (a, NS.augmenteLeRisqueDe, c)),
        (x, NS.augmenteLeRisqueDe, c),
    ),
    Rule(
        "R6-alerte-contre-indication",
        "un patient qui prend un médicament contre-indiqué pour sa pathologie déclenche une alerte",
        ((p, NS.prend, d), (d, NS.contreIndiquePour, c), (p, NS.souffreDe, c)),
        (p, NS.alerteContreIndication, d),
    ),
    Rule(
        "R7-alerte-interaction",
        "un patient qui prend deux médicaments en interaction déclenche une alerte",
        ((p, NS.prend, d1), (p, NS.prend, d2), (d1, NS.interagitAvec, d2)),
        (p, NS.alerteInteraction, d1),
    ),
)


@dataclass
class Derivation:
    rule: str
    premises: tuple[Triple, ...]


@dataclass
class InferenceResult:
    graph: Graph  # faits assertés + inférés
    provenance: dict[Triple, Derivation] = field(default_factory=dict)
    iterations: int = 0

    @property
    def n_inferred(self) -> int:
        return len(self.provenance)


def _match(graph: Graph, body: tuple[Pattern, ...], binding: dict):
    """Générateur de (binding, prémisses) satisfaisant toutes les conditions du corps."""
    if not body:
        yield binding, []
        return
    first, rest = body[0], body[1:]
    terms = [binding.get(t, t) if isinstance(t, Variable) else t for t in first]
    query = tuple(None if isinstance(t, Variable) else t for t in terms)
    for triple in graph.triples(query):
        new = dict(binding)
        ok = True
        for term, value in zip(terms, triple):
            if isinstance(term, Variable):
                if new.get(term, value) != value:
                    ok = False
                    break
                new[term] = value
        if ok:
            for final, premises in _match(graph, rest, new):
                yield final, [triple, *premises]


def run_inference(
    kg: KnowledgeGraph, rules: tuple[Rule, ...] = DEFAULT_RULES, max_iter: int = 20
) -> InferenceResult:
    """Chaînage avant jusqu'au point fixe. Ne modifie pas le graphe d'origine."""
    graph = Graph()
    for t in kg.graph:
        graph.add(t)
    result = InferenceResult(graph=graph)
    for it in range(1, max_iter + 1):
        new_facts: dict[Triple, Derivation] = {}
        for rule in rules:
            for binding, premises in _match(graph, rule.body, {}):
                head = tuple(binding.get(t, t) if isinstance(t, Variable) else t for t in rule.head)
                if head not in graph and head not in new_facts:
                    new_facts[head] = Derivation(rule.name, tuple(premises))
        if not new_facts:
            break
        for triple, deriv in new_facts.items():
            graph.add(triple)
            result.provenance[triple] = deriv
        result.iterations = it
    return result


# --------------------------------------------------------------------------- #
# Explication
# --------------------------------------------------------------------------- #
_PRED_TEXT = {
    str(RDF.type): "est de type",
    str(RDFS.subClassOf): "est une sous-classe de",
}


class Explainer:
    def __init__(self, kg: KnowledgeGraph, inference: InferenceResult, max_depth: int = 6):
        self.kg, self.inf, self.max_depth = kg, inference, max_depth

    def sentence(self, t: Triple) -> str:
        s, pr, o = t
        pred = _PRED_TEXT.get(str(pr), str(pr).split("#")[-1])
        return f"{self.kg.label(s)} {pred} {self.kg.label(o)}"

    def explain(self, triple: Triple, depth: int = 0) -> list[str]:
        pad = "  " * depth
        deriv = self.inf.provenance.get(triple)
        if deriv is None:
            src = self.kg.sources.get(triple)
            suffix = f'  [source : « {src} »]' if src else "  [ontologie]"
            return [f"{pad}• {self.sentence(triple)}{suffix}"]
        lines = [f"{pad}• {self.sentence(triple)}  [inféré par {deriv.rule}]"]
        if depth < self.max_depth:
            for premise in deriv.premises:
                lines.extend(self.explain(premise, depth + 1))
        return lines
