"""Couche NEURONALE (3/3) + SYMBOLIQUE : question en langage naturel -> SPARQL -> réponse + preuve.

Deux traducteurs question -> SPARQL partagent la même interface :
  - RuleBasedTranslator : gabarits en français (démo hors ligne) ;
  - LLMTranslator       : le LLM écrit la requête, validée avant exécution.
Dans les deux cas c'est le graphe (et ses règles) qui répond, jamais le LLM : pas d'hallucination de faits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from rdflib import RDF, Graph, URIRef, Variable
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.algebra import traverse

from .kg import NS, KnowledgeGraph
from .llm import LLMClient
from .reasoning import Explainer, InferenceResult
from .textutils import normalize_apostrophes, strip_accents, strip_determiner

PREFIXES = f"PREFIX nsqa: <{NS}>\nPREFIX rdf: <{RDF}>\n"


class Translator(Protocol):
    def translate(self, question: str) -> str | None: ...


# --------------------------------------------------------------------------- #
# Traducteur à gabarits
# --------------------------------------------------------------------------- #
_OF = r"(?:de |d')"


class RuleBasedTranslator:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        # (regex sur la question normalisée, constructeur de requête). Ordre = priorité.
        self.templates = [
            (r"^avec quoi (.+?) interagi\w*", self._interacts),
            (rf"^quels? medicaments? (?:sont|est) contre-indiques? (?:en cas {_OF}|pour )(.+)$", self._contraindicated),
            (rf"^quels? medicaments? augmentent? le risque {_OF}(.+)$", self._raises_risk),
            (r"^quels? medicaments? traitent? (.+)$", self._treated_by),
            (r"^quelles? alertes? (?:pour|concernent?) (.+)$", self._alerts),
            (r"^quelle est la classe (?:de |du |de la |de l')(.+)$", self._classes),
            (r"^que traite (.+)$", self._treats),
            (r"^qui prend (.+)$", self._who_takes),
            (rf"^(.+?) est-(?:il|elle) contre-indique\w* (?:en cas {_OF}|pour )(.+)$", self._is_contraindicated),
        ]

    @staticmethod
    def _normalize(q: str) -> str:
        q = strip_accents(normalize_apostrophes(q)).lower().strip()
        return q.rstrip(" ?!.")

    def _uri(self, phrase: str) -> URIRef | None:
        return self.kg.lookup(strip_determiner(phrase))

    def translate(self, question: str) -> str | None:
        q = self._normalize(question)
        for pattern, builder in self.templates:
            m = re.match(pattern, q)
            if m:
                uris = [self._uri(g) for g in m.groups()]
                if any(u is None for u in uris):
                    return None
                return PREFIXES + builder(*uris)
        return None

    # --- constructeurs de requêtes ---
    @staticmethod
    def _interacts(x):
        return f"SELECT ?r WHERE {{ <{x}> nsqa:interagitAvec ?r }}"

    @staticmethod
    def _contraindicated(c):
        return f"SELECT ?r WHERE {{ ?r a nsqa:Medicament ; nsqa:contreIndiquePour <{c}> }}"

    @staticmethod
    def _raises_risk(c):
        return f"SELECT ?r WHERE {{ ?r a nsqa:Medicament ; nsqa:augmenteLeRisqueDe <{c}> }}"

    @staticmethod
    def _treated_by(c):
        return f"SELECT ?r WHERE {{ ?r a nsqa:Medicament ; nsqa:traite <{c}> }}"

    @staticmethod
    def _alerts(pt):
        return (
            f"SELECT ?r WHERE {{ {{ <{pt}> nsqa:alerteContreIndication ?r }} "
            f"UNION {{ <{pt}> nsqa:alerteInteraction ?r }} }}"
        )

    @staticmethod
    def _classes(x):
        return f"SELECT ?r WHERE {{ <{x}> a ?r }}"

    @staticmethod
    def _treats(x):
        return f"SELECT ?r WHERE {{ <{x}> nsqa:traite ?r }}"

    @staticmethod
    def _who_takes(d):
        return f"SELECT ?r WHERE {{ ?r nsqa:prend <{d}> }}"

    @staticmethod
    def _is_contraindicated(d, c):
        return f"ASK {{ <{d}> nsqa:contreIndiquePour <{c}> }}"


# --------------------------------------------------------------------------- #
# Traducteur LLM
# --------------------------------------------------------------------------- #
class LLMTranslator:
    def __init__(self, client: LLMClient, kg: KnowledgeGraph):
        self.client, self.kg = client, kg

    def _system_prompt(self) -> str:
        entities = ", ".join(sorted(s for s in self.kg.aliases))
        return (
            "Tu traduis une question française en requête SPARQL (SELECT ou ASK) sur un graphe médical.\n"
            f"Préfixe : nsqa: <{NS}>. Classes : nsqa:Medicament, nsqa:Patient, nsqa:Pathologie.\n"
            "Relations : nsqa:interagitAvec, nsqa:contreIndiquePour, nsqa:prend, nsqa:souffreDe, nsqa:traite, "
            "nsqa:augmenteLeRisqueDe, nsqa:alerteContreIndication, nsqa:alerteInteraction ; 'a' = rdf:type.\n"
            f"Entités connues (identifiants nsqa:<id>) : {entities}.\n"
            "Nomme la variable de résultat ?r pour un SELECT. "
            "Réponds UNIQUEMENT par la requête, sans explication ni balises."
        )

    def translate(self, question: str) -> str | None:
        raw = self.client.complete(self._system_prompt(), question, max_tokens=400)
        raw = re.sub(r"```(?:sparql)?", "", raw).strip()
        if not re.search(r"^\s*(?:PREFIX[^\n]*\n\s*)*(SELECT|ASK)\b", raw, re.I):
            return None  # refuse tout ce qui n'est pas lecture seule
        if "nsqa:" in raw and "PREFIX nsqa" not in raw:
            raw = PREFIXES + raw
        return raw


# --------------------------------------------------------------------------- #
# Exécution + explication
# --------------------------------------------------------------------------- #
@dataclass
class Answer:
    question: str
    sparql: str | None = None
    labels: list[str] = field(default_factory=list)
    boolean: bool | None = None
    proofs: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None

    @property
    def values(self) -> list[str]:
        """Représentation comparable pour l'évaluation."""
        if self.boolean is not None:
            return ["oui" if self.boolean else "non"]
        return self.labels

    def text(self, with_proof: bool = True) -> str:
        if self.error:
            return f"Je ne peux pas répondre : {self.error}"
        if self.boolean is not None:
            head = "Oui." if self.boolean else "Non (aucune preuve dans le graphe)."
        elif self.labels:
            head = "Réponse : " + ", ".join(self.labels) + "."
        else:
            head = "Aucun résultat dans le graphe."
        if with_proof and self.proofs:
            body = []
            for label, lines in self.proofs.items():
                body.append(f"\nPreuve pour « {label} » :" if self.boolean is None else "\nPreuve :")
                body.extend(lines)
            head += "\n" + "\n".join(body)
        return head


def _bgp_patterns(sparql: str):
    algebra = prepareQuery(sparql, initNs={"nsqa": NS, "rdf": RDF}).algebra
    found: list[tuple] = []

    def visit(node):
        if getattr(node, "name", None) == "BGP":
            found.extend(node.triples)

    traverse(algebra, visitPost=visit)
    return found


def execute(
    question: str,
    sparql: str | None,
    graph: Graph,
    kg: KnowledgeGraph,
    inference: InferenceResult | None,
) -> Answer:
    if sparql is None:
        return Answer(question, error="question non comprise ou entité inconnue du graphe.")
    ans = Answer(question, sparql=sparql)
    try:
        result = graph.query(sparql)
        patterns = _bgp_patterns(sparql)
    except Exception as exc:  # requête invalide (typiquement issue d'un LLM)
        ans.error = f"requête SPARQL invalide ({exc.__class__.__name__})."
        return ans

    explainer = Explainer(kg, inference or InferenceResult(graph=graph)) if kg else None

    def proof_for(binding: dict) -> list[str]:
        lines, seen = [], set()
        for pat in patterns:
            terms = tuple(binding.get(t, t) if isinstance(t, Variable) else t for t in pat)
            if any(isinstance(t, Variable) for t in terms) or terms in seen:
                continue
            seen.add(terms)
            if terms in graph:
                lines.extend(explainer.explain(terms))
        return lines

    if result.type == "ASK":
        ans.boolean = bool(result.askAnswer)
        if ans.boolean:
            ans.proofs["oui"] = proof_for({})
        return ans

    var = result.vars[0]
    for row in result:
        node = row[var]
        if node is None:
            continue
        label = kg.label(node) if isinstance(node, URIRef) else str(node)
        if label not in ans.labels:
            ans.labels.append(label)
            ans.proofs[label] = proof_for({var: node})
    return ans
