"""Le graphe de connaissances (RDF) : ontologie + faits extraits + provenance."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

from .extraction import RawTriple
from .textutils import slugify

NS = Namespace("http://example.org/nsqa#")

PREDICATES: dict[str, URIRef] = {
    "type": RDF.type,
    "subClassOf": RDFS.subClassOf,
    "interagitAvec": NS.interagitAvec,
    "contreIndiquePour": NS.contreIndiquePour,
    "prend": NS.prend,
    "souffreDe": NS.souffreDe,
    "traite": NS.traite,
    "augmenteLeRisqueDe": NS.augmenteLeRisqueDe,
}

Triple = tuple[URIRef, URIRef, URIRef]


@dataclass
class KnowledgeGraph:
    graph: Graph
    aliases: dict[str, URIRef] = field(default_factory=dict)  # slug -> URI
    sources: dict[Triple, str] = field(default_factory=dict)  # triplet -> phrase d'origine
    asserted: set[Triple] = field(default_factory=set)  # faits extraits (hors ontologie)

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_ontology(cls, ontology_path: str | Path) -> "KnowledgeGraph":
        g = Graph()
        g.bind("nsqa", NS)
        g.parse(str(ontology_path), format="turtle")
        kg = cls(graph=g)
        # Les libellés de l'ontologie servent d'alias (« patiente » -> nsqa:Patient).
        for prop in (RDFS.label, SKOS.altLabel):
            for s, _, lit in g.triples((None, prop, None)):
                if isinstance(s, URIRef):
                    kg.aliases[slugify(str(lit))] = s
        return kg

    def entity(self, label: str) -> URIRef:
        """Résout un libellé vers une URI, en créant l'entité si elle est nouvelle."""
        slug = slugify(label)
        if slug in self.aliases:
            return self.aliases[slug]
        uri = NS[slug]
        self.aliases[slug] = uri
        self.graph.add((uri, RDFS.label, Literal(label, lang="fr")))
        return uri

    def lookup(self, phrase: str) -> URIRef | None:
        """Comme `entity` mais sans créer : renvoie None si l'entité est inconnue."""
        slug = slugify(phrase)
        for candidate in (slug, slug.rstrip("s")):
            if candidate in self.aliases:
                return self.aliases[candidate]
        return None

    def add(self, raw: RawTriple) -> Triple | None:
        pred = PREDICATES.get(raw.predicate)
        if pred is None:
            return None
        triple: Triple = (self.entity(raw.subject), pred, self.entity(raw.object))
        self.graph.add(triple)
        self.asserted.add(triple)
        self.sources.setdefault(triple, raw.source)
        return triple

    # ---- lecture --------------------------------------------------------- #
    def label(self, node: URIRef) -> str:
        lit = self.graph.value(node, RDFS.label)
        return str(lit) if lit is not None else str(node).split("#")[-1]

    def __len__(self) -> int:
        return len(self.asserted)


def build_kg(raw_triples: list[RawTriple], ontology_path: str | Path) -> KnowledgeGraph:
    kg = KnowledgeGraph.from_ontology(ontology_path)
    for raw in raw_triples:
        kg.add(raw)
    return kg
