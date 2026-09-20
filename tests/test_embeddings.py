import numpy as np
from rdflib import RDF

from nsqa.embeddings import TransE, evaluate_link_prediction
from nsqa.kg import NS


def test_transe_learns_a_simple_pattern():
    # 10 entités en cycle : i --r0--> i+1. TransE doit classer le bon voisin dans le top 3.
    triples = np.array([(i, 0, (i + 1) % 10) for i in range(10)])
    model = TransE(dim=16, epochs=400, lr=0.05, seed=1).fit(triples, 10, 1)
    known = {tuple(map(int, t)) for t in triples}
    metrics = evaluate_link_prediction(model, triples, known, ks=(1, 3))
    assert metrics["Hits@3"] >= 0.8


def test_suggestions_respect_ontology_types(pipe):
    completer = pipe.completer(epochs=150)
    graph = pipe.inference.graph
    suggestions = completer.suggest(NS.traite, top_k=20)
    assert suggestions
    for s in suggestions:
        # tête = médicament, queue = pathologie (domaine/portée déclarés dans l'ontologie)
        assert (s.head, RDF.type, NS.Medicament) in graph
        assert (s.tail, RDF.type, NS.Pathologie) in graph
        # uniquement des liens NOUVEAUX
        assert (s.head, s.relation, s.tail) not in graph
