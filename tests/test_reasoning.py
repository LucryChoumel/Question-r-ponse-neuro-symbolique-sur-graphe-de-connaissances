from rdflib import RDF

from nsqa.kg import NS
from nsqa.reasoning import Explainer


def uri(pipe, label):
    return pipe.kg.lookup(label)


def test_corpus_is_valid_and_facts_counted(pipe):
    assert pipe.report.conforms
    assert len(pipe.kg) == 25


def test_inheritance_of_contraindication(pipe):
    g = pipe.inference.graph
    assert (uri(pipe, "aspirine"), NS.contreIndiquePour, uri(pipe, "ulcère gastrique")) in g
    assert (uri(pipe, "paracétamol"), NS.contreIndiquePour, uri(pipe, "ulcère gastrique")) not in g


def test_symmetry_and_type_inheritance(pipe):
    g = pipe.inference.graph
    assert (uri(pipe, "warfarin"), NS.interagitAvec, uri(pipe, "aspirine")) in g
    assert (uri(pipe, "ibuprofène"), RDF.type, NS.Medicament) in g


def test_alerts(pipe):
    g = pipe.inference.graph
    marie, paul, sophie = (uri(pipe, n) for n in ("Marie", "Paul", "Sophie"))
    assert (marie, NS.alerteContreIndication, uri(pipe, "ibuprofène")) in g
    assert (paul, NS.alerteInteraction, uri(pipe, "warfarin")) in g
    assert not list(g.triples((sophie, NS.alerteContreIndication, None)))
    assert not list(g.triples((sophie, NS.alerteInteraction, None)))


def test_every_inferred_fact_has_a_proof_down_to_sources(pipe):
    explainer = Explainer(pipe.kg, pipe.inference)
    for triple in pipe.inference.provenance:
        lines = explainer.explain(triple)
        assert any("[source" in l or "[ontologie]" in l for l in lines)


def test_original_graph_untouched_by_inference(pipe):
    assert (uri(pipe, "warfarin"), NS.interagitAvec, uri(pipe, "aspirine")) not in pipe.kg.graph
