import json

from nsqa.evaluate import run_evaluation
from nsqa.qa import LLMTranslator, execute

from conftest import DATA


def test_reasoning_needed_question(pipe):
    q = "Quels médicaments sont contre-indiqués en cas d'ulcère gastrique ?"
    assert set(pipe.ask(q).labels) == {"aspirine", "ibuprofène"}
    assert pipe.ask(q, reasoning=False).labels == []  # sans règles : rien


def test_yes_no(pipe):
    assert pipe.ask("L'ibuprofène est-il contre-indiqué en cas d'ulcère gastrique ?").boolean is True
    assert pipe.ask("Le paracétamol est-il contre-indiqué en cas d'ulcère gastrique ?").boolean is False


def test_answer_carries_proof(pipe):
    a = pipe.ask("Quelles alertes pour Marie ?")
    assert a.labels == ["ibuprofène"]
    assert any("R6-alerte-contre-indication" in l for l in a.proofs["ibuprofène"])


def test_unknown_entity_abstains(pipe):
    a = pipe.ask("Que traite la licorne ?")
    assert a.error and not a.labels


def test_unparseable_question_abstains(pipe):
    assert pipe.ask("Quel temps fait-il ?").error


def test_hybrid_beats_kg_without_reasoning(pipe):
    systems = run_evaluation(pipe, DATA / "questions.json")
    baseline, hybrid = systems[0], systems[1]
    assert hybrid["F1"] > baseline["F1"]
    assert hybrid["exact_match"] == 1.0


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, max_tokens=1000):
        return self.reply


def test_llm_translator_runs_generated_sparql(pipe):
    sparql = "```sparql\nSELECT ?r WHERE { nsqa:paracetamol nsqa:traite ?r }\n```"
    tr = LLMTranslator(FakeClient(sparql), pipe.kg)
    out = tr.translate("Que traite le paracétamol ?")
    ans = execute("q", out, pipe.inference.graph, pipe.kg, pipe.inference)
    assert ans.labels == ["douleur"]


def test_llm_translator_rejects_updates(pipe):
    tr = LLMTranslator(FakeClient("DELETE WHERE { ?s ?p ?o }"), pipe.kg)
    assert tr.translate("Efface tout") is None


def test_invalid_sparql_is_reported_not_raised(pipe):
    ans = execute("q", "SELECT ?r WHERE { nsqa:x nsqa:y", pipe.inference.graph, pipe.kg, pipe.inference)
    assert ans.error
