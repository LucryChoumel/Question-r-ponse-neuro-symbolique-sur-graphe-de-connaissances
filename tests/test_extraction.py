import json

from nsqa.extraction import LLMExtractor, RuleBasedExtractor, parse_llm_triples
from nsqa.textutils import slugify, strip_determiner


def triples(sentence):
    return [(t.subject, t.predicate, t.object) for t in RuleBasedExtractor().extract(sentence)]


def test_slugify_and_determiners():
    assert slugify("Ulcère gastrique") == "ulcere_gastrique"
    assert strip_determiner("de l'ibuprofène") == "ibuprofène"
    assert strip_determiner("d'ulcère gastrique") == "ulcère gastrique"
    assert strip_determiner("dexaméthasone") == "dexaméthasone"  # pas de faux positif sur « de »


def test_instance_vs_class():
    assert triples("L'aspirine est un AINS.") == [("aspirine", "type", "AINS")]
    assert triples("Un AINS est un médicament.") == [("AINS", "subClassOf", "médicament")]


def test_relations():
    assert triples("Marie prend de l'ibuprofène.") == [("Marie", "prend", "ibuprofène")]
    assert triples("Marie souffre d'ulcère gastrique.") == [("Marie", "souffreDe", "ulcère gastrique")]
    assert triples("Les AINS sont contre-indiqués en cas d'ulcère gastrique.") == [
        ("AINS", "contreIndiquePour", "ulcère gastrique")
    ]
    assert triples("Un AINS augmente le risque de saignement.") == [("AINS", "augmenteLeRisqueDe", "saignement")]


def test_unparseable_sentence_is_skipped():
    assert triples("Il pleut sur Yaoundé.") == []


def test_llm_output_is_validated():
    raw = 'Voici : [{"s":"le paracétamol","p":"traite","o":"la douleur"},{"s":"x","p":"inventé","o":"y"}]'
    out = parse_llm_triples(raw, "src")
    assert [(t.subject, t.predicate, t.object) for t in out] == [("paracétamol", "traite", "douleur")]
    assert parse_llm_triples("pas de json", "src") == []


class FakeClient:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, user, max_tokens=1000):
        return self.reply


def test_llm_extractor_with_fake_client():
    reply = json.dumps([{"s": "aspirine", "p": "type", "o": "AINS"}])
    out = LLMExtractor(FakeClient(reply)).extract("L'aspirine est un AINS.")
    assert out[0].predicate == "type" and out[0].source == "L'aspirine est un AINS."
