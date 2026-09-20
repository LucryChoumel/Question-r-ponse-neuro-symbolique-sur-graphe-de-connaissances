from nsqa.extraction import RawTriple
from nsqa.kg import build_kg
from nsqa.validation import validate_kg

from conftest import DATA


def test_shacl_rejects_ill_typed_triple():
    good = [
        RawTriple("Marie", "type", "patient", "s1"),
        RawTriple("aspirine", "type", "AINS", "s2"),
        RawTriple("AINS", "subClassOf", "médicament", "s3"),
    ]
    bad = RawTriple("Marie", "prend", "Paul", "s4")  # Paul n'est pas un médicament
    kg = build_kg(good + [bad], DATA / "ontology.ttl")
    report = validate_kg(kg, DATA / "shapes.ttl")
    assert not report.conforms
    assert any("médicament" in v.message for v in report.violations)


def test_shacl_accepts_well_typed_triples():
    triples = [
        RawTriple("Marie", "type", "patient", "s1"),
        RawTriple("aspirine", "type", "AINS", "s2"),
        RawTriple("AINS", "subClassOf", "médicament", "s3"),
        RawTriple("Marie", "prend", "aspirine", "s4"),
    ]
    kg = build_kg(triples, DATA / "ontology.ttl")
    assert validate_kg(kg, DATA / "shapes.ttl").conforms
