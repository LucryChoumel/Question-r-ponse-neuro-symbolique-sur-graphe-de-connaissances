"""Couche NEURONALE (1/3) : du texte vers des triplets (sujet, prédicat, objet).

Deux extracteurs partagent la même interface :
  - RuleBasedExtractor : motifs en français, déterministe, sans clé API (mode démo/tests) ;
  - LLMExtractor       : appelle un LLM et exige une sortie JSON stricte.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from .llm import LLMClient
from .textutils import strip_determiner

log = logging.getLogger(__name__)

ALLOWED_PREDICATES = (
    "type",  # instance -> classe
    "subClassOf",  # classe -> classe
    "interagitAvec",
    "contreIndiquePour",
    "prend",
    "souffreDe",
    "traite",
    "augmenteLeRisqueDe",
)


@dataclass(frozen=True)
class RawTriple:
    subject: str
    predicate: str
    object: str
    source: str  # phrase d'origine (provenance)


class Extractor(Protocol):
    def extract(self, sentence: str) -> list[RawTriple]: ...


def read_sentences(text: str) -> list[str]:
    """Une phrase par ligne ; les lignes vides et les commentaires (#) sont ignorés."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# --------------------------------------------------------------------------- #
# Extracteur à base de règles
# --------------------------------------------------------------------------- #
_VERBS = r"(?:est|sont|interagit|interagissent|prend|prennent|souffre|souffrent|traite|traitent|augmente|augmentent)"
_SUBJECT = re.compile(
    rf"^(?:(?P<det>un|une|les|le|la)\s+|(?P<elide>l'))?(?P<name>.+?)\s+(?P<rest>{_VERBS}\b.*)$",
    re.IGNORECASE,
)
_OF = r"(?:de\s+|d'|du\s+|des\s+)"
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"^(?:est|sont)\s+contre-indiqu\w*\s+(?:en cas {_OF}|pour\s+)(.+)$", re.I), "contreIndiquePour"),
    (re.compile(r"^(?:est|sont)\s+(?:un|une|des)\s+(.+)$", re.I), "isa"),
    (re.compile(r"^(?:interagit|interagissent)\s+avec\s+(.+)$", re.I), "interagitAvec"),
    (re.compile(r"^(?:prend|prennent)\s+(.+)$", re.I), "prend"),
    (re.compile(rf"^(?:souffre|souffrent)\s+{_OF}(.+)$", re.I), "souffreDe"),
    (re.compile(r"^(?:traite|traitent)\s+(.+)$", re.I), "traite"),
    (re.compile(rf"^(?:augmente|augmentent)\s+le risque\s+{_OF}(.+)$", re.I), "augmenteLeRisqueDe"),
]


class RuleBasedExtractor:
    """Substitut déterministe d'un LLM pour les phrases simples du corpus de démo.

    Règle de lecture : un sujet introduit par « Un/Une/Les » désigne une CLASSE
    (-> subClassOf), un sujet en « Le/La/L' » ou un nom propre désigne une INSTANCE (-> type).
    """

    def extract(self, sentence: str) -> list[RawTriple]:
        s = sentence.strip().rstrip(".").replace("’", "'")
        m = _SUBJECT.match(s)
        if not m:
            log.warning("Phrase non analysée : %r", sentence)
            return []
        subject = m.group("name").strip()
        is_class = (m.group("det") or "").lower() in {"un", "une", "les"}
        rest = m.group("rest")
        for pattern, predicate in _PATTERNS:
            pm = pattern.match(rest)
            if pm:
                obj = strip_determiner(pm.group(1))
                if predicate == "isa":
                    predicate = "subClassOf" if is_class else "type"
                return [RawTriple(subject, predicate, obj, sentence)]
        log.warning("Verbe/structure non géré : %r", sentence)
        return []


# --------------------------------------------------------------------------- #
# Extracteur LLM
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "Tu extrais des triplets de connaissances d'une phrase médicale française. "
    "Réponds UNIQUEMENT par un tableau JSON d'objets {\"s\": ..., \"p\": ..., \"o\": ...}, sans texte autour. "
    f"Prédicats autorisés : {', '.join(ALLOWED_PREDICATES)}. "
    "'type' relie une instance à sa classe ; 'subClassOf' relie une classe à sa super-classe. "
    "Écris les entités au singulier, sans article (ex : 'ulcère gastrique'). "
    "N'invente rien : si la phrase ne contient pas de fait exploitable, renvoie []."
)


def parse_llm_triples(raw: str, source: str) -> list[RawTriple]:
    """Extrait et valide le JSON renvoyé par le LLM (tolère du texte parasite autour)."""
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        log.warning("JSON invalide pour %r", source)
        return []
    triples = []
    for it in items:
        if not isinstance(it, dict):
            continue
        s, p, o = (str(it.get(k, "")).strip() for k in ("s", "p", "o"))
        if s and o and p in ALLOWED_PREDICATES:
            triples.append(RawTriple(strip_determiner(s), p, strip_determiner(o), source))
    return triples


class LLMExtractor:
    """Un appel LLM par phrase (simple et traçable ; regroupez par lots pour de gros corpus)."""

    def __init__(self, client: LLMClient):
        self.client = client

    def extract(self, sentence: str) -> list[RawTriple]:
        raw = self.client.complete(_SYSTEM, sentence, max_tokens=500)
        return parse_llm_triples(raw, sentence)
