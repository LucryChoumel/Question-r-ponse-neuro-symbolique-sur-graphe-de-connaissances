"""Petits utilitaires de normalisation de texte français."""
from __future__ import annotations

import re
import unicodedata


def normalize_apostrophes(s: str) -> str:
    return s.replace("’", "'").replace("`", "'")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def slugify(s: str) -> str:
    """'Ulcère gastrique' -> 'ulcere_gastrique' (identifiant stable pour les URI)."""
    s = strip_accents(normalize_apostrophes(s)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


# Déterminants : ceux qui finissent par une lettre exigent un espace (évite "dexaméthasone" -> "xaméthasone").
_DET = re.compile(
    r"^(?:(?:de la|du|des|de|le|la|les|un|une)\s+|(?:de l'|d'|l')\s*)+",
    re.IGNORECASE,
)


def strip_determiner(phrase: str) -> str:
    phrase = normalize_apostrophes(phrase).strip()
    return _DET.sub("", phrase).strip(" .;,")
