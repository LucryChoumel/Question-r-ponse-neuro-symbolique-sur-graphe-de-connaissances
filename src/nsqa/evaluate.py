"""Évaluation : ablation « faits extraits seuls » vs « faits + règles » (+ LLM seul si disponible)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .pipeline import Pipeline
from .textutils import slugify


def _set(values) -> set[str]:
    return {slugify(v) for v in values}


def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_system(name: str, answers: list[tuple[list[str], dict, bool]], gold: list[dict]) -> dict:
    """answers[i] = (valeurs prédites, preuves, abstention)."""
    P = R = F = em = proved = non_empty = abst = 0
    for (values, proofs, abstained), g in zip(answers, gold):
        p, r, f = prf(_set(values), _set(g["expected"]))
        P += p; R += r; F += f
        em += int(_set(values) == _set(g["expected"]))
        abst += int(abstained)
        if values:
            non_empty += 1
            proved += int(all(proofs.get(v) for v in values) if proofs else False)
    n = len(gold)
    return {
        "système": name,
        "précision": P / n,
        "rappel": R / n,
        "F1": F / n,
        "exact_match": em / n,
        "abstention": abst / n,
        "réponses_avec_preuve": (proved / non_empty) if non_empty else 0.0,
    }


def _llm_only(client, question: str) -> list[str]:
    system = (
        "Réponds à la question médicale par un tableau JSON de chaînes courtes (noms uniquement). "
        "Pour une question oui/non, réponds [\"oui\"] ou [\"non\"]. Aucun autre texte."
    )
    raw = client.complete(system, question, max_tokens=200)
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return [str(x) for x in json.loads(m.group(0))] if m else []
    except json.JSONDecodeError:
        return []


def run_evaluation(pipe: Pipeline, questions_path: str | Path) -> list[dict]:
    gold = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    systems = []

    for name, reasoning in (("KG sans raisonnement", False), ("KG + raisonnement (hybride)", True)):
        answers = []
        for g in gold:
            a = pipe.ask(g["question"], reasoning=reasoning)
            answers.append((a.values, a.proofs, a.error is not None))
        systems.append(score_system(name, answers, gold))

    if pipe.client is not None:  # backend anthropic : baseline « LLM seul »
        answers = [(_llm_only(pipe.client, g["question"]), {}, False) for g in gold]
        systems.append(score_system("LLM seul (sans graphe)", answers, gold))
    return systems


def format_table(systems: list[dict]) -> str:
    cols = [
        ("précision", "Précision"), ("rappel", "Rappel"), ("F1", "F1"),
        ("exact_match", "Exact"), ("abstention", "Abstent."), ("réponses_avec_preuve", "Preuves"),
    ]
    width = max(len(s["système"]) for s in systems)
    lines = [f'{"Système":<{width}}  ' + "  ".join(f"{label:>9}" for _, label in cols)]
    for s in systems:
        lines.append(f'{s["système"]:<{width}}  ' + "  ".join(f"{s[key]:>9.2f}" for key, _ in cols))
    return "\n".join(lines)
