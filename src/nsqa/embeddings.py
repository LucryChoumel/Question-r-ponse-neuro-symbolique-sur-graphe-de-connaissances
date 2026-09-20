"""Couche NEURONALE (2/3) : plongements de graphe (TransE, NumPy pur) et complétion de liens.

Les suggestions du modèle sont FILTRÉES par les contraintes de type de l'ontologie
(domaine/portée) : le neuronal propose, le symbolique dispose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdflib import RDF, RDFS, URIRef

from .kg import NS, KnowledgeGraph


class TransE:
    """TransE (Bordes et al., 2013) : h + r ≈ t, distance L1, perte à marge, corruption aléatoire."""

    def __init__(self, dim=32, margin=1.0, lr=0.05, epochs=300, batch_size=64, seed=0):
        self.dim, self.margin, self.lr = dim, margin, lr
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.E: np.ndarray | None = None
        self.R: np.ndarray | None = None

    def fit(self, triples: np.ndarray, n_entities: int, n_relations: int) -> "TransE":
        rng = np.random.default_rng(self.seed)
        bound = 6.0 / np.sqrt(self.dim)
        self.E = rng.uniform(-bound, bound, (n_entities, self.dim))
        self.R = rng.uniform(-bound, bound, (n_relations, self.dim))
        self.R /= np.linalg.norm(self.R, axis=1, keepdims=True)
        n = len(triples)
        for _ in range(self.epochs):
            self.E /= np.maximum(np.linalg.norm(self.E, axis=1, keepdims=True), 1e-9)
            perm = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                pos = triples[perm[start : start + self.batch_size]]
                neg = pos.copy()
                corrupt_head = rng.random(len(pos)) < 0.5
                rand_ent = rng.integers(0, n_entities, len(pos))
                neg[corrupt_head, 0] = rand_ent[corrupt_head]
                neg[~corrupt_head, 2] = rand_ent[~corrupt_head]
                self._step(pos, neg)
        return self

    def _step(self, pos: np.ndarray, neg: np.ndarray) -> None:
        E, R = self.E, self.R
        d_pos = E[pos[:, 0]] + R[pos[:, 1]] - E[pos[:, 2]]
        d_neg = E[neg[:, 0]] + R[neg[:, 1]] - E[neg[:, 2]]
        loss = self.margin + np.abs(d_pos).sum(1) - np.abs(d_neg).sum(1)
        act = loss > 0
        if not act.any():
            return
        g_pos, g_neg = np.sign(d_pos[act]), np.sign(d_neg[act])
        pos, neg = pos[act], neg[act]
        lr = self.lr
        np.add.at(E, pos[:, 0], -lr * g_pos)
        np.add.at(E, pos[:, 2], lr * g_pos)
        np.add.at(E, neg[:, 0], lr * g_neg)
        np.add.at(E, neg[:, 2], -lr * g_neg)
        np.add.at(R, pos[:, 1], -lr * (g_pos - g_neg))

    # Scores : plus grand = plus plausible
    def score_tails(self, h: int, r: int) -> np.ndarray:
        return -np.abs(self.E[h] + self.R[r] - self.E).sum(1)

    def score_heads(self, r: int, t: int) -> np.ndarray:
        return -np.abs(self.E + self.R[r] - self.E[t]).sum(1)


def evaluate_link_prediction(
    model: TransE, test: np.ndarray, known: set[tuple[int, int, int]], ks=(1, 3, 10)
) -> dict[str, float]:
    """Classement FILTRÉ (les autres vrais triplets ne pénalisent pas le rang) : MRR et Hits@k."""
    ranks = []
    for h, r, t in test:
        for side in ("tail", "head"):
            scores = model.score_tails(h, r) if side == "tail" else model.score_heads(r, t)
            target = t if side == "tail" else h
            truth = scores[target]
            mask = np.zeros(len(scores), dtype=bool)
            for cand in range(len(scores)):
                if cand != target:
                    triple = (h, r, cand) if side == "tail" else (cand, r, t)
                    if triple in known:
                        mask[cand] = True
            scores = np.where(mask, -np.inf, scores)
            ranks.append(1 + int((scores > truth).sum()))
    ranks_arr = np.array(ranks)
    metrics = {"MRR": float((1.0 / ranks_arr).mean())}
    for k in ks:
        metrics[f"Hits@{k}"] = float((ranks_arr <= k).mean())
    return metrics


# --------------------------------------------------------------------------- #
# Complétion du graphe de connaissances
# --------------------------------------------------------------------------- #
@dataclass
class Suggestion:
    head: URIRef
    relation: URIRef
    tail: URIRef
    score: float


class KGCompleter:
    def __init__(self, kg: KnowledgeGraph, graph=None, **transe_kwargs):
        self.kg = kg
        self.graph = graph if graph is not None else kg.graph
        keep = (RDF.type, RDFS.subClassOf)
        triples = [
            (s, p, o)
            for s, p, o in self.graph
            if isinstance(s, URIRef) and isinstance(o, URIRef)
            and str(s).startswith(str(NS)) and str(o).startswith(str(NS))
            and (p in keep or str(p).startswith(str(NS)))
        ]
        self.entities = sorted({t[0] for t in triples} | {t[2] for t in triples}, key=str)
        self.relations = sorted({t[1] for t in triples}, key=str)
        self.e_id = {e: i for i, e in enumerate(self.entities)}
        self.r_id = {r: i for i, r in enumerate(self.relations)}
        self.ids = np.array([(self.e_id[s], self.r_id[p], self.e_id[o]) for s, p, o in triples])
        self.model = TransE(**transe_kwargs).fit(self.ids, len(self.entities), len(self.relations))

    def _instances_of(self, cls: URIRef) -> list[URIRef]:
        return [e for e in self.entities if (e, RDF.type, cls) in self.graph]

    def suggest(self, relation: URIRef, top_k: int = 5) -> list[Suggestion]:
        """Propose des liens manquants, restreints par domaine/portée déclarés dans l'ontologie."""
        if relation not in self.r_id:
            return []
        domain = self.graph.value(relation, RDFS.domain)
        rng_ = self.graph.value(relation, RDFS.range)
        heads = self._instances_of(domain) if domain else self.entities
        tails = self._instances_of(rng_) if rng_ else self.entities
        out = []
        for h in heads:
            scores = self.model.score_tails(self.e_id[h], self.r_id[relation])
            for t in tails:
                if t != h and (h, relation, t) not in self.graph:
                    out.append(Suggestion(h, relation, t, float(scores[self.e_id[t]])))
        out.sort(key=lambda s: -s.score)
        return out[:top_k]
