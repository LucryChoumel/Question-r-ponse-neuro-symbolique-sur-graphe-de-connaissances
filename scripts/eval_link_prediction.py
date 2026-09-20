#!/usr/bin/env python3
"""Benchmark de complétion de liens (TransE) sur un jeu standard au format TSV.

Format attendu : un fichier par split, lignes « tête<TAB>relation<TAB>queue ».
Exemple avec FB15k-237 ou WN18RR :

    python scripts/eval_link_prediction.py --data /chemin/FB15k-237 --epochs 200 --limit 1000

Le dossier doit contenir train.txt, valid.txt (optionnel) et test.txt.
Métriques : MRR et Hits@1/3/10 en classement filtré (protocole standard).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nsqa.embeddings import TransE, evaluate_link_prediction


def load(path: Path) -> list[tuple[str, str, str]]:
    rows = [line.rstrip("\n").split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [(h, r, t) for h, r, t in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None, help="n'évaluer que les N premiers triplets de test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    splits = {n: load(args.data / f"{n}.txt") for n in ("train", "valid", "test") if (args.data / f"{n}.txt").exists()}
    entities = sorted({x for rows in splits.values() for h, _, t in rows for x in (h, t)})
    relations = sorted({r for rows in splits.values() for _, r, _ in rows})
    e_id, r_id = {e: i for i, e in enumerate(entities)}, {r: i for i, r in enumerate(relations)}
    enc = {n: np.array([(e_id[h], r_id[r], e_id[t]) for h, r, t in rows]) for n, rows in splits.items()}
    known = {tuple(map(int, t)) for arr in enc.values() for t in arr}

    print(f"{len(entities)} entités, {len(relations)} relations, train={len(enc['train'])}")
    model = TransE(dim=args.dim, epochs=args.epochs, lr=args.lr, margin=args.margin, seed=args.seed)
    model.fit(enc["train"], len(entities), len(relations))
    test = enc["test"][: args.limit] if args.limit else enc["test"]
    metrics = evaluate_link_prediction(model, test, known)
    for k, v in metrics.items():
        print(f"{k:>8} : {v:.3f}")


if __name__ == "__main__":
    main()
