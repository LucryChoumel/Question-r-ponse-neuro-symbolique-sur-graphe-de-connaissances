"""Interface en ligne de commande : `nsqa build|ask|evaluate|complete|demo`."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .evaluate import format_table, run_evaluation
from .kg import NS
from .pipeline import Pipeline, default_data_dir


def _pipeline(args) -> Pipeline:
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    return Pipeline(data_dir=data_dir, backend=args.backend).build()


def cmd_build(args) -> None:
    pipe = _pipeline(args)
    files = pipe.export(args.out)
    print(f"Faits extraits      : {len(pipe.kg)}")
    print(f"Faits inférés       : {pipe.inference.n_inferred} (en {pipe.inference.iterations} passes)")
    print(f"Validation SHACL    : {'OK' if pipe.report.conforms else 'ÉCHEC'}")
    for v in pipe.report.violations:
        print(f"  ! {v.focus} : {v.message}")
    print("Exports             : " + ", ".join(str(f) for f in files))


def cmd_ask(args) -> None:
    pipe = _pipeline(args)
    answer = pipe.ask(args.question, reasoning=not args.no_reasoning)
    if args.show_sparql and answer.sparql:
        print(answer.sparql.split("\n", 2)[-1], "\n")
    print(answer.text(with_proof=not args.no_proof))


def cmd_evaluate(args) -> None:
    pipe = _pipeline(args)
    q_path = Path(args.questions) if args.questions else pipe.data_dir / "questions.json"
    systems = run_evaluation(pipe, q_path)
    print(format_table(systems))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(systems, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_complete(args) -> None:
    pipe = _pipeline(args)
    completer = pipe.completer(epochs=args.epochs)
    relation = NS[args.relation]
    print(f"Liens suggérés pour « {args.relation} » (score TransE, filtrés par domaine/portée) :")
    for s in completer.suggest(relation, top_k=args.top_k):
        print(f"  {pipe.kg.label(s.head)} —{args.relation}→ {pipe.kg.label(s.tail)}   ({s.score:.2f})")


def cmd_demo(args) -> None:
    pipe = _pipeline(args)
    print("=" * 70)
    print(f"KG : {len(pipe.kg)} faits extraits, {pipe.inference.n_inferred} inférés ; SHACL : "
          f"{'OK' if pipe.report.conforms else 'ÉCHEC'}")
    print("=" * 70)
    for q in (
        "Quels médicaments sont contre-indiqués en cas d'ulcère gastrique ?",
        "Quelles alertes pour Paul ?",
        "L'ibuprofène est-il contre-indiqué en cas d'ulcère gastrique ?",
    ):
        print(f"\n> {q}")
        print(pipe.ask(q).text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nsqa", description=__doc__)
    parser.add_argument("--data-dir", help="dossier contenant corpus.txt, ontology.ttl, shapes.ttl")
    parser.add_argument("--backend", choices=["offline", "anthropic"], default="offline",
                        help="offline = règles locales (défaut) ; anthropic = LLM réel")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="construit le KG, valide, infère, exporte en Turtle")
    p.add_argument("--out", default="output")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="pose une question en français")
    p.add_argument("question")
    p.add_argument("--no-reasoning", action="store_true", help="ignore les règles (faits extraits seuls)")
    p.add_argument("--no-proof", action="store_true")
    p.add_argument("--show-sparql", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("evaluate", help="compare les systèmes sur data/questions.json")
    p.add_argument("--questions")
    p.add_argument("--out", default="output")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("complete", help="suggère des liens manquants (TransE + contraintes de type)")
    p.add_argument("--relation", default="traite")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.set_defaults(func=cmd_complete)

    p = sub.add_parser("demo", help="démonstration rapide")
    p.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
