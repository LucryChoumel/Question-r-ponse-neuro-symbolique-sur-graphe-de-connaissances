"""Orchestration : texte -> triplets -> KG -> validation -> raisonnement -> question-réponse."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .embeddings import KGCompleter
from .extraction import Extractor, LLMExtractor, RuleBasedExtractor, read_sentences
from .kg import KnowledgeGraph, build_kg
from .llm import AnthropicClient
from .qa import Answer, LLMTranslator, RuleBasedTranslator, Translator, execute
from .reasoning import DEFAULT_RULES, InferenceResult, Rule, run_inference
from .validation import ValidationReport, validate_kg

log = logging.getLogger(__name__)


def default_data_dir() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / "data", Path.cwd() / "data"):
        if (candidate / "corpus.txt").exists():
            return candidate
    raise FileNotFoundError("Dossier data/ introuvable : utilisez --data-dir.")


@dataclass
class Pipeline:
    data_dir: Path = field(default_factory=default_data_dir)
    backend: str = "offline"  # "offline" | "anthropic"
    rules: tuple[Rule, ...] = DEFAULT_RULES

    kg: KnowledgeGraph | None = None
    report: ValidationReport | None = None
    inference: InferenceResult | None = None
    translator: Translator | None = None
    client: object | None = None

    def build(self) -> "Pipeline":
        self.data_dir = Path(self.data_dir)
        if self.backend == "anthropic":
            self.client = AnthropicClient()
            extractor: Extractor = LLMExtractor(self.client)
        elif self.backend == "offline":
            extractor = RuleBasedExtractor()
        else:
            raise ValueError(f"Backend inconnu : {self.backend}")

        sentences = read_sentences((self.data_dir / "corpus.txt").read_text(encoding="utf-8"))
        raw = [t for s in sentences for t in extractor.extract(s)]
        self.kg = build_kg(raw, self.data_dir / "ontology.ttl")
        self.report = validate_kg(self.kg, self.data_dir / "shapes.ttl")
        self.inference = run_inference(self.kg, self.rules)
        self.translator = (
            LLMTranslator(self.client, self.kg) if self.backend == "anthropic" else RuleBasedTranslator(self.kg)
        )
        return self

    def ask(self, question: str, reasoning: bool = True) -> Answer:
        """`reasoning=False` interroge uniquement les faits extraits (ablation)."""
        assert self.kg and self.inference, "Appelez build() d'abord."
        sparql = self.translator.translate(question)
        graph = self.inference.graph if reasoning else self.kg.graph
        return execute(question, sparql, graph, self.kg, self.inference if reasoning else None)

    def completer(self, **kwargs) -> KGCompleter:
        return KGCompleter(self.kg, graph=self.inference.graph, **kwargs)

    def export(self, out_dir: str | Path) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = [out / "kg_asserted.ttl", out / "kg_inferred.ttl"]
        self.kg.graph.serialize(files[0], format="turtle")
        self.inference.graph.serialize(files[1], format="turtle")
        return files
