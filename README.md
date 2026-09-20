# NSQA — Question-réponse sur graphe de connaissances (neuro-symbolisme)

Un **LLM** lit le texte et comprend les questions ; un **graphe de connaissances** garantit les faits ; des **règles
logiques** raisonnent et **prouvent**. Le LLM ne répond jamais lui-même : il traduit, et c'est le graphe qui répond.

Domaine de démonstration : sécurité médicamenteuse (corpus fictif, non validé médicalement).

## 1. Architecture

```
ENTRÉES
  corpus.txt (phrases) · ontology.ttl (schéma) · shapes.ttl (contraintes) · question en français
      │
      ▼
TRAITEMENT
  ① Extraction        phrases ─────────────► triplets (sujet, relation, objet)
  ② Construction      triplets + ontologie ─► graphe RDF
  ③ Validation        graphe ──────────────► conforme / violations
  ④ Inférence         graphe validé ───────► graphe enrichi + preuves
  ⑤ Complétion        graphe enrichi ──────► liens suggérés (hypothèses)
  ⑥ Traduction        question ────────────► requête SPARQL
  ⑦ Exécution         SPARQL + graphe ─────► réponse + arbre de preuve
      │
      ▼
SORTIES
  Réponse + preuve · graphe Turtle (asserté / inféré) · liens suggérés · tableau d'évaluation
```

## 2. Technologies et modèles par étape

| # | Étape | Technologie / modèle | Module |
|---|-------|----------------------|--------|
| ① | Extraction | **Mode `offline`** : regex Python (`re`).<br>**Mode `anthropic`** : LLM **Claude** (`claude-sonnet-5` par défaut, réglable via `NSQA_MODEL`) avec sortie JSON stricte | `extraction.py`, `llm.py` |
| ② | Construction | **RDFLib** (RDF, Turtle) ; ontologie en vocabulaire **RDFS/OWL** | `kg.py` |
| ③ | Validation | **pySHACL** (contraintes **SHACL**) | `validation.py` |
| ④ | Inférence | **Chaînage avant** écrit en Python sur RDFLib : 7 règles + provenance de chaque fait | `reasoning.py` |
| ⑤ | Complétion | **TransE** implémenté en **NumPy** (distance L1, perte à marge), filtré par domaine/portée de l'ontologie | `embeddings.py` |
| ⑥ | Traduction | **Mode `offline`** : gabarits regex.<br>**Mode `anthropic`** : **Claude** génère du SPARQL, refusé s'il n'est pas `SELECT`/`ASK` | `qa.py` |
| ⑦ | Exécution | Moteur **SPARQL de RDFLib** + explicateur d'arbre de preuve maison | `qa.py`, `reasoning.py` |
| — | Évaluation / tests | Métriques (précision, rappel, F1) maison ; **pytest** (25 tests) | `evaluate.py`, `tests/` |

Neuronal : ①, ⑤, ⑥ (en mode `offline`, ① et ⑥ sont remplacées par des règles). Symbolique : ②, ③, ④, ⑦.
Le raisonnement est fait par les règles maison, pas par un raisonneur OWL.

**Règles :** symétrie d'`interagitAvec` · transitivité de `subClassOf` · héritage des types · héritage des
contre-indications et des risques · alerte contre-indication · alerte interaction.

## 3. Installation

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
```

Aucune clé API n'est nécessaire en mode `offline` (par défaut).

## 4. Utilisation

```bash
nsqa demo                                    # démonstration
nsqa build                                   # construit, valide, infère, exporte output/*.ttl
nsqa ask "Quelles alertes pour Paul ?"
nsqa ask --no-reasoning "Avec quoi le warfarin interagit-il ?"   # sans les règles
nsqa evaluate                                # compare les systèmes
nsqa complete --relation traite              # liens suggérés (TransE)
pytest
```

Mode LLM réel :

```bash
pip install -e ".[llm]" && export ANTHROPIC_API_KEY="sk-ant-..."
nsqa --backend anthropic ask "Quelles alertes pour Marie ?"
nsqa --backend anthropic evaluate            # ajoute la baseline « LLM seul »
```

**Exemple de sortie :**

```text
$ nsqa ask "Quels médicaments sont contre-indiqués en cas d'ulcère gastrique ?"
Réponse : aspirine, ibuprofène.

Preuve pour « aspirine » :
• aspirine contreIndiquePour ulcère gastrique  [inféré par R4-héritage-contre-indication]
  • aspirine est de type AINS  [source : « L'aspirine est un AINS. »]
  • AINS contreIndiquePour ulcère gastrique  [source : « Les AINS sont contre-indiqués en cas d'ulcère gastrique. »]
```

Le corpus ne dit jamais que l'aspirine est contre-indiquée : la réponse est déduite, et chaque maillon renvoie à sa phrase source.

## 5. Évaluation

`nsqa evaluate` sur 13 questions (`data/questions.json`), mode `offline` :

```text
Système                      Précision     Rappel         F1      Exact
KG sans raisonnement              0.54       0.46       0.49       0.38
KG + raisonnement (hybride)       0.90       0.90       0.90       0.90
```

Ce jeu a été écrit avec le corpus et les règles : le 1.00 montre que la chaîne fonctionne, pas qu'elle généralise.
Le résultat utile est le gain dû aux règles (+0.51 de F1). En mode `anthropic`, une 3e ligne « LLM seul » est ajoutée.
Pour la complétion de liens sur données standards : `python scripts/eval_link_prediction.py --data <FB15k-237>`.

## 6. Limites

- Le mode `anthropic` n'a pas été exécuté sur l'API réelle (pas de clé lors du développement) ; il est testé avec un faux client.
- L'extracteur `offline` ne comprend que les phrases simples du corpus.
- TransE sur 36 faits ne produit que des hypothèses (parfois fausses) : à valider par un humain.
- Monde fermé : ce qui n'est pas dans le graphe est traité comme faux.

