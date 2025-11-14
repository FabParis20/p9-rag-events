# 🎭 Puls-Events - Système RAG pour Événements Culturels Parisiens

![CI/CD](https://github.com/FabParis20/p9-rag-events/actions/workflows/ci.yml/badge.svg)

Système de recommandation intelligent d'événements culturels parisiens utilisant RAG (Retrieval-Augmented Generation). Le chatbot répond aux questions sur les événements culturels à Paris en s'appuyant sur une base de 100 événements réels issus de l'API OpenAgenda.

---

## 🚀 Quick Start
```bash
# 1. Cloner le projet
git clone https://github.com/FabParis20/p9-rag-events.git
cd p9-rag-events

# 2. Configurer les clés API
cp .env.example .env
# Éditer .env avec les clés VOYAGE_API_KEY et ANTHROPIC_API_KEY

# 3. Lancer avec Docker
docker-compose up

# 4. Tester l'API
# Ouvrir http://localhost:8000/docs (Swagger UI)
# Ou utiliser curl:
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts de jazz ce week-end ?"}'
```

**Note** : L'API nécessite des clés API valides (Voyage AI + Anthropic). Les résultats pré-calculés sont disponibles dans `data/processed/` et `data/evaluation/` pour évaluation sans régénération.

---

## 📊 Architecture

### Schéma UML
*(Voir [architecture_uml_v2.mmd](docs/architecture_uml_v2.mmd))*

### Pipeline RAG
**Données** → **Preprocessing** (chunking 229 chunks) → **Embeddings** (Voyage AI 512D) → **Vector Store** (Faiss) → **Retrieval** (Top-3) → **Generation** (Claude Sonnet 4.5) → **API REST** (FastAPI)

### Composants principaux
- **Data Loader** : Récupération OpenAgenda + nettoyage HTML + chunking intelligent
- **Embeddings** : Voyage AI (voyage-3-lite) pour vectorisation sémantique
- **Vector Store** : Faiss IndexFlatL2 pour recherche par similarité
- **RAG Orchestrator** : LangChain + historique de conversation
- **API** : FastAPI avec endpoints `/ask` et `/health`
- **Deployment** : Docker optimisé (500MB, build 2-3 min)

---

## 🛠️ Technologies

| Composant | Technologie |
|-----------|-------------|
| Embeddings | Voyage AI (voyage-3-lite) |
| Vector Store | Faiss |
| LLM Generation | Claude Sonnet 4.5 (Anthropic) |
| Orchestration | LangChain |
| API Framework | FastAPI |
| Deployment | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Tests | pytest |
| Évaluation | Ragas |

---

## 📂 Structure du Projet
```
p9-rag-events/
├── api/                        # API FastAPI
│   └── main.py                 # Endpoints /ask, /health
├── rag/                        # Modules RAG
│   ├── data_loader.py          # Chargement + preprocessing
│   └── langchain_rag.py        # Orchestrateur principal
├── data/
│   ├── processed/
│   │   ├── events_real.json    # 100 événements réels
│   │   └── faiss_index/        # Index vectoriel (229 chunks)
│   └── evaluation/
│       ├── test_set.json       # 13 questions annotées
│       └── ragas_results.json  # Scores d'évaluation
├── scripts/                    # Scripts utilitaires
│   ├── fetch_openagenda.py     # Récupération API
│   ├── reindex_with_chunks.py  # Réindexation
│   └── evaluate_ragas_safe.py  # Évaluation Ragas
├── tests/                      # Tests unitaires
│   └── test_data_loader.py     # 5 tests pytest
├── .github/workflows/
│   └── ci.yml                  # Pipeline CI/CD
├── Dockerfile                  # Image Docker optimisée
├── docker-compose.yml          # Orchestration
└── requirements.txt            # Dépendances (version Docker)
```

---

## ✅ Tests & Évaluation

### Tests Unitaires
```bash
# En local (avec uv)
uv run pytest tests/ -v

# Ou via Docker
docker-compose run puls-events-api pytest tests/ -v
```

**Résultat** : 5 tests passent ✅ (chargement, nettoyage HTML, chunking)

### Évaluation Ragas
```bash
# Reproduire l'évaluation
uv run python scripts/evaluate_ragas_safe.py
```

**Résultats** (13 questions annotées) :
- **Faithfulness** : 0.545 (fidélité aux documents sources)
- **Answer Relevancy** : NaN (problème technique identifié)
- **Context Precision** : 0.111 (précision du retrieval)
- **Context Recall** : 0.141 (rappel du retrieval)

**Interprétation** : Le système génère des réponses fidèles aux sources mais le retrieval nécessite optimisation (meilleur chunking, filtrage temporel programmatique).

---

## 📦 Livrables

| Livrable | Localisation |
|----------|--------------|
| **Système RAG fonctionnel** | Code complet dans `rag/` et `api/` |
| **API REST** | `api/main.py` + Docker deployment |
| **Rapport technique** | `docs/rapport_technique.pdf` |
| **Tests unitaires** | `tests/test_data_loader.py` |
| **Jeu de test annoté** | `data/evaluation/test_set.json` |
| **Résultats évaluation** | `data/evaluation/ragas_results.json` |
| **CI/CD** | `.github/workflows/ci.yml` |

---

## 🔑 Configuration

### Variables d'environnement (.env)
```bash
VOYAGE_API_KEY=pa-xxx          # Embeddings Voyage AI
ANTHROPIC_API_KEY=sk-ant-xxx   # Generation Claude
```

**Note** : Un fichier `.env.example` est fourni comme template. Ces services nécessitent une inscription (Voyage AI offre 200M tokens gratuits pour voyage-3-lite).

---

## 🎯 Choix Techniques Clés

- **Voyage AI** : Spécialisé embeddings (meilleur que modèles généralistes)
- **Claude Sonnet 4.5** : Stabilité + qualité génération (pivot depuis Mistral)
- **Chunking** : 229 chunks (800 car.) pour embeddings plus précis
- **Prompt intelligent** : Priorité événements futurs, indication claire si passés
- **Docker optimisé** : Image 7x plus légère (500MB vs 7GB initial)

---

## 📈 Perspectives d'Amélioration

- **Pagination API** : Récupérer plus de 100 événements
- **Filtrage temporel programmatique** : Éviter événements passés
- **Fine-tuning prompt** : Améliorer scores Ragas
- **Expansion géographique** : Au-delà de Paris
- **Production** : Authentification, monitoring, scalabilité

---

## 👨‍💻 Auteur

**Fabrice VANSPEYBROCK** - Projet 9 OpenClassrooms ML Engineer  
📧 Contact via GitHub : [FabParis20](https://github.com/FabParis20)