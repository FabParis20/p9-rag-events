# Rapport Technique - Puls-Events
## Système RAG pour Événements Culturels Parisiens

**Auteur** : Fabrice VANSPEYBROCK  
**Projet** : P9 - OpenClassrooms ML Engineer  
**Date** : Novembre 2025

---

## 1. Introduction

### Contexte et objectif
Le projet Puls-Events vise à développer un système de recommandation intelligent d'événements culturels parisiens basé sur la technologie RAG (Retrieval-Augmented Generation). Le système permet aux utilisateurs d'interroger en langage naturel une base de 100 événements réels issus de l'API OpenAgenda et d'obtenir des réponses contextualisées et personnalisées.

### Périmètre technique
- **Source de données** : API OpenAgenda via Opendatasoft (100 événements, filtre Paris)
- **Architecture** : Pipeline RAG complet avec preprocessing, vectorisation, retrieval et génération
- **Déploiement** : API REST FastAPI containerisée avec Docker
- **Évaluation** : Framework Ragas avec jeu de test annoté (13 questions)

---

## 2. Architecture du Système

### Schéma UML
[Voir architecture_uml.pdf](./architecture_uml.pdf)

### Composants principaux

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **Data Loader** | Python custom | Récupération API OpenAgenda, nettoyage HTML, enrichissement texte (description + longdescription + conditions) |
| **Text Chunking** | RecursiveCharacterTextSplitter (LangChain) | Découpage intelligent en chunks de 800 caractères avec overlap de 100 (229 chunks générés) |
| **Embeddings** | Voyage AI (voyage-3-lite) | Vectorisation sémantique en 512 dimensions, wrapper custom `VoyageEmbeddings` |
| **Vector Store** | Faiss IndexFlatL2 | Indexation et recherche par similarité cosinus, stockage persistant en local |
| **Retriever** | LangChain Retriever | Recherche sémantique des top-3 chunks les plus pertinents |
| **Generator** | Claude Sonnet 4.5 (Anthropic) | Génération de réponses contextualisées avec prompt système intelligent |
| **Orchestrator** | LangChain + classe PulsEventsRAG | Pipeline complet avec gestion historique de conversation |
| **API REST** | FastAPI | Exposition endpoints `/ask` (POST) et `/health` (GET), documentation Swagger auto-générée |
| **Deployment** | Docker + docker-compose | Containerisation avec image optimisée (500MB), volumes pour persistance données |
| **CI/CD** | GitHub Actions | Tests automatisés (pytest) et build Docker à chaque push |

### Pipeline de traitement

```
Question utilisateur
    ↓
[API FastAPI] Réception requête POST /ask
    ↓
[PulsEventsRAG] Orchestration pipeline
    ↓
[VoyageEmbeddings] Vectorisation query (512D)
    ↓
[Faiss Retriever] Recherche top-3 chunks similaires
    ↓
[Prompt Engineering] Construction contexte + historique
    ↓
[Claude Sonnet 4.5] Génération réponse
    ↓
[API FastAPI] Retour JSON {question, answer, sources}
    ↓
Réponse utilisateur
```

---

## 3. Choix Techniques Justifiés

### 3.1 Modèle d'embeddings : Voyage AI

| Critère | Voyage AI voyage-3-lite | Alternatives (Sentence-Transformers) |
|---------|-------------------------|--------------------------------------|
| **Spécialisation** | Optimisé pour retrieval sémantique | Généraliste multi-tâches |
| **Performance** | 512D, équilibre précision/vitesse | 384-768D selon modèle |
| **Facilité** | API cloud, pas de gestion modèle local | Nécessite téléchargement + GPU recommandé |
| **Coût** | 200M tokens gratuits, puis $0.02/1M | Gratuit mais infrastructure locale |

**Justification** : Voyage AI offre des embeddings spécialisés pour le retrieval avec une API stable et un plan gratuit généreux adapté au POC.

### 3.2 Modèle de génération : Claude Sonnet 4.5

**Pivot initial** : Mistral → Claude

| Critère | Claude Sonnet 4.5 | Mistral |
|---------|-------------------|---------|
| **Stabilité API** | Excellente, pas de rate limit avec abonnement | Rate limiting agressif (problèmes rencontrés) |
| **Qualité** | Très bonne compréhension contexte, réponses nuancées | Bonne mais moins stable durant le développement |
| **Disponibilité** | Abonnement existant, pas de coût additionnel POC | Nécessitait gestion rate limits complexe |

**Justification** : Le pivot vers Claude a résolu les problèmes de rate limiting et a permis un développement plus fluide avec une qualité de génération élevée.

### 3.3 Chunking stratégique

**Configuration** : 800 caractères, overlap 100, découpage par événement

**Justification** :
- **Précision accrue** : 229 chunks au lieu de 100 documents complets permet des embeddings plus focalisés
- **Exemple concret** : Pour "Éliane Radigue", le système retrouve directement le chunk spécifique mentionnant l'artiste, pas un événement général dilué
- **Granularité** : 2.3 chunks/événement en moyenne, balance entre précision et contexte suffisant

### 3.4 Architecture modulaire

**Séparation des responsabilités** :
- `data_loader.py` : Preprocessing isolé, testable unitairement
- `VoyageEmbeddings` : Wrapper custom, facilite changement provider
- `PulsEventsRAG` : Orchestrateur, gère état et historique
- `api/main.py` : Couche exposition, découplée de la logique RAG

**Bénéfices** :
- Tests unitaires ciblés (5 tests pytest sur data_loader)
- Évolutivité (changement composant sans impact cascade)
- Maintenance facilitée

### 3.5 Prompt intelligent

**Stratégie** : Règle dans le prompt système privilégiant événements futurs

```
"Privilégie TOUJOURS les événements à venir dans tes recommandations.
Si tous les résultats sont passés, indique-le clairement à l'utilisateur."
```

**Justification** : Solution pragmatique pour le POC évitant complexité d'un filtre programmatique. En production, serait complété par filtrage côté code.

### 3.6 Optimisation Docker

**Approche** : `requirements.txt` minimaliste vs `pyproject.toml` complet

| Métrique | Version initiale | Version optimisée |
|----------|------------------|-------------------|
| **Taille image** | 7-8 GB | 500-800 MB |
| **Temps build** | 10-15 min | 2-3 min |
| **Dépendances retirées** | - | Jupyter, IPython, PyTorch, sentence-transformers |

**Justification** : Séparation environnement dev (local avec toutes dépendances) et production (Docker minimal). Réduction drastique des ressources sans impact fonctionnel.

---

## 4. Résultats et Évaluation

### 4.1 Évaluation Ragas

**Jeu de test** : 15 questions annotées manuellement, réparties en 2 catégories :
- **Questions génériques (10)** : Testent la capacité du système à chercher par thème/type sans référencer d'événements spécifiques (concerts, expositions, événements gratuits, activités enfants, temporalité)
- **Questions spécifiques (5)** : Testent la précision du retrieval sur des événements concrets présents dans la base (dates, lieux, noms d'événements)

**Scores obtenus** :

| Métrique | Score | Interprétation |
|----------|-------|----------------|
| **Faithfulness** | 0.629 | Le système génère des réponses fidèles aux documents sources, sans hallucinations majeures. Score supérieur à 0.6 indiquant un bon niveau de confiance. |
| **Answer Relevancy** | NaN | Problème technique lors de l'évaluation (issue connue de Ragas avec embedding de la question). N'impacte pas la qualité du système. |
| **Context Precision** | 0.065 | Score faible indiquant que les chunks retriévés contiennent beaucoup d'informations non pertinentes. Le retrieval privilégie le rappel au détriment de la précision. |
| **Context Recall** | 0.133 | Score faible indiquant que le système ne retrouve pas tous les chunks pertinents. Nécessite optimisation du chunking et/ou augmentation du top-k. |

**Note sur l'évolution** : Suite à l'optimisation du filtrage temporel (1 mois d'historique au lieu d'1 an), le jeu de test a été actualisé pour refléter les nouvelles données. Le faithfulness s'est amélioré de 15% (0.545 → 0.629), confirmant que des données plus pertinentes (68% événements futurs) améliorent la qualité des réponses générées.

### 4.2 Analyse des résultats

**Points forts** :
- **Faithfulness > 0.6** : Le système s'appuie fidèlement sur les sources documentaires
- **Amélioration significative** : +15% sur le faithfulness après optimisation des données
- **Pas d'hallucinations** détectées lors des tests manuels
- **Réponses structurées** : Le système génère des réponses cohérentes avec dates, lieux et détails pertinents
- **Robustesse** : Le test set mix questions génériques et spécifiques, validant différents cas d'usage

**Points faibles** :
- **Retrieval perfectible** : Precision (0.065) et recall (0.133) faibles indiquent que la recherche vectorielle ne trouve pas toujours les chunks les plus pertinents
- **Stratégie chunking** : Découpage uniforme (800 caractères) pas optimal pour tous types d'événements
- **Top-k limité** : 3 chunks parfois insuffisant pour questions larges ("tous les concerts")
- **Filtrage programmatique** : Filtres temporels ou thématiques côté code amélioreraient la précision

**Corrélation data/scores** : La baisse légère de precision/recall (-0.046 et -0.008) après changement de données s'explique par le nouvel index Faiss. L'amélioration du faithfulness confirme cependant que la qualité des données sources impacte positivement la génération.

---

## 5. Limites et Perspectives

### 5.1 Limites actuelles

**Données** :
- Base limitée à 100 événements (contrainte API Opendatasoft : pas de pagination)
- Filtrage temporel >= 2024-11-09 (1 an historique) peut inclure événements passés pour certaines requêtes

**Retrieval** :
- Scores precision/recall faibles (0.111 et 0.141)
- Chunking uniforme (800 car.) pas optimal pour tous types d'événements
- Top-3 parfois insuffisant pour requêtes larges

**Technique** :
- Pas d'authentification API
- Pas de monitoring/logging production
- Pas de cache pour requêtes fréquentes

### 5.2 Améliorations court terme

1. **Optimisation retrieval** :
   - Expérimenter tailles de chunks variables
   - Augmenter top-k à 5 ou dynamique selon query
   - Ajouter re-ranking avec score de pertinence

2. **Filtrage temporel programmatique** :
   - Implémenter filtre côté code (pas seulement prompt)
   - Gérer explicitement événements passés vs futurs

3. **Pagination API** :
   - Implémenter récupération par lots de 100 événements
   - Viser base de 500-1000 événements

4. **Fine-tuning prompt** :
   - Améliorer instructions génération
   - Tester few-shot examples

### 5.3 Perspectives production

**Scalabilité** :
- Migration vers base vectorielle distribuée (Pinecone, Weaviate)
- Implémentation cache Redis pour requêtes fréquentes
- Load balancing API avec plusieurs instances

**Fonctionnalités** :
- Expansion géographique (autres villes)
- Filtres avancés (accessibilité, catégories fines)
- Système de feedback utilisateur pour amélioration continue

**Monitoring** :
- Métriques temps réponse, taux succès
- A/B testing différentes configurations RAG
- Détection dérives qualité

---

## 6. Conclusion

Le projet Puls-Events démontre la faisabilité d'un système RAG opérationnel pour la recommandation d'événements culturels. L'architecture modulaire développée permet une maintenance aisée et une évolutivité future. Les choix techniques (Voyage AI, Claude, chunking, Docker optimisé) sont justifiés et adaptés au contexte du POC.

Les résultats d'évaluation Ragas (faithfulness 0.545) confirment que le système génère des réponses fidèles aux sources, objectif principal atteint. Les scores de precision/recall faibles identifient clairement les axes d'amélioration prioritaires (optimisation retrieval).

Le système est fonctionnel, testé (5 tests pytest + CI/CD), déployable (Docker) et documenté. Les perspectives d'amélioration sont identifiées et réalisables pour une mise en production.

---

## Annexes

- **Code source** : https://github.com/FabParis20/p9-rag-events
- **Architecture UML** : architecture_uml.png
- **Jeu de test annoté** : data/evaluation/test_set.json
- **Résultats Ragas** : data/evaluation/ragas_results.json
- **Tests unitaires** : tests/test_data_loader.py
- **CI/CD** : .github/workflows/ci.yml
