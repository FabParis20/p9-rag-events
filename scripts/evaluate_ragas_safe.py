import json
import os
import sys
from dotenv import load_dotenv
from datasets import Dataset
from langchain_anthropic import ChatAnthropic

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Ajouter le chemin parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.langchain_rag import PulsEventsRAG

# Charger les variables d'environnement
load_dotenv()

# Configuration du LLM pour Ragas
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
os.environ["OPENAI_API_KEY"] = "dummy"

# Fichiers de sauvegarde
TEMP_FILE = "data/evaluation/temp_rag_responses.json"
RESULTS_FILE = "data/evaluation/ragas_results.json"

# Charger le jeu de test
print("📂 Chargement du jeu de test...")
with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)
print(f"✅ {len(test_data['test_cases'])} questions chargées\n")

# Vérifier si on a déjà des réponses sauvegardées
if os.path.exists(TEMP_FILE):
    print("💾 Fichier de sauvegarde trouvé ! Chargement des réponses...")
    with open(TEMP_FILE, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    questions = saved_data["questions"]
    answers = saved_data["answers"]
    contexts = saved_data["contexts"]
    ground_truths = saved_data["ground_truths"]
    print(f"✅ {len(questions)} réponses chargées depuis la sauvegarde\n")
else:
    # Initialiser le RAG
    print("🚀 Initialisation du système RAG...")
    rag = PulsEventsRAG()
    print("✅ RAG prêt !\n")

    # Interroger le RAG
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("🔄 Interrogation du RAG pour chaque question...\n")
    for i, test_case in enumerate(test_data['test_cases'], 1):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        
        print(f"Question {i}/{len(test_data['test_cases'])}: {question}")
        
        result = rag.ask(question)
        answer = result['answer']
        context_list = result['sources']
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
        
        # Sauvegarder après chaque question
        temp_data = {
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }
        with open(TEMP_FILE, "w", encoding="utf-8") as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Réponse générée et sauvegardée\n")

# Créer le dataset pour Ragas
print("📊 Création du dataset pour Ragas...")
evaluation_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})
print(f"✅ Dataset créé avec {len(evaluation_dataset)} exemples\n")

# Lancer l'évaluation Ragas
print("🎯 Lancement de l'évaluation Ragas...")
print("⏳ Cela peut prendre 2-3 minutes...\n")

results = evaluate(
    evaluation_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=None
)

print("✅ Évaluation terminée !\n")

# Afficher les résultats
print("=" * 50)
print("📊 RÉSULTATS DE L'ÉVALUATION RAGAS")
print("=" * 50)
print(f"\n🎯 Faithfulness:       {results.to_pandas()['faithfulness'].mean():.3f}")
print(f"🎯 Answer Relevancy:   {results.to_pandas()['answer_relevancy'].mean():.3f}")
print(f"🎯 Context Precision:  {results.to_pandas()['context_precision'].mean():.3f}")
print(f"🎯 Context Recall:     {results.to_pandas()['context_recall'].mean():.3f}")
print("\n" + "=" * 50)

# Sauvegarder les résultats
results_dict = {
    "faithfulness": float(results.to_pandas()['faithfulness'].mean()),
    "answer_relevancy": float(results.to_pandas()['answer_relevancy'].mean()),
    "context_precision": float(results.to_pandas()['context_precision'].mean()),
    "context_recall": float(results.to_pandas()['context_recall'].mean()),
    "num_questions": len(test_data['test_cases'])
}

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés dans : {RESULTS_FILE}")

# Nettoyer le fichier temporaire
if os.path.exists(TEMP_FILE):
    os.remove(TEMP_FILE)
    print("🗑️  Fichier temporaire supprimé")

print("\n✅ Évaluation MVP7 terminée avec succès !")