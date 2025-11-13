import json
import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from langchain_anthropic import ChatAnthropic

# Configuration du LLM pour Ragas
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")

# Charger les variables d'environnement
load_dotenv()

# Charger le jeu de test annoté
print("📂 Chargement du jeu de test...")
with open("data/evaluation/test_set.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"✅ {len(test_data['test_cases'])} questions chargées\n")

# Importer le système RAG
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.langchain_rag import PulsEventsRAG

# Initialiser le RAG
print("🚀 Initialisation du système RAG...")
rag = PulsEventsRAG()
print("✅ RAG prêt !\n")

# Préparer les listes pour Ragas
questions = []
answers = []
contexts = []
ground_truths = []

print("🔄 Interrogation du RAG pour chaque question...\n")
for i, test_case in enumerate(test_data['test_cases'], 1):
    question = test_case['question']
    ground_truth = test_case['ground_truth']
    
    print(f"Question {i}/{len(test_data['test_cases'])}: {question}")
    
    # Interroger le RAG
    result = rag.ask(question)
    
    # Extraire la réponse et les sources
    answer = result['answer']
    retrieved_docs = result['sources']
    
    # Construire la liste des contextes (texte des documents récupérés)
    context_list = retrieved_docs
    
    # Ajouter aux listes
    questions.append(question)
    answers.append(answer)
    contexts.append(context_list)
    ground_truths.append(ground_truth)
    
    print(f"✅ Réponse générée\n")

# Créer le dataset au format Ragas
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
print("⏳ Cela peut prendre 2-3 minutes (appels LLM pour calculer les métriques)...\n")

os.environ["OPENAI_API_KEY"] = "dummy" # Bloquer OpenAI

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

# Sauvegarder les résultats en JSON
output_path = "data/evaluation/ragas_results.json"
results_dict = {
    "faithfulness": float(results.to_pandas()['faithfulness'].mean()),
    "answer_relevancy": float(results.to_pandas()['answer_relevancy'].mean()),
    "context_precision": float(results.to_pandas()['context_precision'].mean()),
    "context_recall": float(results.to_pandas()['context_recall'].mean()),
    "num_questions": len(test_data['test_cases'])
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés dans : {output_path}")
print("\n✅ Évaluation MVP7 terminée avec succès !")