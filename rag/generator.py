"""
Module de génération de réponses (generation)
MVP1 : Répond aux questions en s'appuyant sur les événements trouvés
Utilise Claude (Anthropic) pour la génération
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

from retriever import search

# Charger les variables d'environnement
load_dotenv()


def build_prompt(query, retrieved_events):
    """
    Construit le prompt pour Mistral avec la question et les événements.
    
    Args:
        query (str): Question de l'utilisateur
        retrieved_events (list): Liste des événements pertinents
        
    Returns:
        str: Prompt formaté
    """
    # Construire le contexte avec les événements
    context = "\n\n".join([
        f"Événement {event['rank']}:\n{event['text']}"
        for event in retrieved_events
    ])
    
    # Prompt système
    prompt = f"""Tu es un assistant spécialisé dans la recommandation d'événements culturels à Paris.

Voici les événements pertinents pour répondre à la question de l'utilisateur :

{context}

Question de l'utilisateur : {query}

Réponds de manière claire et concise en recommandant le ou les événements les plus adaptés. Mentionne le titre, le lieu et la date."""
    
    return prompt


def generate_response(query, k=3):
    """
    Génère une réponse complète à une question (RAG complet).
    
    Args:
        query (str): Question de l'utilisateur
        k (int): Nombre d'événements à récupérer
        
    Returns:
        dict: Réponse avec le texte généré et les sources
    """
    print(f"\n🤖 Génération de la réponse pour : '{query}'")
    
    # 1. Retrieval : Chercher les événements pertinents
    retrieved_events = search(query, k=k)
    
    # 2. Augmentation : Construire le prompt
    print("\n📝 Construction du prompt...")
    prompt = build_prompt(query, retrieved_events)
    
    # 3. Generation : Appeler Claude pour générer la réponse
    print("🧠 Génération de la réponse avec Claude...\n")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("❌ ANTHROPIC_API_KEY non trouvée")
    
    client = Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    answer = message.content[0].text
    
    return {
        "question": query,
        "answer": answer,
        "sources": retrieved_events
    }


if __name__ == "__main__":
    # Test du RAG complet
    print("=" * 70)
    print("🧪 TEST DU RAG COMPLET (Retrieval + Generation)")
    print("=" * 70)
    
    # Question test
    test_query = "Je cherche un concert de jazz à Paris"
    
    # Générer la réponse
    result = generate_response(test_query, k=3)
    
    # Afficher la réponse
    print("=" * 70)
    print("💬 RÉPONSE FINALE")
    print("=" * 70)
    print(f"\n❓ Question : {result['question']}\n")
    print(f"✅ Réponse :\n{result['answer']}\n")
    print("=" * 70)
    print("📚 SOURCES UTILISÉES")
    print("=" * 70)
    for source in result['sources']:
        print(f"\n🏆 Rang {source['rank']} (Distance: {source['distance']:.4f})")
        print(source['text'][:100] + "...")
