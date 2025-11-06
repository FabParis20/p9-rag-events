"""
Module de recherche (retrieval) dans l'index Faiss
MVP1 : Recherche dans les 5 événements dummy
Utilise Voyage AI pour les embeddings
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
import voyageai
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


def load_index_and_texts(data_dir="data/processed"):
    """
    Charge l'index Faiss et les textes indexés.
    
    Args:
        data_dir (str): Dossier contenant l'index
        
    Returns:
        tuple: (index_faiss, texts_list)
    """
    base_path = Path(__file__).parent.parent / data_dir
    
    # Charger l'index Faiss
    index_path = base_path / "faiss_index.bin"
    index = faiss.read_index(str(index_path))
    print(f"📂 Index Faiss chargé : {index.ntotal} vecteurs")
    
    # Charger les textes
    texts_path = base_path / "indexed_texts.json"
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)
    print(f"📄 {len(texts)} textes chargés")
    
    return index, texts


def generate_query_embedding(query):
    """
    Génère l'embedding d'une question avec Voyage AI.
    
    Args:
        query (str): Question de l'utilisateur
        
    Returns:
        np.array: Embedding de la question
    """
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ VOYAGE_API_KEY non trouvée")
    
    vo = voyageai.Client(api_key=api_key)
    
    # Générer l'embedding
    result = vo.embed(
        texts=[query],
        model="voyage-3-lite",
        input_type="query"
    )
    
    embedding = np.array([result.embeddings[0]], dtype=np.float32)
    return embedding


def search(query, k=3):
    """
    Recherche les k événements les plus pertinents pour une question.
    
    Args:
        query (str): Question de l'utilisateur
        k (int): Nombre de résultats à retourner
        
    Returns:
        list: Liste des k textes les plus pertinents
    """
    print(f"\n🔍 Recherche pour : '{query}'")
    print(f"📊 Top {k} résultats demandés\n")
    
    # Charger l'index et les textes
    index, texts = load_index_and_texts()
    
    # Générer l'embedding de la question
    print("🧮 Génération de l'embedding de la question...")
    query_embedding = generate_query_embedding(query)
    
    # Rechercher dans Faiss
    print("🔎 Recherche dans l'index Faiss...")
    distances, indices = index.search(query_embedding, k)
    
    # Récupérer les textes correspondants
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            "rank": i + 1,
            "distance": float(distance),
            "text": texts[idx]
        })
    
    return results


if __name__ == "__main__":
    # Test du retriever
    print("=" * 60)
    print("🧪 TEST DU RETRIEVER")
    print("=" * 60)
    
    # Question test
    test_query = "Je cherche un concert de jazz à Paris"
    
    # Rechercher
    results = search(test_query, k=3)
    
    # Afficher les résultats
    print("\n" + "=" * 60)
    print("📋 RÉSULTATS")
    print("=" * 60)
    
    for result in results:
        print(f"\n🏆 Rang {result['rank']} - Distance: {result['distance']:.4f}")
        print("-" * 60)
        print(result['text'])
        print("-" * 60)
