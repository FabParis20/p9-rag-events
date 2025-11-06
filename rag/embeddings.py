"""
Module de génération d'embeddings et indexation Faiss
MVP1 : 5 événements dummy
Utilise Voyage AI pour les embeddings (partenaire Anthropic)
"""

import os
import numpy as np
import faiss
from pathlib import Path
import voyageai
from dotenv import load_dotenv

from data_loader import load_events, format_event_for_rag

# Charger les variables d'environnement depuis .env
load_dotenv()


def create_embeddings(events):
    """
    Génère les embeddings pour une liste d'événements.
    
    Args:
        events (list): Liste des événements
        
    Returns:
        tuple: (embeddings_array, texts_list)
    """
    # Initialiser le client Voyage AI
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("❌ VOYAGE_API_KEY non trouvée dans les variables d'environnement")
    
    vo = voyageai.Client(api_key=api_key)
    
    # Formater les événements en texte
    texts = [format_event_for_rag(event) for event in events]
    
    print(f"📝 Génération des embeddings pour {len(texts)} événements...")
    
    # Générer les embeddings avec Voyage AI
    result = vo.embed(
        texts=texts,
        model="voyage-3-lite",  # Modèle léger et rapide
        input_type="document"
    )
    
    # Extraire les embeddings
    embeddings_array = np.array(result.embeddings, dtype=np.float32)
    
    print(f"✅ Embeddings générés : shape {embeddings_array.shape}")
    
    return embeddings_array, texts


def create_faiss_index(embeddings):
    """
    Crée un index Faiss à partir des embeddings.
    
    Args:
        embeddings (np.array): Array des embeddings
        
    Returns:
        faiss.Index: Index Faiss
    """
    dimension = embeddings.shape[1]  # Dimension des embeddings Mistral = 1024
    
    print(f"🔧 Création index Faiss (dimension={dimension})...")
    
    # Créer un index simple (Flat L2)
    index = faiss.IndexFlatL2(dimension)
    
    # Ajouter les embeddings
    index.add(embeddings)
    
    print(f"✅ Index créé avec {index.ntotal} vecteurs")
    
    return index


def save_index(index, texts, save_dir="data/processed"):
    """
    Sauvegarde l'index Faiss et les textes.
    
    Args:
        index: Index Faiss
        texts (list): Liste des textes correspondants
        save_dir (str): Dossier de sauvegarde
    """
    save_path = Path(__file__).parent.parent / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder l'index Faiss
    index_file = save_path / "faiss_index.bin"
    faiss.write_index(index, str(index_file))
    print(f"💾 Index sauvegardé : {index_file}")
    
    # Sauvegarder les textes (pour récupérer les événements lors du retrieval)
    import json
    texts_file = save_path / "indexed_texts.json"
    with open(texts_file, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    print(f"💾 Textes sauvegardés : {texts_file}")


def build_rag_index(source="dummy"):
    """
    Pipeline complet : charger → embedder → indexer → sauvegarder.
    
    Args:
        source (str): "dummy" ou "real"
    """
    print("🚀 Démarrage du pipeline RAG\n")
    
    # 1. Charger les événements
    events = load_events(source=source)
    print(f"📥 {len(events)} événements chargés\n")
    
    # 2. Générer les embeddings
    embeddings, texts = create_embeddings(events)
    print()
    
    # 3. Créer l'index Faiss
    index = create_faiss_index(embeddings)
    print()
    
    # 4. Sauvegarder
    save_index(index, texts)
    print()
    
    print("✅ Pipeline RAG terminé avec succès !")
    
    return index, texts


if __name__ == "__main__":
    # Test du module
    build_rag_index(source="dummy")
