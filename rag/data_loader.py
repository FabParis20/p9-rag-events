"""
Module de chargement des données événementielles
MVP1 : Charge les données depuis events_dummy.json
MVP4 : Chargera depuis events_real.json (même code)
"""

import json
from pathlib import Path
import re

def clean_html(text):
    """
    Nettoie le HTML d'un texte
    Enlève les balises et garde le texte
    """
    if not text:
        return ""
    
    # Enlever les balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    
    # Nettoyer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    # Nettoyer les sauts de ligne multiples
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_event_text(event):
    """
    Découpe un événement en chunks avec métadonnées
    Retourne une liste de dicts {text, metadata}
    """
    # Nettoyer le HTML
    clean_text = clean_html(event.get("description_fr", ""))
    
    # Ajouter le titre au début
    full_text = f"{event.get('title_fr', '')}\n\n{clean_text}"
    
    # Créer le splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # Taille d'un chunk
        chunk_overlap=100,   # Recouvrement entre chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Découper
    chunks = text_splitter.split_text(full_text)
    
    # Ajouter les métadonnées à chaque chunk
    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_docs.append({
            "text": chunk,
            "metadata": {
                "uid": event.get("uid", ""),
                "title_fr": event.get("title_fr", ""),
                "location_name": event.get("location_name", ""),
                "firstdate_begin": event.get("firstdate_begin", ""),
                "chunk_index": i
            }
        })
    
    return chunk_docs

# def load_events(source="dummy"):
def load_events(source="real"):
    """
    Charge les événements depuis le fichier JSON.
    
    Args:
        source (str): "dummy" pour MVP1, "real" pour MVP4
        
    Returns:
        list: Liste des événements (results)
    """
    # Chemin du fichier selon la source
    if source == "dummy":
        file_path = Path(__file__).parent.parent / "data" / "processed" / "events_dummy.json"
    else:
        file_path = Path(__file__).parent.parent / "data" / "processed" / "events_real.json"
    
    # Charger le JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Retourner la liste des événements
    return data['results']


def format_event_for_rag(event):
    """
    Formate un événement pour le RAG (texte à embedder).
    
    Args:
        event (dict): Un événement du JSON
        
    Returns:
        str: Texte formaté pour embeddings
    """
    # Construire le texte avec les infos clés
    text = f"""Titre: {event['title_fr']}
Description: {event['description_fr']}
Lieu: {event['location_name']}, {event['location_address']}
Date: {event['firstdate_begin']}
Mots-clés: {event.get('keywords_fr', 'Aucun')}"""
    
    return text


if __name__ == "__main__":
    # Test du module
    print("🔍 Test du chargement des événements...\n")
    
    events = load_events(source="dummy")
    print(f"✅ {len(events)} événements chargés\n")
    
    # Afficher le premier événement formaté
    print("📄 Premier événement formaté pour le RAG:")
    print("-" * 50)
    print(format_event_for_rag(events[0]))
