"""
Test du chunking
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data_loader import chunk_event_text

def main():
    # Charger les événements
    events_path = Path(__file__).parent.parent / "data" / "processed" / "events_real.json"
    
    with open(events_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prendre le premier événement
    event = data["results"][10]
    
    print("="*60)
    print(f"📝 Événement : {event['title_fr']}")
    print("="*60)
    
    # Chunker
    chunks = chunk_event_text(event)
    
    print(f"\n✅ {len(chunks)} chunk(s) créé(s)\n")
    
    # Afficher chaque chunk
    for i, chunk_doc in enumerate(chunks):
        print(f"--- CHUNK {i+1} ---")
        print(f"Taille : {len(chunk_doc['text'])} caractères")
        print(f"Texte : {chunk_doc['text'][:200]}...")
        print(f"Métadonnées : {chunk_doc['metadata']}")
        print()

if __name__ == "__main__":
    main()