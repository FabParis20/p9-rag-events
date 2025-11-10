"""
Test du nettoyage HTML
"""
import json
from pathlib import Path
import sys

# Ajouter le chemin parent pour importer rag
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data_loader import clean_html

def main():
    # Charger un événement
    events_path = Path(__file__).parent.parent / "data" / "processed" / "events_real.json"
    
    with open(events_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prendre le premier événement avec du HTML
    event = data["results"][0]
    
    print("="*60)
    print("📝 AVANT nettoyage :")
    print("="*60)
    print(event["description_fr"][:500])  # 500 premiers caractères
    
    print("\n" + "="*60)
    print("✨ APRÈS nettoyage :")
    print("="*60)
    cleaned = clean_html(event["description_fr"])
    print(cleaned[:500])

if __name__ == "__main__":
    main()