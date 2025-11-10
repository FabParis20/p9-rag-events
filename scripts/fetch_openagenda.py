"""
Script de récupération des événements OpenAgenda via Opendatasoft API
Sauvegarde dans data/processed/events_real.json au format compatible avec events_dummy.json
"""

import requests
import json
from pathlib import Path
from datetime import datetime

# ✅ URL CORRIGÉE de l'API Opendatasoft pour OpenAgenda
API_URL = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda@public/records"

# Paramètres de la requête
PARAMS = {
    "where": 'location_city:"Paris" AND firstdate_begin >= "2024-11-09T00:00:00"',
    "limit": 100 #"J'ai limité à 100 événements car c'est la limite maximale par requête de l'API Opendatasoft. En production, on utiliserait la pagination pour récupérer plus d'événements par lots successifs."
}


def transform_event(api_event):
    """
    Transforme un événement de l'API au format enrichi MVP5
    Combine description + longdescription + conditions
    """
    # Récupérer les différents champs textuels
    description = api_event.get("description_fr", "")
    long_desc = api_event.get("longdescription_fr", "")
    conditions = api_event.get("conditions_fr", "")
    
    # Combiner intelligemment les textes
    full_text = description if description else ""
    
    if long_desc and long_desc != description:
        if full_text:
            full_text += "\n\n" + long_desc
        else:
            full_text = long_desc
    
    if conditions:
        full_text += "\n\nConditions: " + conditions
    
    # Si aucun texte, message par défaut
    if not full_text.strip():
        full_text = "Pas de description disponible"
    
    return {
        "uid": api_event.get("uid", ""),
        "title_fr": api_event.get("title_fr", "Sans titre"),
        "description_fr": full_text,  # ⚠️ Texte enrichi ici
        "location_city": api_event.get("location_city", "Paris"),
        "location_name": api_event.get("location_name", ""),
        "location_address": api_event.get("location_address", ""),
        "location_coordinates": api_event.get("location_coordinates", {"lon": 0, "lat": 0}),
        "firstdate_begin": api_event.get("firstdate_begin", ""),
        "lastdate_begin": api_event.get("lastdate_begin", ""),
        "image": api_event.get("image", ""),
        "keywords_fr": api_event.get("keywords_fr", "")
    }


def main():
    print("🔍 Récupération des événements OpenAgenda...")
    
    try:
        # Appel API
        response = requests.get(API_URL, params=PARAMS)
        response.raise_for_status()
        
        data = response.json()
        
        # Transformer les événements
        events_transformed = []
        for event in data.get("results", []):
            events_transformed.append(transform_event(event))
        
        # Créer la structure finale (identique à events_dummy.json)
        output_data = {
            "total_count": len(events_transformed),
            "results": events_transformed
        }
        
        # Sauvegarder
        output_path = Path(__file__).parent.parent / "data" / "processed" / "events_real.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {len(events_transformed)} événements sauvegardés dans {output_path}")
        print(f"📅 Date de récupération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur lors de l'appel API : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")


if __name__ == "__main__":
    main()