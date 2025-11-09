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
    "where": 'location_city:"Paris"',  # Filtre sur Paris
    "limit": 50,                        # Récupérer 50 événements
    "select": "uid,title_fr,description_fr,longdescription_fr,location_city,location_name,location_address,location_coordinates,firstdate_begin,lastdate_begin,image,keywords_fr"
}


def transform_event(api_event):
    """
    Transforme un événement de l'API au format events_dummy.json
    Gère les champs null
    """
    return {
        "uid": api_event.get("uid", ""),
        "title_fr": api_event.get("title_fr", "Sans titre"),
        "description_fr": api_event.get("description_fr") or api_event.get("longdescription_fr") or "Pas de description disponible",
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