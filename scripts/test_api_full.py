import requests
import json

API_URL = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda@public/records"

# Récupérer un seul événement, tous le champs
PARAMS = {
    "where": 'location_city:"Paris"',
    "limit": 1
}

def main():
    print("🔍 Récupération d'un événement complet...\n")
    
    response = requests.get(API_URL, params=PARAMS)
    response.raise_for_status()
    data = response.json()
    
    if data.get("results"):
        event = data["results"][0]
        
        # Afficher les champs texte importants
        print("="*60)
        print("📝 TITLE:")
        print(event.get("title_fr", ""))
        
        print("\n" + "="*60)
        print("📝 DESCRIPTION COURTE:")
        print(event.get("description_fr", ""))
        
        print("\n" + "="*60)
        print("📝 DESCRIPTION LONGUE:")
        print(event.get("longdescription_fr", ""))
        
        print("\n" + "="*60)
        print("📝 CONDITIONS:")
        print(event.get("conditions_fr", ""))
        
    else:
        print("❌ Aucun événement trouvé")

if __name__ == "__main__":
    main()