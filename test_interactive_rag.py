"""
Script interactif pour tester le système RAG avec historique de conversation.
Lance le script et pose tes questions avec des pauses naturelles !
"""

from rag.langchain_rag import PulsEventsRAG

def main():
    print("=" * 70)
    print("🎭 PULS-EVENTS RAG - MODE INTERACTIF")
    print("=" * 70)
    print()
    
    # Initialisation
    print("🚀 Initialisation du système...")
    rag = PulsEventsRAG()
    
    print()
    print("=" * 70)
    print("✅ Système prêt ! Tu peux maintenant poser tes questions.")
    print("=" * 70)
    print()
    print("💡 CONSEILS :")
    print("   - Pose des questions de suivi pour tester l'historique")
    print("   - Attends ~20 secondes entre les questions (rate limit Voyage AI)")
    print("   - Tape 'quit' ou 'exit' pour quitter")
    print("   - Tape 'clear' pour effacer l'historique")
    print()
    print("=" * 70)
    print()
    
    question_count = 0
    
    while True:
        # Demande de question
        user_input = input("❓ Ta question : ").strip()
        
        # Commandes spéciales
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 À bientôt !")
            break
        
        if user_input.lower() == 'clear':
            rag.clear_history()
            question_count = 0
            print()
            continue
        
        if not user_input:
            print("⚠️  Question vide, réessaye !")
            continue
        
        # Traitement de la question
        try:
            question_count += 1
            print()
            print(f"🔍 Recherche en cours... (Question #{question_count})")
            
            result = rag.ask(user_input)
            
            print()
            print("=" * 70)
            print("💬 RÉPONSE")
            print("=" * 70)
            print(result["answer"])
            print("=" * 70)
            print()
            
            # Avertissement pour le rate limit
            if question_count < 3:
                print("⏳ Attends ~20 secondes avant la prochaine question (rate limit)")
                print()
        
        except Exception as e:
            print()
            print("=" * 70)
            print("❌ ERREUR")
            print("=" * 70)
            print(f"Type : {type(e).__name__}")
            print(f"Message : {str(e)}")
            print()
            
            if "RateLimitError" in str(type(e)):
                print("💡 TIP : Attends 60 secondes et réessaye")
            
            print("=" * 70)
            print()

if __name__ == "__main__":
    main()
