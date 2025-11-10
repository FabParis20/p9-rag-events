"""
Réindexation avec chunking pour MVP5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.langchain_rag import create_vectorstore

def main():
    print("🚀 Début de la réindexation avec chunking (MVP5)")
    print("⏱️ Temps estimé : 15-30 minutes à cause du rate limit Voyage AI\n")
    
    create_vectorstore(source="real")
    
    print("\n🎉 Réindexation terminée !")

if __name__ == "__main__":
    main()