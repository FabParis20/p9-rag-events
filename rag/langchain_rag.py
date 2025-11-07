"""
Pipeline RAG avec LangChain pour Puls-Events
Utilise : Voyage AI (embeddings) + Faiss (vector store) + Claude (generation)
"""

import os
import json
from pathlib import Path
from typing import List

import numpy as np
import voyageai
from dotenv import load_dotenv

# Imports LangChain
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import des fonctions existantes
# Try/except pour supporter les deux contextes :
# - Exécution directe : python rag/langchain_rag.py
# - Import depuis racine : from rag.langchain_rag import
try:
    from data_loader import load_events, format_event_for_rag
except ImportError:
    from rag.data_loader import load_events, format_event_for_rag

load_dotenv()


# ============================================================
# WRAPPER VOYAGE AI POUR LANGCHAIN
# ============================================================

class VoyageEmbeddings(Embeddings):
    """
    Wrapper pour utiliser Voyage AI avec LangChain.
    
    Analogie : Un adaptateur qui permet à Voyage AI de "parler"
    le langage de LangChain 🔌
    """
    
    def __init__(self, model: str = "voyage-3-lite"):
        """
        Initialise le client Voyage AI.
        
        Args:
            model: Modèle d'embeddings (par défaut voyage-3-lite)
        """
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY non trouvée dans .env")
        
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de documents.
        
        Args:
            texts: Liste des textes à encoder
            
        Returns:
            Liste des vecteurs d'embeddings
        """
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type="document"  # Type pour documents (vs query)
        )
        return result.embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Génère l'embedding pour une question utilisateur.
        
        Args:
            text: Question à encoder
            
        Returns:
            Vecteur d'embedding
        """
        result = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="query"  # Type pour questions
        )
        return result.embeddings[0]


# ============================================================
# CLASSE RAG AVEC LANGCHAIN
# ============================================================

class PulsEventsRAG:
    """
    Système RAG complet pour recommander des événements culturels.
    
    Pipeline : Question → Retrieval → Augmentation → Generation
    """
    
    def __init__(self, data_dir: str = "data/processed", k: int = 3):
        """
        Initialise le système RAG.
        
        Args:
            data_dir: Dossier contenant l'index Faiss
            k: Nombre d'événements à récupérer
        """
        print("🚀 Initialisation du système RAG LangChain...")
        
        self.k = k
        self.data_dir = Path(__file__).parent.parent / data_dir
        
        # 1. Chargement des embeddings Voyage AI
        print("📦 Chargement du modèle d'embeddings Voyage AI...")
        self.embeddings = VoyageEmbeddings()
        
        # 2. Chargement du vector store Faiss
        print(f"🗄️ Chargement du vector store depuis {self.data_dir}...")
        self.vectorstore = self._load_vectorstore()
        
        # 3. Configuration du retriever
        print(f"🔍 Configuration du retriever (top-{k})...")
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # 4. Configuration de Claude
        print("🤖 Configuration de Claude Sonnet 4.5...")
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0.7,
            max_tokens=1024
        )
        
        # 5. Création du prompt template
        self.prompt = self._create_prompt_template()
        
        # 6. Construction de la chaîne RAG
        self.chain = self._build_chain()
        
        # 7. Initialisation de l'historique
        self.history = []
        
        print("✅ Système RAG prêt !\n")
    
    def _load_vectorstore(self) -> FAISS:
        """
        Charge le vector store Faiss existant.
        
        Returns:
            FAISS vectorstore
        """
        index_path = str(self.data_dir / "faiss_index.bin")
        texts_path = self.data_dir / "indexed_texts.json"
        
        # Vérification des fichiers
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index Faiss non trouvé : {index_path}")
        if not texts_path.exists():
            raise FileNotFoundError(f"Textes non trouvés : {texts_path}")
        
        # Chargement des textes
        with open(texts_path) as f:
            texts = json.load(f)
        
        print(f"   📄 {len(texts)} événements chargés")
        
        # Chargement du vector store avec les embeddings
        vectorstore = FAISS.load_local(
            str(self.data_dir),
            self.embeddings,
            "faiss_index",
            allow_dangerous_deserialization=True  # Nécessaire pour Faiss
        )
        
        return vectorstore
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Crée le template de prompt pour Claude.
        
        Returns:
            ChatPromptTemplate configuré
        """
        template = """Tu es un assistant spécialisé dans la recommandation d'événements culturels à Paris.

Voici les événements pertinents pour répondre à la question de l'utilisateur :

{context}

Réponds de manière claire, engageante et concise en recommandant le ou les événements les plus adaptés. 
Mentionne toujours le titre, le lieu et la date de chaque événement recommandé.

Si l'utilisateur fait référence à une conversation précédente, prends en compte l'historique."""

        return ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    
    def _format_docs(self, docs) -> str:
        """
        Formate les documents retrievés pour le prompt.
        
        Args:
            docs: Documents LangChain
            
        Returns:
            String formaté avec numérotation
        """
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"Événement {i}:\n{doc.page_content}\n")
        return "\n".join(formatted)
    
    def _build_chain(self):
        """
        Construit la chaîne RAG LangChain.
        
        Pipeline : retriever → format_docs → prompt → llm → parser
        
        Returns:
            Chaîne RAG exécutable
        """
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.history
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask(self, question: str, use_history: bool = True) -> dict:
        """
        Pose une question au système RAG.
        
        Args:
            question: Question de l'utilisateur
            use_history: Utiliser l'historique de conversation
            
        Returns:
            dict avec question, réponse et sources
        """
        print(f"❓ Question : {question}")
        print(f"🔍 Recherche des événements pertinents...")
        
        # Récupération des documents pertinents pour les sources
        docs = self.retriever.invoke(question)
        
        # Génération de la réponse
        print(f"🤖 Génération de la réponse avec Claude...\n")
        answer = self.chain.invoke(question)
        
        # Mise à jour de l'historique si activé
        if use_history:
            self.history.append(HumanMessage(content=question))
            self.history.append(AIMessage(content=answer))
        
        return {
            "question": question,
            "answer": answer,
            "sources": [doc.page_content for doc in docs]
        }
    
    def clear_history(self):
        """Efface l'historique de conversation."""
        self.history = []
        print("🗑️ Historique effacé")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def create_vectorstore(source: str = "dummy", save_dir: str = "data/processed"):
    """
    Crée un nouveau vector store à partir des événements.
    
    Fonction utilitaire pour construire l'index initial.
    
    Args:
        source: "dummy" ou "real"
        save_dir: Dossier de sauvegarde
    """
    print("🏗️ Création du vector store...")
    
    # Chargement des événements
    events = load_events(source=source)
    texts = [format_event_for_rag(event) for event in events]
    print(f"📥 {len(texts)} événements chargés")
    
    # Création des embeddings
    embeddings = VoyageEmbeddings()
    
    # Création du vector store
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings
    )
    
    # Sauvegarde
    save_path = Path(__file__).parent.parent / save_dir
    vectorstore.save_local(str(save_path), "faiss_index")
    
    # Sauvegarde des textes pour référence
    texts_path = save_path / "indexed_texts.json"
    with open(texts_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vector store sauvegardé : {save_path}")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 TEST DU SYSTÈME RAG LANGCHAIN")
    print("=" * 70 + "\n")
    
    # Initialisation
    rag = PulsEventsRAG()
    
    # Test 1 : Question simple
    print("\n" + "=" * 70)
    print("TEST 1 : Question simple")
    print("=" * 70 + "\n")
    
    result = rag.ask("Je cherche un concert de jazz à Paris")
    
    print("=" * 70)
    print("💬 RÉPONSE")
    print("=" * 70)
    print(result["answer"])
    
    # Test 2 : Question de suivi (avec historique)
    print("\n" + "=" * 70)
    print("TEST 2 : Question de suivi (historique activé)")
    print("=" * 70 + "\n")
    
    # Délai pour respecter le rate limit Voyage AI (3 RPM sur plan gratuit)
    print("⏳ Attente de 20 secondes pour respecter le rate limit Voyage AI...")
    import time
    time.sleep(20)
    print("✅ Reprise du test\n")
    
    result2 = rag.ask("Et pour un spectacle de danse ?")
    
    print("=" * 70)
    print("💬 RÉPONSE")
    print("=" * 70)
    print(result2["answer"])
    
    print("\n" + "=" * 70)
    print("✅ Tests terminés !")
    print("=" * 70)
