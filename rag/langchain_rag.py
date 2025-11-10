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
        Charge le vector store Faiss existant (format LangChain MVP5)
        
        Returns:
            FAISS vectorstore
        """
        index_dir = self.data_dir / "faiss_index"
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index introuvable: {index_dir}")
        
        # Charger avec LangChain
        vectorstore = FAISS.load_local(
            str(index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vectorstore
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Crée le template de prompt pour Claude.
        
        Returns:
            ChatPromptTemplate configuré
        """
        template = """Tu es un assistant spécialisé dans les événements culturels à Paris.

    ⚠️ RÈGLE IMPORTANTE : Privilégie TOUJOURS les événements futurs (à venir). 
    Si tous les événements trouvés sont passés, précise-le clairement à l'utilisateur 
    en disant "Cet événement a déjà eu lieu le [date]".

    Contexte des événements trouvés :
    {context}

    Question de l'utilisateur : {question}

    Historique de conversation :
    {chat_history}

    Réponds de manière naturelle, chaleureuse et précise. Si tu ne trouves pas d'événement 
    correspondant, propose des alternatives ou demande des précisions.
    """
        
        return ChatPromptTemplate.from_template(template)
    
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

def create_vectorstore(source="real", save_dir="data/processed"):
    """
    Crée un vectorstore FAISS avec chunking
    MVP5 : Utilise chunk_event_text pour découper les événements
    """
    from pathlib import Path
    try:
        from data_loader import load_events, chunk_event_text
    except ImportError:
        from rag.data_loader import load_events, chunk_event_text
    
    print(f"📚 Chargement des événements (source={source})...")
    events = load_events(source=source)
    print(f"✅ {len(events)} événements chargés")
    
    # ⚠️ NOUVEAU MVP5 : Chunking de tous les événements
    print("✂️ Découpage en chunks...")
    all_chunks = []
    for event in events:
        chunks = chunk_event_text(event)
        all_chunks.extend(chunks)
    
    print(f"✅ {len(all_chunks)} chunks créés (moyenne: {len(all_chunks)/len(events):.1f} chunks/événement)")
    
    # Créer les documents LangChain
    from langchain_core.documents import Document
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in all_chunks
    ]
    
    # Créer embeddings
    print("🔢 Création des embeddings...")
    embeddings = VoyageEmbeddings()
    
    # Créer vectorstore
    print("💾 Création du vectorstore FAISS...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Sauvegarder
    save_path = Path(__file__).parent.parent / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path / "faiss_index"))
    
    print(f"✅ Vectorstore sauvegardé dans {save_path / 'faiss_index'}")
    
    return vectorstore


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
