from fastapi import FastAPI
from pydantic import BaseModel
from rag.langchain_rag import PulsEventsRAG

# Créer l'application FastAPI
app = FastAPI(title="Puls-Events RAG API")

# Initialiser le RAG au démarrage
rag = PulsEventsRAG()

# Modèle de données pour la requête
class Question(BaseModel):
    question: str

# Endpoint principal
@app.post("/ask")
def ask_question(q: Question):
    """Endpoint pour poser une question au RAG"""
    result = rag.ask(q.question)
    return result

# Endpoint de santé
@app.get("/health")
def health_check():
    """Vérifier que l'API fonctionne"""
    return {"status": "ok"}