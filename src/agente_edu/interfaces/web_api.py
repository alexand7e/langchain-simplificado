import logging

from fastapi import FastAPI
from pydantic import BaseModel

from agente_edu.chains.qa_chain import responder
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()

app = FastAPI(title="Agente Educacional")


class Pergunta(BaseModel):
    mensagem: str
    persona: str = settings.default_persona


class Resposta(BaseModel):
    resposta: str


@app.post("/chat")
def chat(pergunta: Pergunta):
    contexto = retriever.buscar(pergunta.mensagem, k=settings.top_k)
    resposta = responder(pergunta.mensagem, contexto, persona=pergunta.persona)
    return Resposta(resposta=resposta)


@app.get("/health")
def health():
    return {"status": "ok"}


def rodar():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.web_port)
