import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agente_edu.chains.qa_chain import DebugInfo, responder, responder_com_detalhes
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()

STATIC_DIR = Path("static").resolve()

app = FastAPI(title="Agente Educacional")


class Pergunta(BaseModel):
    mensagem: str
    persona: str = settings.default_persona


class Resposta(BaseModel):
    resposta: str


class RespostaDebug(BaseModel):
    pergunta: str
    contexto: str
    persona: str
    prompt_montado: str
    resposta_bruta: str
    guardrail_aplicado: bool
    resposta_final: str
    passos: list[str]


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/chat")
def chat(pergunta: Pergunta):
    contexto = retriever.buscar(pergunta.mensagem, k=settings.top_k)
    resposta = responder(pergunta.mensagem, contexto, persona=pergunta.persona)
    return Resposta(resposta=resposta)


@app.post("/chat/debug")
def chat_debug(pergunta: Pergunta):
    contexto = retriever.buscar(pergunta.mensagem, k=settings.top_k)
    try:
        info = responder_com_detalhes(pergunta.mensagem, contexto, persona=pergunta.persona)
        return RespostaDebug(**info.__dict__)
    except RuntimeError as e:
        return RespostaDebug(
            pergunta=pergunta.mensagem,
            contexto=contexto,
            persona=pergunta.persona,
            prompt_montado="(erro na geração do prompt)",
            resposta_bruta=str(e),
            guardrail_aplicado=True,
            resposta_final="Desculpe, a IA está temporariamente indisponível. Tente novamente mais tarde.",
            passos=[
                "RAG: documentos encontrados na base vetorial",
                f"LLM: erro ao chamar API - {e}",
                "Guardrail: resposta substituída devido a erro no LLM",
            ],
        )


@app.get("/health")
def health():
    return {"status": "ok"}


def rodar():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.web_port)
