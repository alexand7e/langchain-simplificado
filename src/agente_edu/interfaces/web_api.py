import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agente_edu.chains.qa_chain import DebugInfo, responder, responder_com_detalhes
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()

STATIC_DIR = Path("static").resolve()

app = FastAPI(title="Agente Educacional")


# ── Modelos de request/response ───────────────────────────────────────────────

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


class IngestRequest(BaseModel):
    texto: str
    titulo: str = ""


class IngestResponse(BaseModel):
    ok: bool
    mensagem: str
    caracteres: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/chat")
def chat(pergunta: Pergunta):
    contexto = retriever.buscar(pergunta.mensagem, k=settings.top_k)
    resposta_txt = responder(pergunta.mensagem, contexto, persona=pergunta.persona)
    return Resposta(resposta=resposta_txt)


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
                f"LLM: erro ao chamar API — {e}",
                "Guardrail: resposta substituída devido a erro no LLM",
            ],
        )


@app.post("/ingest")
def ingest(req: IngestRequest):
    """Adiciona texto ao índice vetorial em tempo real (sem reiniciar o servidor)."""
    if not req.texto.strip():
        raise HTTPException(status_code=400, detail="Texto não pode ser vazio.")
    try:
        retriever.adicionar(req.texto, titulo=req.titulo)
        logger.info(f"Ingestão manual: {len(req.texto)} chars, título='{req.titulo}'")
        return IngestResponse(
            ok=True,
            mensagem=f"✅ Conteúdo adicionado! O agente já pode responder sobre '{req.titulo or 'este tema'}'.",
            caracteres=len(req.texto),
        )
    except Exception as e:
        logger.error(f"Erro na ingestão: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph")
def get_graph():
    """Retorna a estrutura do grafo LangGraph para visualização no frontend."""
    return {
        "nodes": [
            {"id": "recuperar",  "label": "recuperar",  "tipo": "no",    "descricao": "Busca inicial no índice vetorial (RAG)"},
            {"id": "decidir",    "label": "decidir()",  "tipo": "router", "descricao": "Decide: há contexto?"},
            {"id": "buscar_mais","label": "buscar_mais","tipo": "no",    "descricao": "Segunda busca com k dobrado"},
            {"id": "responder",  "label": "responder",  "tipo": "no",    "descricao": "Gera a resposta com o LLM"},
            {"id": "END",        "label": "FIM",        "tipo": "fim",   "descricao": ""},
        ],
        "edges": [
            {"from": "recuperar",  "to": "decidir",    "label": ""},
            {"from": "decidir",    "to": "responder",  "label": "contexto encontrado"},
            {"from": "decidir",    "to": "buscar_mais","label": "contexto vazio (1ª vez)"},
            {"from": "buscar_mais","to": "responder",  "label": ""},
            {"from": "responder",  "to": "END",        "label": ""},
        ],
        "mermaid": (
            "flowchart TD\n"
            "  recuperar([🔍 recuperar]) --> decidir{decidir}\n"
            "  decidir -->|contexto encontrado| responder([🧠 responder])\n"
            "  decidir -->|contexto vazio| buscar_mais([🔎 buscar_mais])\n"
            "  buscar_mais --> responder\n"
            "  responder --> FIM([🏁 FIM])\n"
            "  style recuperar  fill:#e8f0fe,stroke:#1a73e8,color:#1a1a1a\n"
            "  style buscar_mais fill:#e8f0fe,stroke:#1a73e8,color:#1a1a1a\n"
            "  style responder  fill:#e6fffa,stroke:#38a169,color:#1a1a1a\n"
            "  style decidir    fill:#fef9e7,stroke:#b7791f,color:#1a1a1a\n"
            "  style FIM        fill:#f0f0f0,stroke:#aaa,color:#555\n"
        ),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def rodar():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.web_port)
