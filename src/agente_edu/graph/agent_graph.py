import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agente_edu.chains.qa_chain import responder
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()


class Estado(TypedDict):
    pergunta: str
    contexto: str
    resposta: str
    tentativas: int


def recuperar(estado: Estado) -> dict:
    logger.info(f"Recuperando contexto para: {estado['pergunta']}")
    contexto = retriever.buscar(estado["pergunta"], k=settings.top_k)
    return {"contexto": contexto}


def responder_no(estado: Estado) -> dict:
    logger.info("Gerando resposta")
    resposta = responder(estado["pergunta"], estado["contexto"], persona=settings.default_persona)
    return {"resposta": resposta}


def decidir(estado: Estado) -> Literal["responder", "buscar_mais"]:
    if not estado["contexto"] or estado["contexto"].strip() == "":
        if estado.get("tentativas", 0) < 1:
            return "buscar_mais"
    return "responder"


def construir_grafo() -> StateGraph:
    grafo = StateGraph(Estado)
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("responder", responder_no)
    grafo.add_conditional_edges("recuperar", decidir)
    grafo.add_edge("responder", END)
    grafo.set_entry_point("recuperar")
    return grafo.compile()
