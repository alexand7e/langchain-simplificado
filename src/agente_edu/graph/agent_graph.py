"""
Grafo do agente educacional — construído com LangGraph.

Fluxo:
    recuperar → decidir ──► responder → FIM
                        ↘
                         buscar_mais → responder → FIM

- Se a primeira busca encontrar contexto, vai direto para "responder".
- Se não encontrar, tenta uma segunda busca mais ampla ("buscar_mais")
  antes de responder (o guardrail ainda bloqueia se continuar vazio).
"""
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agente_edu.chains.qa_chain import responder
from agente_edu.config import settings
from agente_edu.rag.retriever import Retriever

logger = logging.getLogger(__name__)
retriever = Retriever()


# ── Estado ────────────────────────────────────────────────────────────────────
# TypedDict define os campos que trafegam entre os nós do grafo.
# Cada nó pode ler e devolver partes desse dicionário.

class Estado(TypedDict):
    pergunta: str    # entrada do aluno
    contexto: str    # trechos recuperados pelo RAG
    resposta: str    # resposta final do agente
    tentativas: int  # quantas vezes já buscamos


# ── Nós ───────────────────────────────────────────────────────────────────────
# Cada função recebe o estado atual e devolve um dict com os campos atualizados.

def recuperar(estado: Estado) -> dict:
    """Nó 1 — busca inicial no índice vetorial (RAG)."""
    logger.info(f"Recuperando contexto para: {estado['pergunta']}")
    contexto = retriever.buscar(estado["pergunta"], k=settings.top_k)
    return {"contexto": contexto, "tentativas": 0}


def buscar_mais(estado: Estado) -> dict:
    """Nó 2 (opcional) — segunda tentativa com busca mais ampla (k dobrado)."""
    logger.info("Contexto vazio — tentando busca mais ampla")
    contexto = retriever.buscar(estado["pergunta"], k=settings.top_k * 2)
    return {"contexto": contexto, "tentativas": 1}


def responder_no(estado: Estado) -> dict:
    """Nó 3 — gera a resposta com o LLM usando o contexto recuperado."""
    logger.info("Gerando resposta")
    resposta = responder(estado["pergunta"], estado["contexto"], persona=settings.default_persona)
    return {"resposta": resposta}


# ── Roteador ──────────────────────────────────────────────────────────────────
# Aresta condicional: decide qual nó vem depois de "recuperar".

def decidir(estado: Estado) -> Literal["responder", "buscar_mais"]:
    """Se não encontrou contexto na primeira busca, tenta uma vez mais."""
    contexto_vazio = not estado["contexto"] or estado["contexto"].strip() == ""
    if contexto_vazio and estado.get("tentativas", 0) < 1:
        return "buscar_mais"
    return "responder"


# ── Montagem do grafo ─────────────────────────────────────────────────────────

def construir_grafo() -> StateGraph:
    grafo = StateGraph(Estado)

    # Registrar os nós
    grafo.add_node("recuperar", recuperar)
    grafo.add_node("buscar_mais", buscar_mais)
    grafo.add_node("responder", responder_no)

    # Registrar as arestas
    grafo.add_conditional_edges("recuperar", decidir)  # roteador decide o próximo
    grafo.add_edge("buscar_mais", "responder")          # busca extra sempre vai para responder
    grafo.add_edge("responder", END)                    # resposta pronta → fim

    grafo.set_entry_point("recuperar")
    return grafo.compile()
