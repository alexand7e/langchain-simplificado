"""
Testes do grafo LangGraph.

Aqui testamos a *estrutura* do grafo (nós, arestas, roteador) sem
chamar o LLM ou o banco vetorial de verdade. Isso é possível porque
cada função é testável isoladamente — um dos benefícios de manter
os nós como funções puras que recebem e devolvem dicts.
"""
import pytest

from agente_edu.graph.agent_graph import Estado, construir_grafo, decidir


# ── Testes do roteador ────────────────────────────────────────────────────────

def test_decidir_vai_para_responder_quando_ha_contexto():
    estado: Estado = {
        "pergunta": "Qual o PIB do Piauí?",
        "contexto": "Piauí, pib_2023_bilhoes: 63.4",
        "resposta": "",
        "tentativas": 0,
    }
    assert decidir(estado) == "responder"


def test_decidir_vai_para_buscar_mais_quando_contexto_vazio():
    estado: Estado = {
        "pergunta": "Qual o PIB do Piauí?",
        "contexto": "",
        "resposta": "",
        "tentativas": 0,
    }
    assert decidir(estado) == "buscar_mais"


def test_decidir_vai_para_responder_apos_segunda_tentativa():
    # Depois de tentativas=1, não tenta mais — evita loop infinito.
    estado: Estado = {
        "pergunta": "Qual o PIB do Piauí?",
        "contexto": "",
        "resposta": "",
        "tentativas": 1,
    }
    assert decidir(estado) == "responder"


def test_decidir_ignora_contexto_so_espacos():
    estado: Estado = {
        "pergunta": "Teste",
        "contexto": "   ",
        "resposta": "",
        "tentativas": 0,
    }
    assert decidir(estado) == "buscar_mais"


# ── Testes da estrutura do grafo ──────────────────────────────────────────────

def test_grafo_compila_sem_erros():
    """O grafo deve compilar sem levantar exceção."""
    grafo = construir_grafo()
    assert grafo is not None


def test_grafo_tem_nos_esperados():
    grafo = construir_grafo()
    nos = set(grafo.nodes)
    assert "recuperar" in nos
    assert "buscar_mais" in nos
    assert "responder" in nos
