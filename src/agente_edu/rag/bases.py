"""
Registro de bases de conhecimento disponíveis.

Para adicionar uma nova base: inclua uma entrada no dict BASES.
O sistema detecta automaticamente se o arquivo-fonte e o índice existem.
"""
from dataclasses import dataclass
from pathlib import Path

from agente_edu.config import BASE_DIR


@dataclass
class BaseConhecimento:
    id: str
    nome: str
    descricao: str
    icone: str
    tipo: str       # "csv" | "pdf" | "codigo"
    caminho: str    # relativo a BASE_DIR
    index_dir: str  # onde salvar o índice FAISS (relativo a BASE_DIR)


BASES: dict[str, BaseConhecimento] = {
    "pib": BaseConhecimento(
        id="pib",
        nome="Dados Econômicos",
        descricao="PIB, IDH e dados socioeconômicos de todos os estados brasileiros (2022-2023)",
        icone="📊",
        tipo="csv",
        caminho="data/pib_piaui.csv",
        index_dir=".index/pib",
    ),
    "codigo": BaseConhecimento(
        id="codigo",
        nome="Código deste Projeto",
        descricao="Código-fonte + documentação do agente educacional — LangChain, LangGraph, RAG, guardrails e o guia de adaptação",
        icone="💻",
        tipo="codigo",
        caminho="src/agente_edu",
        index_dir=".index/codigo",
    ),
    "livro": BaseConhecimento(
        id="livro",
        nome="História — 1º Ano EM",
        descricao="Moderna Plus História — 1º ano do Ensino Médio (PNLD 2026, ~500 páginas)",
        icone="📚",
        tipo="pdf",
        caminho="data/livro.pdf",
        index_dir=".index/livro",
    ),
}


def caminho_absoluto(base: BaseConhecimento) -> Path:
    return BASE_DIR / base.caminho


def index_absoluto(base: BaseConhecimento) -> Path:
    return BASE_DIR / base.index_dir


def arquivo_disponivel(base: BaseConhecimento) -> bool:
    """True se o arquivo-fonte (CSV, PDF ou pasta de código) existe."""
    return caminho_absoluto(base).exists()


def base_indexada(base: BaseConhecimento) -> bool:
    """True se o índice FAISS já foi construído em disco."""
    idx = index_absoluto(base)
    return idx.exists() and any(idx.iterdir())
