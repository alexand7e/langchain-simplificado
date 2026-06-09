"""
RAG — Retrieval-Augmented Generation: etapa de ingestão.

Este módulo transforma arquivos de dados em um índice vetorial que o
agente pode consultar. O processo tem três etapas:

  ETAPA 1 — CARREGAR: lê o arquivo e converte em objetos Document.
  ETAPA 2 — INDEXAR:  gera embeddings e armazena em um banco vetorial.
  ETAPA 3 — PREPARAR: orquestra as etapas acima, com cache em disco.

Por que embeddings?
  Embeddings são representações numéricas de texto. Textos com significado
  parecido ficam "próximos" nesse espaço numérico, permitindo busca
  semântica (não apenas por palavra-chave).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.schema.document import Document
from langchain_core.embeddings import Embeddings

from agente_edu.config import INDEX_DIR, settings

logger = logging.getLogger(__name__)


# ── Embeddings ────────────────────────────────────────────────────────────────
# Em produção, usamos um modelo real (HuggingFace).
# Para testes sem GPU/internet, usamos um mock determinístico.

class _EmbeddingsFallback(Embeddings):
    """Embeddings fictícios para rodar testes sem acesso à API."""

    def __init__(self):
        logger.warning("Usando embeddings mock (apenas para testes unitários).")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._mock_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._mock_vector(text)

    @staticmethod
    def _mock_vector(text: str) -> list[float]:
        # Semente determinística baseada no texto → mesma entrada, mesmo vetor
        rng = np.random.default_rng(sum(ord(c) for c in text))
        return rng.uniform(-0.1, 0.1, size=384).tolist()


def _criar_embeddings() -> Embeddings:
    """
    Usa OpenAIEmbeddings apontando para a Mandu (mesma base URL do LLM).
    Qualquer API compatível com o padrão OpenAI funciona aqui.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embeddings_model,
            base_url=settings.soberano_api_base_url,
            api_key=settings.soberano_api_key,
        )
    except ImportError:
        logger.warning("langchain-openai não instalado. Usando embeddings mock.")
        return _EmbeddingsFallback()


def _obter_faiss():
    from langchain_community.vectorstores import FAISS
    return FAISS


# ── ETAPA 1: Carregar ─────────────────────────────────────────────────────────
# Lê o arquivo de dados e converte cada linha em um Document do LangChain.
# Document tem dois campos: page_content (texto) e metadata (dict com dados originais).

def carregar(caminho: str) -> list[Document]:
    """Lê um CSV e converte cada linha em um Document."""
    caminho_resolvido = Path(caminho)
    if caminho_resolvido.suffix != ".csv":
        raise ValueError(f"Formato não suportado: {caminho_resolvido.suffix}")
    if not caminho_resolvido.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    if caminho_resolvido.suffix == ".csv":
        df = pd.read_csv(caminho_resolvido)
        documentos = []
        for _, row in df.iterrows():
            # page_content: texto legível que o LLM vai receber como contexto
            texto = ", ".join(f"{col}: {val}" for col, val in row.items())
            # metadata: dados originais para filtragem ou rastreabilidade
            documentos.append(Document(page_content=texto, metadata=row.to_dict()))
        return documentos
    raise ValueError(f"Formato não suportado: {caminho_resolvido.suffix}")


# ── ETAPA 2: Indexar ──────────────────────────────────────────────────────────
# Gera os embeddings e armazena em um banco vetorial (FAISS ou Chroma).
# FAISS: biblioteca do Meta, roda localmente, ótima para demos.
# Chroma: alternativa com persistência nativa, requer instalação extra.

def indexar(documentos: list[Document]) -> object:
    """Gera embeddings dos documentos e cria o banco vetorial."""
    embeddings = _criar_embeddings()
    if settings.vector_store == "chroma":
        try:
            from langchain_chroma import Chroma
            return Chroma.from_documents(documentos, embeddings, persist_directory=str(INDEX_DIR))
        except ImportError:
            logger.warning("langchain-chroma não instalado. Usando FAISS.")
    FAISS = _obter_faiss()
    return FAISS.from_documents(documentos, embeddings)


def _salvar_indice(vectorstore) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    FAISS = _obter_faiss()
    if isinstance(vectorstore, FAISS):
        vectorstore.save_local(str(INDEX_DIR))


def _carregar_indice():
    if not INDEX_DIR.exists():
        return None
    embeddings = _criar_embeddings()
    if settings.vector_store == "chroma":
        try:
            from langchain_chroma import Chroma
            return Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings)
        except ImportError:
            pass
    FAISS = _obter_faiss()
    try:
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


# ── ETAPA 3: Preparar (ponto de entrada) ─────────────────────────────────────
# Tenta carregar o índice do disco (cache). Se não existir, cria do zero.

def preparar():
    """Retorna o banco vetorial pronto para uso (cria se necessário)."""
    indice = _carregar_indice()
    if indice is not None:
        logger.info("Índice carregado do disco.")
        return indice
    logger.info("Índice não encontrado. Criando a partir dos dados...")
    documentos = carregar(settings.data_path)
    indice = indexar(documentos)
    _salvar_indice(indice)
    return indice


# ── CLI ───────────────────────────────────────────────────────────────────────
# Permite rodar este arquivo diretamente: python -m agente_edu.rag.ingest [--rebuild]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cria ou reconstrói o índice vetorial.")
    parser.add_argument("--rebuild", action="store_true", help="Apaga o índice e recria do zero")
    args = parser.parse_args()
    if args.rebuild:
        import shutil
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
            logger.info("Índice anterior removido.")
    preparar()
    print(f"Índice pronto em {INDEX_DIR}")
