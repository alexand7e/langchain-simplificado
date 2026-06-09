"""
RAG — Retrieval-Augmented Generation: etapa de ingestão.

Transforma arquivos de dados em um índice vetorial consultável.
Suporta três tipos de fonte:

  "csv"    — planilha de dados (ex: PIB dos estados)
  "pdf"    — livro ou documento em PDF (1 Document por página)
  "codigo" — arquivos .py de um diretório (1 Document por arquivo)

Fluxo:
  ETAPA 1 — CARREGAR: lê a fonte e converte em objetos Document.
  ETAPA 2 — INDEXAR:  gera embeddings e armazena em banco vetorial.
  ETAPA 3 — PREPARAR: orquestra com cache em disco por base.

Por que embeddings?
  Representações numéricas de texto onde textos com significado parecido
  ficam "próximos" — permitindo busca semântica, não por palavra-chave.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.schema.document import Document
from langchain_core.embeddings import Embeddings

from agente_edu.config import BASE_DIR, settings

logger = logging.getLogger(__name__)


# ── Embeddings ────────────────────────────────────────────────────────────────

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
        rng = np.random.default_rng(sum(ord(c) for c in text))
        return rng.uniform(-0.1, 0.1, size=384).tolist()


def _criar_embeddings() -> Embeddings:
    """OpenAIEmbeddings apontando para a Mandu — padrão OpenAI, qualquer provider funciona."""
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

def carregar(caminho: str) -> list[Document]:
    """Carrega um CSV e converte cada linha em um Document."""
    caminho_resolvido = Path(caminho)
    if caminho_resolvido.suffix != ".csv":
        raise ValueError(f"Formato não suportado: {caminho_resolvido.suffix}")
    if not caminho_resolvido.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho_resolvido)
    documentos = []
    for _, row in df.iterrows():
        texto = ", ".join(f"{col}: {val}" for col, val in row.items())
        documentos.append(Document(page_content=texto, metadata=row.to_dict()))
    return documentos


def _carregar_pdf(caminho: Path) -> list[Document]:
    """1 Document por página do PDF."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError("pypdf não instalado. Execute: pip install pypdf")
    loader = PyPDFLoader(str(caminho))
    paginas = loader.load()
    logger.info(f"PDF carregado: {len(paginas)} páginas de '{caminho.name}'")
    return paginas


def _carregar_codigo(caminho: Path) -> list[Document]:
    """1 Document por arquivo .py encontrado recursivamente."""
    documentos = []
    for arquivo in sorted(caminho.rglob("*.py")):
        try:
            texto = arquivo.read_text(encoding="utf-8")
            # Caminho relativo ao projeto para facilitar referência
            relativo = arquivo.relative_to(BASE_DIR)
            documentos.append(Document(
                page_content=f"# arquivo: {relativo}\n\n{texto}",
                metadata={"arquivo": str(relativo), "tipo": "codigo"},
            ))
        except Exception as e:
            logger.warning(f"Erro ao ler {arquivo}: {e}")
    logger.info(f"Código carregado: {len(documentos)} arquivos .py")
    return documentos


def carregar_base(base) -> list[Document]:
    """Despacha para o loader correto com base no tipo da BaseConhecimento."""
    caminho = BASE_DIR / base.caminho
    if base.tipo == "csv":
        return carregar(str(caminho))
    if base.tipo == "pdf":
        return _carregar_pdf(caminho)
    if base.tipo == "codigo":
        return _carregar_codigo(caminho)
    raise ValueError(f"Tipo de base desconhecido: '{base.tipo}'")


# ── ETAPA 2: Indexar ──────────────────────────────────────────────────────────

def indexar(documentos: list[Document], index_dir: Path) -> object:
    """Gera embeddings e cria o banco vetorial FAISS."""
    embeddings = _criar_embeddings()
    FAISS = _obter_faiss()
    store = FAISS.from_documents(documentos, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    logger.info(f"Índice salvo em {index_dir} ({len(documentos)} documentos)")
    return store


def _carregar_indice(index_dir: Path):
    """Carrega índice FAISS do disco. Retorna None se não existir."""
    if not index_dir.exists() or not any(index_dir.iterdir()):
        return None
    embeddings = _criar_embeddings()
    FAISS = _obter_faiss()
    try:
        store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        logger.info(f"Índice carregado do disco: {index_dir}")
        return store
    except Exception as e:
        logger.warning(f"Erro ao carregar índice de {index_dir}: {e}")
        return None


# ── ETAPA 3: Preparar ─────────────────────────────────────────────────────────

def preparar(base=None):
    """
    Retorna o banco vetorial pronto para uso.
    Se `base` for None, usa a base padrão do settings (retrocompatibilidade).
    Carrega do disco se disponível; caso contrário, cria do zero e salva.
    """
    if base is None:
        # Retrocompatibilidade: usa config legada
        from agente_edu.config import INDEX_DIR
        from agente_edu.rag.bases import BASES
        base = BASES.get(settings.default_base_id, list(BASES.values())[0])
        index_dir = INDEX_DIR
    else:
        from agente_edu.rag.bases import index_absoluto
        index_dir = index_absoluto(base)

    indice = _carregar_indice(index_dir)
    if indice is not None:
        return indice

    logger.info(f"Criando índice para base '{base.id}'...")
    documentos = carregar_base(base)
    return indexar(documentos, index_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────
# python -m agente_edu.rag.ingest --base=pib [--rebuild]

if __name__ == "__main__":
    import shutil
    from agente_edu.rag.bases import BASES, index_absoluto

    parser = argparse.ArgumentParser(description="Pré-indexa uma base de conhecimento.")
    parser.add_argument("--base", choices=list(BASES.keys()), default="pib",
                        help="Base a indexar (default: pib)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Apaga o índice existente e recria do zero")
    args = parser.parse_args()

    base = BASES[args.base]
    idx  = index_absoluto(base)

    if args.rebuild and idx.exists():
        shutil.rmtree(idx)
        print(f"Índice anterior removido: {idx}")

    preparar(base)
    print(f"✅ Índice pronto: {idx}")
