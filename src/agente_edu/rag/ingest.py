import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from langchain.schema.document import Document
from langchain_core.embeddings import Embeddings

from agente_edu.config import INDEX_DIR, settings

logger = logging.getLogger(__name__)


class _EmbeddingsFallback(Embeddings):
    def __init__(self):
        logger.warning("Usando embeddings mock (apenas para teste). Instale '.[full]' para embeddings reais.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._mock_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._mock_vector(text)

    @staticmethod
    def _mock_vector(text: str) -> list[float]:
        rng = np.random.default_rng(sum(ord(c) for c in text))
        return rng.uniform(-0.1, 0.1, size=384).tolist()


def _criar_embeddings() -> Embeddings:
    try:
        import sentence_transformers  # noqa: F401
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.embeddings_model)
    except ImportError:
        logger.warning("Pacote opcional não instalado. Usando embeddings mock (apenas para teste).")
        logger.warning("Instale '.[full]' para embeddings reais.")
        return _EmbeddingsFallback()


def _obter_faiss():
    from langchain_community.vectorstores import FAISS
    return FAISS


def carregar(caminho: str) -> list[Document]:
    caminho_resolvido = Path(caminho)
    if not caminho_resolvido.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    if caminho_resolvido.suffix == ".csv":
        df = pd.read_csv(caminho_resolvido)
        documentos = []
        for _, row in df.iterrows():
            texto = ", ".join(f"{col}: {val}" for col, val in row.items())
            metadados = row.to_dict()
            documentos.append(Document(page_content=texto, metadata=metadados))
        return documentos
    raise ValueError(f"Formato não suportado: {caminho_resolvido.suffix}")


def indexar(documentos: list[Document]) -> object:
    embeddings = _criar_embeddings()
    if settings.vector_store == "chroma":
        try:
            from langchain_chroma import Chroma
            return Chroma.from_documents(documentos, embeddings, persist_directory=str(INDEX_DIR))
        except ImportError:
            logger.warning("langchain-chroma não instalado. Usando FAISS.")
    FAISS = _obter_faiss()
    return FAISS.from_documents(documentos, embeddings)


def _salvar_indice(vectorstore):
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


def preparar():
    indice = _carregar_indice()
    if indice is not None:
        return indice
    documentos = carregar(settings.data_path)
    indice = indexar(documentos)
    _salvar_indice(indice)
    return indice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Força reconstrução do índice")
    args = parser.parse_args()
    if args.rebuild:
        import shutil
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
    preparar()
    print(f"Índice pronto em {INDEX_DIR}")
