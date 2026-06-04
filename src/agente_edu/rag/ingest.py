import argparse
from pathlib import Path

import pandas as pd
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agente_edu.config import INDEX_DIR, settings


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


def _criar_embeddings():
    return HuggingFaceEmbeddings(model_name=settings.embeddings_model)


def indexar(documentos: list[Document]) -> FAISS | Chroma:
    embeddings = _criar_embeddings()
    if settings.vector_store == "chroma":
        return Chroma.from_documents(documentos, embeddings, persist_directory=str(INDEX_DIR))
    return FAISS.from_documents(documentos, embeddings)


def _salvar_indice(vectorstore: FAISS | Chroma):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(vectorstore, FAISS):
        vectorstore.save_local(str(INDEX_DIR))
    elif isinstance(vectorstore, Chroma):
        pass


def _carregar_indice() -> FAISS | Chroma | None:
    if not INDEX_DIR.exists():
        return None
    embeddings = _criar_embeddings()
    if settings.vector_store == "chroma":
        return Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings)
    try:
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None


def preparar() -> FAISS | Chroma:
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
