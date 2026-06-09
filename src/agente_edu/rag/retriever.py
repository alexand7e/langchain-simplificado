from langchain.schema.document import Document

from agente_edu.config import settings
from agente_edu.rag.ingest import preparar


class Retriever:
    def __init__(self):
        self._vectorstore = None

    def _obter_store(self):
        if self._vectorstore is None:
            self._vectorstore = preparar()
        return self._vectorstore

    def buscar(self, pergunta: str, k: int = 3) -> str:
        store = self._obter_store()
        documentos = store.similarity_search(pergunta, k=k)
        if not documentos:
            return ""
        return "\n\n".join(doc.page_content for doc in documentos)

    def adicionar(self, texto: str, titulo: str = "") -> None:
        """Adiciona um documento ao índice em tempo real (sem persistir em disco)."""
        store = self._obter_store()
        doc = Document(
            page_content=texto,
            metadata={"titulo": titulo, "fonte": "ingestão_manual"},
        )
        store.add_documents([doc])
