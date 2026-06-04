from pathlib import Path

import pytest

from agente_edu.rag.ingest import carregar


def test_carregar_csv_valido():
    caminho = Path(__file__).resolve().parent.parent / "data" / "pib_piaui.csv"
    documentos = carregar(str(caminho))
    assert len(documentos) > 0
    for doc in documentos:
        assert "estado" in doc.metadata
        assert "pib_2023_bilhoes" in doc.metadata


def test_carregar_arquivo_inexistente():
    with pytest.raises(FileNotFoundError):
        carregar("data/nao_existe.csv")


def test_carregar_formato_invalido():
    with pytest.raises(ValueError, match="Formato não suportado"):
        carregar("data/arquivo.pdf")
