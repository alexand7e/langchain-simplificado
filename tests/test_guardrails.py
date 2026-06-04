from agente_edu.guardrails.rules import REGRA, validar


def test_regra_nao_vazia():
    assert len(REGRA) > 0
    assert "Contexto" in REGRA
    assert "não encontrei" in REGRA.lower()


def test_validar_contexto_vazio():
    resposta = "Qualquer resposta aqui"
    resultado = validar(resposta, contexto="")
    assert resultado == "Não encontrei essa informação nas fontes disponíveis."
    resultado = validar(resposta, contexto="   ")
    assert resultado == "Não encontrei essa informação nas fontes disponíveis."


def test_validar_contexto_valido():
    resposta = "O PIB do Piauí é 63.4 bilhões."
    contexto = "Piauí, pib_2023_bilhoes: 63.4"
    resultado = validar(resposta, contexto)
    assert resultado == resposta
