from agente_edu.guardrails.rules import FRASE_FALLBACK, REGRA, validar


def test_regra_nao_vazia():
    assert len(REGRA) > 0
    assert "Contexto" in REGRA
    assert "não encontrei" in REGRA.lower()


def test_frase_fallback_consistente():
    # A frase do código e a frase no prompt devem ser idênticas
    assert FRASE_FALLBACK in REGRA


def test_validar_contexto_vazio():
    resposta = "Qualquer resposta aqui"
    assert validar(resposta, contexto="") == FRASE_FALLBACK
    assert validar(resposta, contexto="   ") == FRASE_FALLBACK


def test_validar_contexto_valido():
    resposta = "O PIB do Piauí é 63.4 bilhões."
    contexto = "Piauí, pib_2023_bilhoes: 63.4"
    assert validar(resposta, contexto) == resposta


def test_guardrail_nivel2_independe_do_modelo():
    # Mesmo que o modelo "ignore" a instrução e responda algo,
    # o guardrail de código bloqueia quando o contexto é vazio.
    resposta_inventada = "O PIB do Piauí cresceu 10% em 2024."
    assert validar(resposta_inventada, contexto="") == FRASE_FALLBACK
