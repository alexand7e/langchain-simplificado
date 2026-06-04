REGRA: str = (
    "Você só pode responder com base no Contexto das fontes fornecido. "
    "Se o Contexto estiver vazio ou não contiver a informação perguntada, "
    "responda exatamente: 'Não encontrei essa informação nas fontes disponíveis.' "
    "Não complete nem invente dados."
)


def validar(resposta: str, contexto: str) -> str:
    if not contexto or contexto.strip() == "":
        return "Não encontrei essa informação nas fontes disponíveis."
    return resposta
