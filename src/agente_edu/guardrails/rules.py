"""
Guardrails — regras de segurança do agente.

Existem dois níveis de proteção, e é importante entender os dois:

  1. Guardrail no PROMPT (REGRA): instrui o modelo a se recusar quando
     não tiver informação. Funciona "de dentro" — depende do modelo
     seguir a instrução.

  2. Guardrail no CÓDIGO (validar): verifica programaticamente *antes*
     de enviar a resposta. Funciona "de fora" — independe do modelo.

Os dois níveis trabalham juntos: o prompt reduz alucinações, o código
garante que contexto vazio nunca gere uma resposta inventada.
"""

# Frase padrão de fallback — usada tanto no prompt quanto no código.
FRASE_FALLBACK = "Não encontrei essa informação nas fontes disponíveis."

# Instrução embutida no prompt enviado ao modelo (guardrail nível 1).
REGRA: str = (
    "Você só pode responder com base no Contexto das fontes fornecido. "
    f"Se o Contexto estiver vazio ou não contiver a informação perguntada, "
    f"responda exatamente: '{FRASE_FALLBACK}' "
    "Não complete nem invente dados."
)


def validar(resposta: str, contexto: str) -> str:
    """
    Guardrail nível 2 (código): bloqueia a resposta se o contexto estiver vazio.

    Por que isso é necessário se já temos REGRA no prompt?
    Porque modelos de linguagem podem ignorar instruções. Este filtro
    garante que a política seja respeitada independentemente do modelo.
    """
    if not contexto or contexto.strip() == "":
        return FRASE_FALLBACK
    return resposta
