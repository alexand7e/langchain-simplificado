from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from agente_edu.chains.prompt import montar_prompt
from agente_edu.guardrails.rules import REGRA, validar
from agente_edu.llm.soberano import Soberano


def construir_chain(persona: str = "ensino_medio") -> Runnable:
    modelo = Soberano()
    prompt = montar_prompt(persona)
    parser = StrOutputParser()
    return prompt | modelo | parser


def responder(pergunta: str, contexto: str, persona: str = "ensino_medio") -> str:
    chain = construir_chain(persona)
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta, "regra": REGRA})
    return validar(resposta, contexto)
