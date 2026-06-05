from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from agente_edu.chains.prompt import PERSONAS, TEMPLATE, montar_prompt
from agente_edu.guardrails.rules import REGRA, validar
from agente_edu.llm.soberano import Soberano


@dataclass
class DebugInfo:
    pergunta: str = ""
    contexto: str = ""
    persona: str = "ensino_medio"
    prompt_montado: str = ""
    resposta_bruta: str = ""
    guardrail_aplicado: bool = False
    resposta_final: str = ""
    passos: list[str] = field(default_factory=list)


def construir_chain(persona: str = "ensino_medio") -> Runnable:
    modelo = Soberano()
    prompt = montar_prompt(persona)
    parser = StrOutputParser()
    return prompt | modelo | parser


def responder(pergunta: str, contexto: str, persona: str = "ensino_medio") -> str:
    chain = construir_chain(persona)
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta, "regra": REGRA})
    return validar(resposta, contexto)


def responder_com_detalhes(
    pergunta: str,
    contexto: str,
    persona: str = "ensino_medio",
) -> DebugInfo:
    prompt_template = montar_prompt(persona)
    prompt_texto = prompt_template.format(
        contexto=contexto or "(vazio)",
        pergunta=pergunta,
        regra=REGRA,
    )

    chain = construir_chain(persona)
    resposta_bruta = chain.invoke({
        "contexto": contexto or "(vazio)",
        "pergunta": pergunta,
        "regra": REGRA,
    })

    resposta_final = validar(resposta_bruta, contexto)
    guardrail_ativo = resposta_final != resposta_bruta

    passos = []
    if contexto:
        passos.append("RAG: documentos encontrados na base vetorial")
    else:
        passos.append("RAG: nenhum documento encontrado")
    if guardrail_ativo:
        passos.append("Guardrail: resposta bloqueada (contexto vazio ou insuficiente)")
    else:
        passos.append("Guardrail: resposta aprovada")

    return DebugInfo(
        pergunta=pergunta,
        contexto=contexto,
        persona=persona,
        prompt_montado=prompt_texto,
        resposta_bruta=resposta_bruta,
        guardrail_aplicado=guardrail_ativo,
        resposta_final=resposta_final,
        passos=passos,
    )
