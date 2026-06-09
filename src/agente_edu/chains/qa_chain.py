"""
Chain de perguntas e respostas — coração do agente.

Uma "chain" no LangChain é uma sequência de passos ligados pelo operador "|":

    prompt | modelo | parser

Cada passo recebe a saída do anterior e passa para o próximo.
"""
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from agente_edu.chains.prompt import montar_prompt
from agente_edu.guardrails.rules import REGRA, validar
from agente_edu.llm.soberano import Soberano


# ── Dataclass de debug ────────────────────────────────────────────────────────
# Usado pelo endpoint /chat/debug para expor cada etapa do pipeline ao aluno.

@dataclass
class DebugInfo:
    pergunta: str = ""
    contexto: str = ""
    persona: str = "ensino_medio"
    prompt_montado: str = ""      # o prompt exato enviado ao modelo
    resposta_bruta: str = ""      # o que o modelo devolveu, sem filtros
    guardrail_aplicado: bool = False
    resposta_final: str = ""      # o que o aluno recebe
    passos: list[str] = field(default_factory=list)


# ── Chain principal ───────────────────────────────────────────────────────────

def construir_chain(persona: str = "ensino_medio") -> Runnable:
    """Monta a chain: PromptTemplate → LLM → StrOutputParser."""
    return montar_prompt(persona) | Soberano() | StrOutputParser()


def responder(pergunta: str, contexto: str, persona: str = "ensino_medio") -> str:
    """Resposta simples: executa a chain e aplica o guardrail."""
    chain = construir_chain(persona)
    resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta, "regra": REGRA})
    return validar(resposta, contexto)


# ── Versão com detalhes (para fins didáticos / debug) ────────────────────────

def responder_com_detalhes(
    pergunta: str,
    contexto: str,
    persona: str = "ensino_medio",
) -> DebugInfo:
    """Executa a chain e devolve cada etapa intermediária para inspeção."""
    inputs = {"contexto": contexto or "(vazio)", "pergunta": pergunta, "regra": REGRA}

    # Formata o prompt para exibição — o mesmo texto que vai para o modelo
    prompt_texto = montar_prompt(persona).format(**inputs)

    # Executa a chain uma única vez
    chain = construir_chain(persona)
    resposta_bruta = chain.invoke(inputs)

    # Aplica o guardrail
    resposta_final = validar(resposta_bruta, contexto)
    guardrail_ativo = resposta_final != resposta_bruta

    passos = [
        "RAG: documentos encontrados na base vetorial" if contexto else "RAG: nenhum documento encontrado",
        "Guardrail: resposta bloqueada" if guardrail_ativo else "Guardrail: resposta aprovada",
    ]

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
