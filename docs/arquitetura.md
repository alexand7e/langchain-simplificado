# Arquitetura do Agente Educacional

## Fluxo ponta a ponta

```
Aluno (Telegram/Web)
      │  pergunta
      ▼
 Interface (bot/api)  ──►  RAG: recupera contexto da fonte
      │                          │
      │                          ▼
      │                    Prompt (com persona + contexto + regra)
      │                          │
      │                          ▼
      │                    Soberano 1.1  (API de inferência, na Mandu)
      │                          │
      │                          ▼
      │                    Guardrail valida a resposta
      ▼                          │
 resposta confiável  ◄───────────┘
```

## Camadas

| Camada        | Papel                                       | Pasta              |
|---------------|---------------------------------------------|--------------------|
| Interface     | Recebe a mensagem e devolve a resposta      | `interfaces/`      |
| Orquestração  | Liga RAG → prompt → modelo → guardrail      | `chains/`, `graph/`|
| Conhecimento  | Carrega e busca nas fontes (RAG)            | `rag/`             |
| Modelo        | Cliente do Soberano 1.1 na Mandu            | `llm/`             |
| Segurança     | Regras de comportamento (guardrails)        | `guardrails/`      |
| Configuração  | Variáveis de ambiente e settings            | `config.py`        |

## Princípios

1. **Legível antes de esperto** — funções curtas, nomes claros
2. **RAG primeiro** — nunca responde de memória sobre os dados
3. **Guardrails explícitos** — admite quando não sabe
4. **Configurável** — modelo, dados, persona via `.env`
