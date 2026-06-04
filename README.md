# Agente Educacional — LangChain + Soberano 1.1

Agente de IA conversacional em português que responde perguntas a partir de fontes confiáveis (RAG). Projeto didático que acompanha a palestra **"LangChain na sala de aula"** — do Piauí para o Mundo.

## Quickstart

```bash
git clone <url-do-repositorio>
cd langchain-simplificado
cp .env.example .env
# Preencha SOBERANO_API_KEY e TELEGRAM_BOT_TOKEN no .env
pip install -e ".[dev]"
python -m agente_edu.app
```

> Roda em **menos de 5 minutos**. O índice vetorial é criado automaticamente na primeira execução.

## Arquitetura

```
Aluno (Telegram/Web) → Interface → RAG (busca contexto) → Prompt → Soberano 1.1 → Guardrail → Resposta
```

![Diagrama de arquitetura](docs/img/arquitetura.png)

| Camada        | Pasta              | Responsabilidade                    |
|---------------|--------------------|-------------------------------------|
| Interface     | `interfaces/`      | Telegram / Web (FastAPI)            |
| Orquestração  | `chains/`, `graph/`| Chain LCEL ou LangGraph             |
| Conhecimento  | `rag/`             | Carregar, indexar e buscar fontes   |
| Modelo        | `llm/`             | Cliente Soberano 1.1 (Mandu)        |
| Segurança     | `guardrails/`      | Regras e validação da resposta      |

## Mapa slide → componente

| Slide | Componente | Arquivo |
|-------|------------|---------|
| 6     | Cliente Soberano | `llm/soberano.py` |
| 7     | RAG (ingestão + busca) | `rag/ingest.py`, `rag/retriever.py` |
| 8     | Prompt + Chain | `chains/prompt.py`, `chains/qa_chain.py` |
| 9     | Guardrails | `guardrails/rules.py` |
| 10    | Interfaces | `interfaces/telegram_bot.py`, `interfaces/web_api.py` |
| 12-13 | LangGraph | `graph/agent_graph.py` |

## Próximos passos

Quer adaptar para sua disciplina? Veja o [guia de adaptação](docs/adaptar-disciplina.md).

## Contribuir

Contribuições são bem-vindas! Abra uma issue ou envie um PR.

## Licença

MIT
