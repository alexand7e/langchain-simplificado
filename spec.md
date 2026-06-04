# SPEC — Agente Educacional com LangChain + Soberano 1.1

> Especificação do repositório que dá suporte à palestra **"LangChain na sala de aula"**
> (projeto *Do Piauí para o Mundo*). Este documento é a fonte da verdade para
> construir, organizar e evoluir o repositório. Quem clona deve conseguir **rodar em
> minutos** e **adaptar para a própria disciplina** sem se perder.

---

## 1. Propósito

Construir um agente de IA conversacional, em português, que responde **a partir de
fontes confiáveis** (RAG) e que serve de **base didática**: o código é simples o
suficiente para um aluno do ensino médio ler, e estruturado o suficiente para escalar.

O agente roda sobre o **Soberano 1.1** (LLM em português, via API de inferência) na
infraestrutura **Mandu**, e conversa pelo **Telegram** ou por um **chat web simples**.

Dois públicos, um repositório:
- **Aluno/curioso:** escaneia o QR, conversa com o agente, lê o código e entende como é montado.
- **Quem vai construir:** faz fork, troca os dados pela própria matéria e publica seu agente.

---

## 2. Princípios de design (não negociáveis)

1. **Espelhar os slides.** Cada componente do código corresponde a um slide da palestra
   (ver §6). Quem escaneia o QR reconhece exatamente o que viu.
2. **Legível antes de esperto.** Funções curtas, nomes claros, um arquivo por
   responsabilidade. Nada de abstração prematura.
3. **RAG primeiro.** O agente nunca responde de memória sobre dados; sempre recupera a
   fonte. Não inventar é requisito, não enfeite.
4. **Guardrails explícitos.** Se a resposta não está nos dados, o agente admite.
5. **Configurável, não hardcoded.** Modelo, dados, persona e regras vêm de
   configuração (`.env` / `config.py`), nunca chumbados no código.
6. **Escalável por padrão.** Stateless no caminho da requisição; estado externalizado;
   pronto para uma turma inteira ao mesmo tempo (ver §11).

---

## 3. Arquitetura

Fluxo de uma pergunta, ponta a ponta (igual ao slide 11):

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

Camadas e responsabilidades:

| Camada        | Papel                                                        | Pasta              |
|---------------|-------------------------------------------------------------|--------------------|
| Interface     | Recebe a mensagem e devolve a resposta (Telegram / Web)     | `interfaces/`      |
| Orquestração  | Liga as peças: RAG → prompt → modelo → guardrail            | `chains/`, `graph/`|
| Conhecimento  | Carrega e busca nas fontes (RAG)                            | `rag/`             |
| Modelo        | Cliente do Soberano 1.1 na Mandu                            | `llm/`             |
| Segurança     | Regras de comportamento (guardrails)                        | `guardrails/`      |
| Configuração  | Variáveis de ambiente e settings                           | `config.py`        |

---

## 4. Stack tecnológica

- **Python 3.11+**
- **LangChain** (`langchain`, `langchain-core`, `langchain-community`) + **LangGraph** (fluxo avançado, opcional)
- **Soberano 1.1** via API de inferência (cliente próprio em `llm/soberano.py`)
- **Vector store:** FAISS ou Chroma (local, para a demo) — trocável por store persistente em produção
- **Embeddings:** modelo multilíngue/PT-BR configurável (ou endpoint de embeddings do Soberano/Mandu, se disponível)
- **Interfaces:** `python-telegram-bot` (Telegram) e **FastAPI** (chat web simples)
- **Config:** `pydantic-settings` (lê `.env`)
- **Empacotamento:** Docker + `docker-compose`
- **Qualidade:** `pytest`, `ruff` (lint), `mypy` (opcional), GitHub Actions

> ⚠️ O contrato exato da API do Soberano 1.1 (URL, autenticação, formato de payload)
> deve ser preenchido ao integrar. O cliente em `llm/soberano.py` é um **adaptador**:
> recebe `base_url`, `api_key` e `model` por configuração e expõe a interface de
> LLM/ChatModel que o LangChain espera. Ajustar o corpo da requisição ao contrato real.

---

## 5. Estrutura de diretórios

```
langchain-sala-de-aula/
├── README.md                  # porta de entrada (ver §10)
├── SPEC.md                    # este documento
├── LICENSE                    # licença aberta (MIT recomendada)
├── .env.example               # todas as variáveis, sem segredos reais
├── pyproject.toml             # dependências e metadados
├── Dockerfile
├── docker-compose.yml         # sobe agente + (opcional) Redis/store
├── docs/
│   ├── arquitetura.md         # diagrama e fluxo (espelha slides 6 e 11)
│   ├── adaptar-disciplina.md  # o desafio passo a passo (§10)
│   └── img/                   # diagramas iguais aos da palestra
├── data/
│   ├── README.md              # como adicionar uma fonte
│   └── pib_piaui.csv          # dataset de exemplo (PIB/IDH)
├── src/agente_edu/
│   ├── __init__.py
│   ├── config.py              # settings via .env
│   ├── app.py                 # ponto de entrada (escolhe interface)
│   ├── llm/
│   │   └── soberano.py        # cliente do Soberano 1.1 (Mandu)  [slide 6]
│   ├── rag/
│   │   ├── ingest.py          # carregar + indexar fontes        [slide 7]
│   │   └── retriever.py       # buscar contexto                  [slide 7]
│   ├── chains/
│   │   ├── prompt.py          # PromptTemplate + personas        [slide 8]
│   │   └── qa_chain.py        # prompt | soberano | parser       [slide 8]
│   ├── guardrails/
│   │   └── rules.py           # regras + validação               [slide 9]
│   ├── graph/
│   │   └── agent_graph.py     # StateGraph (decidir/voltar)      [slides 12-13]
│   └── interfaces/
│       ├── telegram_bot.py    # handler do Telegram              [slide 10]
│       └── web_api.py         # chat web simples (FastAPI)
├── tests/
│   ├── test_rag.py
│   ├── test_chain.py
│   └── test_guardrails.py
├── notebooks/
│   └── exemplo_pib.ipynb      # demo didática (opcional)
└── .github/
    ├── workflows/ci.yml       # lint + testes
    └── ISSUE_TEMPLATE/
```

---

## 6. Componentes principais (mapa slide → código)

Cada item lista: **responsabilidade**, **classes/funções-chave** e **arquivo**.

### 6.1 RAG — as fontes  ·  *slide 7*  ·  `rag/`
- **Responsabilidade:** carregar a fonte (CSV, PDF, etc.), indexar para busca semântica e recuperar o trecho relevante de cada pergunta.
- **Chave:** `carregar(caminho) -> list[Document]`, `indexar(docs) -> VectorStore`, `class Retriever.buscar(pergunta, k=3) -> str`.
- **Regra:** o contexto recuperado é **sempre** o que alimenta o prompt; nada de resposta "de cabeça".

### 6.2 Prompt + Chain — a montagem  ·  *slide 8*  ·  `chains/`
- **Responsabilidade:** montar o prompt com `persona`, `contexto` e `pergunta`, e ligar `prompt | soberano | parser`.
- **Chave:** `montar_prompt(persona) -> PromptTemplate`, `construir_chain() -> Runnable`, `PERSONAS: dict` (ex.: `"crianca"`, `"vestibular"`, `"simples"`).
- **Conceito didático:** o operador `|` (LCEL) é como o `>>` da palestra do Airflow — ligar peças.

### 6.3 Guardrails — as regras  ·  *slide 9*  ·  `guardrails/`
- **Responsabilidade:** definir a regra de comportamento e validar a saída.
- **Chave:** `REGRA: str` (entra no prompt do sistema), `validar(resposta, contexto) -> resposta_segura`.
- **Comportamento mínimo:** sem contexto suficiente → responde "não encontrei isso nas fontes".

### 6.4 LLM — o modelo  ·  *slide 6*  ·  `llm/soberano.py`
- **Responsabilidade:** adaptador do Soberano 1.1 (na Mandu) para a interface de LLM do LangChain.
- **Chave:** `class Soberano(BaseChatModel)` com `base_url`, `api_key`, `model`, timeout e retry/backoff.

### 6.5 Interface — o canal  ·  *slide 10*  ·  `interfaces/`
- **Responsabilidade:** receber a mensagem do aluno e devolver a resposta. Telegram e Web são **intercambiáveis** (mesma função `responder(pergunta, persona) -> str` por baixo).
- **Chave:** `telegram_bot.py: on_message`, `web_api.py: POST /chat`.

### 6.6 LangGraph — quando decide  ·  *slides 12-13*  ·  `graph/agent_graph.py`
- **Responsabilidade:** fluxo com decisão/ciclo (ex.: "tem nos dados? → responde / busca mais").
- **Chave:** `StateGraph`, nós (`recuperar`, `responder`, `buscar_mais`), arestas condicionais, `state` (memória da conversa).
- **Opcional:** o agente funciona sem ele (chain linear); o grafo é o "nível avançado".

---

## 7. Configuração (`.env.example`)

Nunca commitar segredos. Copiar para `.env` e preencher.

```dotenv
# --- Modelo (Soberano 1.1 / Mandu) ---
SOBERANO_API_BASE_URL=https://<endpoint-da-mandu>
SOBERANO_API_KEY=
SOBERANO_MODEL=soberano-1.1

# --- Embeddings (RAG) ---
EMBEDDINGS_PROVIDER=local        # local | soberano
EMBEDDINGS_MODEL=intfloat/multilingual-e5-small

# --- Fonte de dados ---
DATA_PATH=data/pib_piaui.csv
VECTOR_STORE=faiss               # faiss | chroma
VECTOR_STORE_PATH=.index

# --- Comportamento ---
DEFAULT_PERSONA=ensino_medio
TOP_K=3

# --- Interface ---
INTERFACE=telegram               # telegram | web
TELEGRAM_BOT_TOKEN=
WEB_PORT=8000

# --- Escala / estado (opcional) ---
REDIS_URL=                       # memória de conversa, se usado
RATE_LIMIT_POR_MIN=10
LOG_LEVEL=INFO
```

Toda variável é lida em `config.py` via `pydantic-settings`, com defaults sensatos e
validação na inicialização (falhar cedo se faltar segredo essencial).

---

## 8. Setup e execução

**Local (desenvolvimento):**
```bash
git clone https://github.com/<org>/langchain-sala-de-aula
cd langchain-sala-de-aula
cp .env.example .env            # preencha SOBERANO_API_KEY e TELEGRAM_BOT_TOKEN
pip install -e ".[dev]"
python -m agente_edu.app        # sobe a interface definida em INTERFACE
```

**Com Docker (produção / Mandu):**
```bash
docker compose up -d --build
docker compose logs -f agente
```

O quickstart do README deve rodar em **menos de 5 minutos**. Sem build manual de índice:
o `ingest` roda na primeira execução e persiste em `VECTOR_STORE_PATH`.

---

## 9. Dados e RAG

- Formato inicial: **CSV** (ex.: PIB/IDH por estado/país). Suportar PDF e TXT depois.
- Adicionar uma fonte = colocar o arquivo em `data/` e apontar `DATA_PATH` (ou lista).
- `data/README.md` explica colunas esperadas e como citar a origem dos dados.
- **Reindexação:** comando `python -m agente_edu.rag.ingest --rebuild` quando a fonte muda.
- **Boa prática:** XCom-style — não jogar arquivos enormes no contexto; recuperar só os `TOP_K` trechos relevantes.

---

## 10. Documentação esperada ("à prova de QR")

O `README.md` deve conter, nesta ordem:
1. Uma frase do que é + GIF/print do agente respondendo.
2. **Quickstart** (copiar/colar, roda em <5 min).
3. **Diagrama de arquitetura** (o mesmo dos slides 6 e 11, em `docs/img/`).
4. Mapa "slide → componente" (o da §6) para quem veio da palestra.
5. Link para `docs/adaptar-disciplina.md`.
6. Como contribuir + licença.

`docs/adaptar-disciplina.md` — **o desafio da palestra**, em 4 passos:
1. Faça um *fork*.
2. Troque os dados em `data/` pela sua disciplina ou cidade (e ajuste `DATA_PATH`).
3. Ajuste a `persona` (em `chains/prompt.py`/`config.py`) e a `REGRA` (em `guardrails/rules.py`).
4. Rode local, teste no Telegram/web e publique (Docker na Mandu ou outro host).

> Regra de ouro da documentação: **o slide e o repositório contam a mesma história.**
> Se um nome de classe muda no código, muda no slide também.

---

## 11. Escalabilidade (uma turma inteira ao mesmo tempo)

- **Stateless no caminho da requisição:** cada mensagem é independente; estado de
  conversa (se houver) vive em `REDIS_URL`, não em memória do processo.
- **Concorrência:** handlers assíncronos; pool de conexões para a API do Soberano.
- **Rate limit por usuário** (`RATE_LIMIT_POR_MIN`) com mensagem amigável ao estourar.
- **Cache:** índice vetorial persistido; cache de embeddings; opcional cache de respostas frequentes.
- **Horizontal:** o bot/API deve rodar em N réplicas atrás de um balanceador.
- **Resiliência:** retry com backoff e timeout na chamada ao Soberano; resposta de
  fallback se a API falhar (não travar a aula).
- **Teste de carga antes do evento:** simular a turma inteira disparando ao mesmo tempo.
- **Observabilidade:** `LOG_LEVEL`, logs estruturados, health check (`/health`).

---

## 12. Segurança e LGPD

- Segredos só por variável de ambiente; `.env` no `.gitignore`.
- Não armazenar dados pessoais de alunos. Se registrar perguntas (caso "RAG reverso"),
  **anonimizar** e deixar claro no README o que é guardado e por quê.
- Dados das fontes devem ser **públicos e citáveis** (coerente com a proposta de soberania).
- Dependências fixadas (`pyproject.toml`) e varredura básica no CI.

---

## 13. Qualidade e CI

- `pytest` cobrindo: RAG retorna contexto, chain monta o prompt certo, guardrail bloqueia
  resposta sem fonte.
- `ruff` para lint/format; `mypy` opcional.
- `.github/workflows/ci.yml`: roda lint + testes em cada PR.
- Testes não devem exigir a API real do Soberano: usar um **LLM falso** (stub) injetável.

---

## 14. Convenção de nomes (decisão)

Para manter o vínculo com a palestra e o público misto, as funções/classes **de fronteira**
(as que aparecem nos slides) usam nomes em português: `carregar`, `buscar`, `montar_prompt`,
`REGRA`, `responder`. Comentários e docstrings em português. Internamente, nomes técnicos
em inglês são aceitáveis. *Se preferir tudo em inglês, troque também nos slides — a regra é
que os dois batam.*

---

## 15. Roadmap sugerido

- **v0.1 — Demo da palestra:** chain linear (RAG + prompt + guardrail) no Telegram, dataset PIB.
- **v0.2 — Web + Docker:** chat web simples e deploy reproduzível na Mandu.
- **v0.3 — LangGraph:** fluxo com decisão/ciclo.
- **v0.4 — RAG reverso:** registrar dúvidas recorrentes (anonimizadas) e realimentar a base.
- **v1.0 — Multi-fonte e escala:** store persistente, réplicas, observabilidade.

---

## 16. Checklist de aceite ("pronto para o QR")

- [ ] `README` com quickstart que roda em < 5 min.
- [ ] `.env.example` completo e sem segredos reais.
- [ ] Um dataset de exemplo em `data/` e o agente respondendo a partir dele.
- [ ] Diagrama de arquitetura igual ao dos slides em `docs/img/`.
- [ ] Mapa "slide → componente" no README.
- [ ] `docs/adaptar-disciplina.md` com os 4 passos do desafio.
- [ ] Guardrail funcionando (resposta fora dos dados → "não encontrei nas fontes").
- [ ] Testes passando no CI.
- [ ] `LICENSE` aberta presente.
- [ ] Teste de carga feito para o tamanho da turma.