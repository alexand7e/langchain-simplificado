# Adaptar o agente para sua disciplina

Desafio da palestra em 4 passos:

## 1. Faça um fork

Crie um fork deste repositório para sua conta do GitHub.

## 2. Troque os dados

Coloque seus próprios arquivos em `data/` (CSV, futuramente PDF/TXT) e ajuste:

```dotenv
DATA_PATH=data/sua_disciplina.csv
```

## 3. Ajuste persona e regras

Edite a persona em `src/agente_edu/chains/prompt.py` (dicionário `PERSONAS`) ou
mude a persona padrão no `.env`:

```dotenv
DEFAULT_PERSONA=vestibular
```

Ajuste a `REGRA` em `src/agente_edu/guardrails/rules.py` para o comportamento
desejado.

## 4. Teste e publique

```bash
pip install -e ".[dev]"
python -m agente_edu.app
```

Teste no Telegram ou em http://localhost:8000/docs (interface web) e publique
com Docker na Mandu ou em qualquer host.
