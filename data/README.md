# Dados do agente educacional

Adicione aqui os arquivos de dados que o agente vai usar como fonte de conhecimento.

## Formato esperado

### CSV
O formato principal é CSV com cabeçalho. Cada linha vira um documento no índice vetorial.

Colunas esperadas no dataset de exemplo (`pib_piaui.csv`):
- `estado` — nome da unidade federativa
- `regiao` — região geográfica
- `pib_2023_bilhoes` — PIB em bilhões de reais (2023)
- `idh_2022` — Índice de Desenvolvimento Humano (2022)
- `populacao_milhoes` — população em milhões
- `area_km2` — área em km²

### Como adicionar uma fonte
1. Coloque o arquivo em `data/`
2. Aponte `DATA_PATH` no `.env` para o novo caminho
3. Execute a reindexação:
   ```bash
   python -m agente_edu.rag.ingest --rebuild
   ```

### Origem dos dados
Os dados do `pib_piaui.csv` são baseados em fontes públicas do IBGE, IPEA e PNUD.
Sempre cite a origem dos seus dados.
