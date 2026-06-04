FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE .
COPY src/ src/
RUN pip install --no-cache-dir ".[dev]"

COPY . .

EXPOSE 8000

CMD ["python", "-m", "agente_edu.app"]
