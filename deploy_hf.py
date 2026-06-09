#!/usr/bin/env python
"""
Publica/atualiza este projeto em um Hugging Face Space do tipo Docker.

Por que existe: o `data/livro.pdf` (~60 MB) está versionado como blob comum, e o
`git push` para o HF exigiria migrar para LFS. O `upload_folder` envia direto do
disco e converte arquivos grandes para LFS automaticamente, sem mexer no histórico
git — e ainda inclui os índices `.index/` (que ficam fora do git pelo .gitignore).

Pré-requisitos:
  pip install -U "huggingface_hub[cli]"
  export HF_TOKEN=hf_...        # token com escopo Write (no PowerShell: $env:HF_TOKEN="hf_...")

Uso:
  python deploy_hf.py                       # usa o SPACE padrão abaixo
  python deploy_hf.py outro-user/outro-space

Lembre-se de configurar o segredo SOBERANO_API_KEY no Space
(Settings → Variables and secrets). As demais configs já têm default em config.py.
"""
import sys

from huggingface_hub import upload_folder

SPACE = sys.argv[1] if len(sys.argv) > 1 else "Alexand7e/sia"

# Não enviar: ambiente local, caches, segredos e certificados.
IGNORE = [
    ".git/*", ".venv/*", "venv/*",
    "**/__pycache__/*", "**/*.pyc",
    ".env", "*.egg-info/*",
    ".claude/*", "*.crt", "*.key",
    ".pytest_cache/*", ".ruff_cache/*", ".mypy_cache/*",
    "deploy_hf.py",
]


def main() -> None:
    url = upload_folder(
        folder_path=".",
        repo_id=SPACE,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy: agente educacional (Docker) + indices RAG pre-construidos",
    )
    print("OK:", url)
    owner, name = SPACE.split("/")
    print(f"App: https://{owner.lower()}-{name.lower()}.hf.space")


if __name__ == "__main__":
    main()
