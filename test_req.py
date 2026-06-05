import json
import urllib.request

payload = json.dumps({"mensagem": "Qual o PIB do Piaui?", "persona": "ensino_medio"}).encode()
req = urllib.request.Request(
    "http://localhost:8000/chat/debug",
    data=payload,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read())
    for k, v in data.items():
        print(f"=== {k} ===")
        print(v)
except Exception as e:
    print("ERRO:", e)
