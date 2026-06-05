import json
import urllib.request

env = {}
with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

base = env.get("SOBERANO_API_BASE_URL", "")
key = env.get("SOBERANO_API_KEY", "")


def probe(url, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"[{method}] {url} -> {r.status}")
            print(r.read()[:600].decode(errors="replace"))
    except urllib.error.HTTPError as e:
        print(f"[{method}] {url} -> HTTP {e.code}")
        print(e.read()[:400].decode(errors="replace"))
    except Exception as e:
        print(f"[{method}] {url} -> ERR {e}")
    print("-" * 60)


# testa modelos de chat
for m in ["Qwen/Qwen3.6-35B-A3B", "soberano-alpha"]:
    print("### chat model:", m)
    probe(f"{base}/chat/completions", "POST",
          {"model": m, "messages": [{"role": "user", "content": "Responda apenas: oi"}], "max_tokens": 20})

# testa embeddings
print("### embeddings bge-m3")
probe(f"{base}/embeddings", "POST", {"model": "BAAI/bge-m3", "input": "teste de embedding"})
