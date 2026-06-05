import json
import os
import urllib.request

# carrega .env simples
env = {}
with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()

base = env.get("SOBERANO_API_BASE_URL", "")
key = env.get("SOBERANO_API_KEY", "")
model = env.get("SOBERANO_MODEL", "")
print("BASE:", base)
print("MODEL:", model)
print("KEY set:", bool(key), "len", len(key))


def probe(url, method="GET", body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"[{method}] {url} -> {r.status}")
            print(r.read()[:800].decode(errors="replace"))
    except urllib.error.HTTPError as e:
        print(f"[{method}] {url} -> HTTP {e.code}")
        print(e.read()[:800].decode(errors="replace"))
    except Exception as e:
        print(f"[{method}] {url} -> ERR {e}")
    print("-" * 60)


probe(f"{base}/models")
probe(f"{base}/chat/completions", "POST",
      {"model": model, "messages": [{"role": "user", "content": "diga oi"}]})
