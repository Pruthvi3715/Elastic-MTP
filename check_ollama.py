"""
Ollama Model Discovery & Local Integration Probe.

Queries local Ollama API (http://localhost:11434/api/tags) and lists available downloaded models.
"""
import urllib.request
import json

def check_ollama():
    url = "http://localhost:11434/api/tags"
    print(f"Connecting to local Ollama server at {url}...")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            models = data.get("models", [])
            print(f"\n[Success] Connected to Ollama! Found {len(models)} local model(s):")
            for idx, m in enumerate(models, 1):
                name = m.get("name")
                size_mb = m.get("size", 0) / (1024 * 1024)
                family = m.get("details", {}).get("family", "N/A")
                print(f"  [{idx}] Model: '{name:<20}' | Size: {size_mb:>7.1f} MB | Family: {family}")
            return models
    except Exception as e:
        print(f"[Notice] Could not connect to running Ollama daemon on port 11434: {e}")
        print("Make sure Ollama desktop app or 'ollama serve' is running!")
        return []

if __name__ == "__main__":
    check_ollama()
