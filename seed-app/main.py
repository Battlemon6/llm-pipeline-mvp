import os
import requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://vllm-server:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "TheBloke/Mistral-7B-Instruct-v0.2-AWQ")
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "120"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

llm_response_data = "No query has been made yet."

def render_index() -> str:
    tpl_path = Path("templates") / "index.html"
    if not tpl_path.exists():
        return f"""<!doctype html>
<html><body>
<h1>LLM Query App</h1>
<p>(templates/index.html not found)</p>
<pre>{llm_response_data}</pre>
</body></html>"""
    html = tpl_path.read_text(encoding="utf-8")
    return html.replace("{{ server_response }}", llm_response_data)

@app.get("/", response_class=HTMLResponse)
async def show_webpage():
    return HTMLResponse(content=render_index(), status_code=200)

@app.post("/query/")
async def process_query(prompt: str = Form(...)):
    """Send the prompt to vLLM (OpenAI-compatible /v1/chat/completions)."""
    global llm_response_data

    url = f"{VLLM_ENDPOINT.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    resp = None
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=VLLM_TIMEOUT)
        resp.raise_for_status()

        data = resp.json()
        content = None
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice0 = data["choices"][0]
            content = (choice0.get("message") or {}).get("content") or choice0.get("text")

        if not content:
            snippet = str(data)[:500]
            raise ValueError(f"Unexpected response schema (no content): {snippet}")

        llm_response_data = f"Prompt: '{prompt}'\n\nResponse:\n{content}"

    except requests.exceptions.Timeout:
        llm_response_data = "Error: the request to vLLM timed out."
    except Exception as e:
        body = (resp.text[:500] if resp is not None and hasattr(resp, "text") else "")
        llm_response_data = f"Error talking to vLLM: {e}\n{body}"

    return RedirectResponse(url="/", status_code=HTTP_303_SEE_OTHER)