# import necessary libraries
import os
import requests
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER
from pathlib import Path

# create a new FastAPI application instance
app = FastAPI()

# Read configuration from environment variables.
# These are set in the kubernetes deployment file.
INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL")
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "120"))
VLLM_API_KEY = os.getenv("VLLM_API_KEY")

# A simple global variable to hold the last response.
# This is just for demonstration purposes.
llm_response_data = "No query has been made yet."

# This function reads the index.html file from the templates directory.
def render_index() -> str:
    tpl_path = Path("templates") / "index.html"
    # A fallback in case the html file is not found.
    if not tpl_path.exists():
        return f"""<!doctype html>
<html><body>
<h1>LLM Query App</h1>
<p>(templates/index.html not found)</p>
<pre>{llm_response_data}</pre>
</body></html>"""
    # Read the html content and replace the placeholder with the actual response.
    html = tpl_path.read_text(encoding="utf-8")
    return html.replace("{{ server_response }}", llm_response_data)

# This is the main endpoint that serves the HTML page.
@app.get("/", response_class=HTMLResponse)
async def show_webpage():
    return HTMLResponse(content=render_index(), status_code=200)

# This endpoint handles the form submission from the webpage.
@app.post("/query/")
async def process_query(prompt: str = Form(...)):
    """Send the prompt to the inference server (OpenAI-compatible /v1/chat/completions)."""
    # We need to modify the global variable to store the new response.
    global llm_response_data

    # Check if the environment variables for the model server are set.
    if not INFERENCE_ENDPOINT or not INFERENCE_MODEL:
        llm_response_data = "Error: INFERENCE_ENDPOINT or INFERENCE_MODEL is not configured."
        return RedirectResponse(url="/", status_code=HTTP_303_SEE_OTHER)

    # Construct the full URL for the model's API endpoint.
    url = f"{INFERENCE_ENDPOINT.rstrip('/')}/v1/chat/completions"
    # Prepare the payload in the format expected by the OpenAI-compatible server.
    payload = {
        "model": INFERENCE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    # If an API key is provided, add it to the request headers.
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    resp = None
    try:
        # Send the request to the inference server.
        resp = requests.post(url, json=payload, headers=headers, timeout=INFERENCE_TIMEOUT)
        # Raise an error if the response status code is not successful (e.g., 4xx or 5xx).
        resp.raise_for_status()

        data = resp.json()
        content = None
        # The response structure can vary slightly, so we check a few common places for the content.
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice0 = data["choices"][0]
            content = (choice0.get("message") or {}).get("content") or choice0.get("text")

        # If we couldn't find any content, the response format is unexpected.
        if not content:
            snippet = str(data)[:500]
            raise ValueError(f"Unexpected response schema (no content): {snippet}")

        # Format the final response to be displayed on the page.
        llm_response_data = f"Prompt: '{prompt}'\n\nResponse:\n{content}"

    # Handle specific error cases, like the request timing out.
    except requests.exceptions.Timeout:
        llm_response_data = "Error: the request to the inference server timed out."
    # Handle any other exceptions that might occur.
    except Exception as e:
        body = (resp.text[:500] if resp is not None and hasattr(resp, "text") else "")
        llm_response_data = f"Error talking to the inference server: {e}\n{body}"

    # After handling the query, redirect the user back to the main page to see the result.
    return RedirectResponse(url="/", status_code=HTTP_303_SEE_OTHER)
