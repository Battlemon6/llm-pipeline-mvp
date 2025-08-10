import requests
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.status import HTTP_303_SEE_OTHER

# Create the FastAPI application instance
app = FastAPI()

# A global variable to hold the latest response from the LLM.
llm_response_data = "No query has been made yet."

@app.get("/", response_class=HTMLResponse)
async def show_webpage():
    """Reads the HTML file, injects the last known response, and serves it."""
    with open("templates/index.html") as f:
        html_content = f.read()
    
    html_with_data = html_content.replace("{{ server_response }}", llm_response_data)
    
    return HTMLResponse(content=html_with_data, status_code=200)

# In main.py, update this function:

@app.post("/query/")
async def process_query(prompt: str = Form(...)):
    """
    Receives a prompt from a form, sends it to the Ollama API, 
    and then redirects the user back to the homepage.
    """
    global llm_response_data

    ollama_api_url = "http://ollama-service:11434/api/generate"
    json_payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        print("--- Checkpoint 1: Sending request to Ollama... ---")
        # Add an explicit timeout of 120 seconds (2 minutes)
        response = requests.post(ollama_api_url, json=json_payload, timeout=120)
        print("--- Checkpoint 2: Request sent. Checking status... ---")
        
        response.raise_for_status()
        print("--- Checkpoint 3: Status OK. Parsing JSON... ---")
        
        response_json = response.json()
        formatted_response = f"Prompt: '{prompt}'\n\nResponse:\n{response_json['response']}"
        llm_response_data = formatted_response
        print("--- Checkpoint 4: Global variable updated successfully. ---")

    except requests.exceptions.Timeout:
        error_message = "Error: The request to Ollama timed out after 2 minutes."
        print(f"--- FAILED: {error_message} ---")
        llm_response_data = error_message
    
    except requests.exceptions.RequestException as e:
        error_message = f"Error communicating with Ollama: {e}"
        print(f"--- FAILED: {error_message} ---")
        llm_response_data = error_message
    
    print("--- Checkpoint 5: Redirecting user to homepage. ---")
    return RedirectResponse(url="/", status_code=HTTP_303_SEE_OTHER)