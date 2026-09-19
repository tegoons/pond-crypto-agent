import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from openai import OpenAI

app = FastAPI(title="Pond Crypto AI Agent", version="1.0.0")

POND_ACCESS_KEY = os.getenv("POND_ACCESS_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def check_auth(authorization: str | None, x_access_key: str | None):
    # Supports the two common API-key styles. Use the exact header required
    # by Pond's Integration Guide if it differs.
    supplied = x_access_key
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()

    if POND_ACCESS_KEY and supplied != POND_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid access key")


@app.get("/")
def root():
    return {"status": "ok", "agent": "Pond Crypto AI Agent"}


@app.get("/manifest")
def manifest():
    return {
        "name": "Pond Crypto AI Agent",
        "description": "An AI agent that explains crypto projects, tokens, DeFi concepts and on-chain data supplied by the requester.",
        "version": "1.0.0",
        "actions": [
            {
                "name": "crypto_research",
                "description": "Analyze a crypto-related question and return a concise, neutral explanation.",
                "input": {
                    "type": "string",
                    "description": "The crypto question or research request."
                },
                "output": {
                    "type": "string",
                    "description": "The agent's research response."
                }
            }
        ],
        "capabilities": ["crypto research", "DeFi explanations", "token/project summaries"]
    }


def extract_prompt(body: Any) -> str:
    if isinstance(body, str):
        return body

    if isinstance(body, dict):
        for key in ("input", "prompt", "query", "message", "task"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Some APIs wrap the input one level deeper.
        for key in ("inputs", "arguments", "params"):
            value = body.get(key)
            if isinstance(value, dict):
                for inner in ("input", "prompt", "query", "message"):
                    if isinstance(value.get(inner), str) and value[inner].strip():
                        return value[inner].strip()

    return str(body)


@app.post("/runs")
async def runs(
    request: Request,
    authorization: str | None = Header(default=None),
    x_access_key: str | None = Header(default=None),
):
    check_auth(authorization, x_access_key)

    body = await request.json()
    prompt = extract_prompt(body)

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server."
        )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are a crypto research assistant. Explain concepts clearly and "
            "separate facts from uncertainty. Do not give personalized financial "
            "advice. Keep answers concise unless the user asks for detail."
        ),
        input=prompt,
    )

    result = response.output_text

    # This is the starter result shape. If Pond's Integration Guide specifies
    # a different response schema, adjust this object to match it exactly.
    return {
        "status": "completed",
        "result": result
    }


@app.get("/tasks/{task_id}")
def task(task_id: str):
    # The starter agent runs synchronously and returns HTTP 200 from /runs,
    # so Pond should not need this endpoint. It is included for compatibility.
    raise HTTPException(status_code=404, detail="No asynchronous task found")
