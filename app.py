import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

ACCESS_KEY = os.getenv("POND_ACCESS_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)


class RunRequest(BaseModel):
    run_id: str
    agent_id: str
    conversation_id: str
    history_truncated: bool
    user: dict
    messages: list
    parameters: dict
    execution: dict
    action_id: str | None = None


def check_auth(
    authorization: str | None = Header(default=None),
    protocol_version: str | None = Header(
        default=None,
        alias="X-Agent-Protocol-Version"
    ),
):
    if authorization != f"Bearer {ACCESS_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if protocol_version != "1.0":
        raise HTTPException(
            status_code=400,
            detail="Unsupported protocol version"
        )


@app.get("/")
def home():
    return {
        "status": "ok",
        "agent": "Crypto Research AI"
    }


@app.get("/manifest")
def manifest():
    return {
        "protocol": "marketplace-agent",
        "protocol_version": "1.0",
        "agent_version": "1.0.0",
        "metadata": {
            "name": "Crypto Research AI",
            "short_description": "AI agent for crypto research.",
            "description": "Researches cryptocurrency, DeFi, blockchain and tokens.",
            "category": "research"
        },
        "actions": [
            {
                "id": "crypto_research",
                "name": "Crypto Research",
                "description": "Research cryptocurrency, DeFi, blockchain and tokens.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        ],
        "capabilities": {
            "sync": True,
            "streaming": False,
            "async_tasks": False,
            "cancellation": False,
            "attachments": False,
            "feedback": False
        },
        "input_modes": ["text/plain"],
        "output_modes": ["text/markdown"]
    }

@app.get("/tasks/{task_id}", dependencies=[Depends(check_auth)])
def get_task(task_id: str):
    raise HTTPException(status_code=404, detail="Task not found")
@app.post("/runs")
def run_agent(
    request: RunRequest,
    authorization: str | None = Header(default=None),
    protocol_version: str | None = Header(
        default=None,
        alias="X-Agent-Protocol-Version"
    ),
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key"
    ),
):
    check_auth(authorization, protocol_version)

    if idempotency_key != request.run_id:
        raise HTTPException(
            status_code=400,
            detail="Idempotency-Key must match run_id"
        )

    if request.action_id != "crypto_research":
        raise HTTPException(
            status_code=400,
            detail="Unsupported action"
        )

    prompt = request.parameters.get("prompt")

    if not prompt:
        raise HTTPException(
            status_code=400,
            detail="Prompt is required"
        )

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            instructions=(
                "You are a crypto research assistant. "
                "Explain crypto, DeFi, blockchain and tokens clearly. "
                "Separate facts from uncertainty. "
                "Do not guarantee profits."
            ),
            input=prompt
        )

        return {
            "run_id": request.run_id,
            "status": "completed",
            "output": [
                {
                    "type": "text",
                    "text": response.output_text
                }
            ],
            "usage": {
                "unit_of_measurement": "result",
                "quantity": 1
            }
        }

    except Exception as e:
        return {
            "run_id": request.run_id,
            "status": "failed",
            "error": {
                "code": "internal_error",
                "message": str(e)
            },
            "usage": {
                "unit_of_measurement": "result",
                "quantity": 0
            }
        }
