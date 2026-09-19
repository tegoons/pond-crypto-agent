import os
import re

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Pond Crypto AI Agent", version="1.0.0")

ACCESS_KEY = os.getenv("POND_ACCESS_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =========================
# Pond Protocol V1: Manifest
# =========================

@app.get("/manifest")
def manifest():
    return {
        "protocol": "marketplace-agent",
        "protocol_version": "1.0",
        "agent_version": "1.0.1",

        "metadata": {
            "name": "Crypto Research AI",
            "short_description": "AI agent for crypto, DeFi and blockchain research.",
            "description": "Explains crypto projects, DeFi concepts, tokens and blockchain topics.",
            "category": "research",
        },

        "actions": [
            {
                "id": "crypto_research",
                "name": "Crypto Research",
                "description": "Use for questions about cryptocurrency, DeFi, blockchain, tokens and crypto projects.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The crypto research question or request.",
                            "minLength": 1
                        }
                    },
                    "required": ["prompt"],
                    "additionalProperties": False
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

        "input_modes": [
            "text/plain"
        ],

        "output_modes": [
            "text/markdown"
        ],

        "limits": {
            "max_request_bytes": 1048576,
            "max_run_seconds": 60
        }
    }


# =========================
# Pond Run Request
# =========================

class RunRequest(BaseModel):
    run_id: str
    agent_id: str
    conversation_id: str
    history_truncated: bool
    action_id: str | None = None
    user: dict
    messages: list[dict]
    parameters: dict
    execution: dict


# =========================
# Pond Authentication
# =========================

def fail(status_code: int, code: str, message: str):
    raise HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message
        }
    )


def authenticate_pond(
    authorization: str | None = Header(default=None),
    pond_version: str | None = Header(
        default=None,
        alias="X-Agent-Protocol-Version"
    ),
):
    if not ACCESS_KEY:
        fail(
            500,
            "configuration_error",
            "POND_ACCESS_KEY is not configured."
        )

    if authorization != f"Bearer {ACCESS_KEY}":
        fail(
            401,
            "unauthorized",
            "The Access Key is missing or invalid."
        )

    if pond_version is None or re.fullmatch(r"\d+\.\d+", pond_version) is None:
        fail(
            400,
            "invalid_request",
            "The protocol version must be Major.Minor."
        )

    if pond_version != "1.0":
        fail(
            400,
            "unsupported_protocol_version",
            f"Protocol version {pond_version} is not supported."
        )


# =========================
# AI Agent
# =========================

def run_agent(prompt: str) -> str:

    if not OPENAI_API_KEY or client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are a crypto research assistant. "
            "Explain cryptocurrency, blockchain, DeFi, tokens and crypto projects clearly. "
            "Separate facts from uncertainty. "
            "Do not provide personalized financial advice. "
            "Do not guarantee profits. "
            "Answer concisely unless the user requests more detail."
        ),
        input=prompt,
    )

    return response.output_text


# =========================
# Pond /runs
# =========================

@app.post(
    "/runs",
    dependencies=[Depends(authenticate_pond)]
)
async def create_run(
    run: RunRequest,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key"
    ),
):

    if idempotency_key != run.run_id:
        fail(
            400,
            "invalid_request",
            "Idempotency-Key must match run_id."
        )

    if run.action_id != "crypto_research":
        fail(
            400,
            "unsupported_operation",
            "The requested action is not supported."
        )

    prompt = run.parameters.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        fail(
            400,
            "invalid_request",
            "A non-empty prompt is required."
        )

    try:
        result = run_agent(prompt)

        return {
            "run_id": run.run_id,
            "status": "completed",
            "output": [
                {
                    "type": "text",
                    "text": result
                }
            ],
            "usage": {
                "unit_of_measurement": "result",
                "quantity": 1
            }
        }

    except Exception as e:
        return {
            "run_id": run.run_id,
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


# =========================
# Pond Error Handling
# =========================

@app.exception_handler(HTTPException)
async def pond_error(_request: Request, error: HTTPException):
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": error.detail
        }
    )


@app.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request,
    _error: RequestValidationError
):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "invalid_request",
                "message": "The request does not match Pond Protocol V1."
            }
        }
    )


@app.get("/")
def root():
    return {
        "status": "ok",
        "agent": "Crypto Research AI"
    @app.get(
    "/tasks/{task_id}",
    dependencies=[Depends(authenticate_pond)]
)
def get_task(task_id: str):
    fail(
        404,
        "task_not_found",
        "The requested task was not found."
    )
