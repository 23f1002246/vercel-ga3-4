import os, json, httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
AIPIPE_URL   = "https://aipipe.org/openai/v1/chat/completions"

class ExtractRequest(BaseModel):
    text: str
    schema: dict[str, str]

SYSTEM_PROMPT = """You are a structured data extraction assistant.
You will be given raw text and a schema describing fields to extract with their types.

Supported types and their rules:
- string: return as a JSON string
- integer: return as a JSON integer (no quotes)
- float: return as a JSON number with decimals (no quotes)
- boolean: return as JSON true or false (no quotes)
- date: return as ISO format string "YYYY-MM-DD"
- array[string]: return as a JSON array of strings
- array[integer]: return as a JSON array of integers

Rules:
- Return ONLY a JSON object with exactly the keys in the schema — no extra keys.
- Use null (not "null") for any field that cannot be found in the text.
- Dates: convert any date format to YYYY-MM-DD (e.g. "3rd March 2026" -> "2026-03-03").
- Floats/integers must be JSON numbers, NOT strings.
- Do not include units, currency symbols, or extra text in values.
- Return ONLY the JSON object. No explanation, no markdown."""

def coerce_value(value: Any, type_str: str) -> Any:
    """Post-process LLM output to ensure correct types."""
    if value is None:
        return None
    try:
        if type_str == "integer":
            return int(float(str(value).replace(",", "")))
        elif type_str == "float":
            return float(str(value).replace(",", ""))
        elif type_str == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "yes", "1")
        elif type_str == "date":
            return str(value)  # LLM should already return YYYY-MM-DD
        elif type_str == "string":
            return str(value)
        elif type_str == "array[string]":
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]
        elif type_str == "array[integer]":
            if isinstance(value, list):
                return [int(v) for v in value]
            return [int(value)]
    except Exception:
        return None
    return value

@app.post("/dynamic-extract")
async def dynamic_extract(body: ExtractRequest):
    schema_desc = "\n".join(f'  "{k}": {v}' for k, v in body.schema.items())
    user_msg = f"""Extract fields from this text according to the schema below.

TEXT:
{body.text}

SCHEMA (field: type):
{{{schema_desc}
}}

Return a JSON object with exactly these keys: {list(body.schema.keys())}"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 500,
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            AIPIPE_URL,
            json=payload,
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"}
        )
        r.raise_for_status()
        raw = json.loads(r.json()["choices"][0]["message"]["content"])

    # Enforce schema: exactly the requested keys, correct types, no extras
    result = {}
    for key, type_str in body.schema.items():
        value = raw.get(key, None)
        result[key] = coerce_value(value, type_str)

    return result

@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /dynamic-extract"}
