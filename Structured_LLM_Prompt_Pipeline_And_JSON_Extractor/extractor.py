import argparse
import json
import sys
from pathlib import Path

import httpx
from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from loaders import read_document
from prompts import build_messages
from schemas import Invoice, Resume

DEFAULT_MODELS = {"ollama": "qwen2.5:7b", "openai": "gpt-4o-mini"}
DEFAULT_MODEL = DEFAULT_MODELS["ollama"]   # run_samples.py imports this
MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 1500    # a valid result is far shorter; stops endless generation
SCHEMAS = {"invoice": Invoice, "resume": Resume}

ollama_client = Client(timeout=240)   # give up on one local model call after 4 minutes


def call_ollama(model: str, messages: list[dict], schema) -> str:
    try:
        response = ollama_client.chat(
            model=model,
            messages=messages,
            format=schema.model_json_schema(),   # forces the JSON structure
            options={"temperature": 0, "num_predict": MAX_OUTPUT_TOKENS},
        )
    except httpx.TimeoutException:
        raise RuntimeError("The model took too long to answer (timeout). Try a smaller model or a shorter document.")
    except ConnectionError:
        raise RuntimeError("Cannot reach Ollama. Install it from ollama.com and make sure it is running.")
    except ResponseError as e:
        raise RuntimeError(f"Ollama error (is the model '{model}' downloaded? try: ollama pull {model}): {e}")
    return response.message.content


def call_openai(model: str, messages: list[dict], schema) -> str:
    """Any OpenAI-compatible API. Reads OPENAI_API_KEY (and optionally OPENAI_BASE_URL) from the environment."""
    try:
        from openai import APITimeoutError, OpenAI, OpenAIError
    except ImportError:
        raise RuntimeError("The openai package is missing. Run: pip install openai")

    # JSON mode guarantees valid JSON; we put the schema in the prompt so the shape matches
    schema_hint = {
        "role": "system",
        "content": "Return a JSON object that follows this JSON Schema exactly:\n"
                   + json.dumps(schema.model_json_schema()),
    }
    full_messages = [messages[0], schema_hint] + messages[1:]
    try:
        client = OpenAI(timeout=60)
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            response_format={"type": "json_object"},
        )
    except APITimeoutError:
        raise RuntimeError("The API took too long to answer (timeout).")
    except OpenAIError as e:
        raise RuntimeError(f"API error: {e}")
    return response.choices[0].message.content


PROVIDERS = {"ollama": call_ollama, "openai": call_openai}


def summarize_errors(error: ValidationError) -> str:
    """Short human-readable list of what was wrong."""
    parts = []
    for e in error.errors():
        where = ".".join(str(p) for p in e["loc"])
        parts.append(f"{where}: {e['msg']}" if where else e["msg"])
    return "; ".join(parts)


def extract(text: str, doc_type: str, model: str | None = None, provider: str = "ollama",
            max_retries: int = MAX_RETRIES) -> BaseModel:
    """Turn unstructured text into a validated Pydantic object."""
    schema = SCHEMAS[doc_type]
    model = model or DEFAULT_MODELS[provider]
    call_model = PROVIDERS[provider]
    messages = build_messages(doc_type, text)
    last_reason = ""

    for attempt in range(1, max_retries + 2):
        raw = call_model(model, messages, schema)
        try:
            # context lets the validators compare the answer with the original text
            return schema.model_validate_json(raw, context={"source_text": text})
        except ValidationError as error:
            last_reason = summarize_errors(error)
            print(f"  Attempt {attempt} failed: {last_reason[:300]}", file=sys.stderr)
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Your JSON failed validation:\n{error}\n"
                           "Fix the problems and return the corrected JSON only.",
            })

    raise RuntimeError(f"No valid output after {max_retries + 1} attempts. Last problem: {last_reason[:300]}")


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from unstructured documents.")
    parser.add_argument("--file", required=True, help="Path to a .txt, .pdf or .docx file")
    parser.add_argument("--type", choices=SCHEMAS.keys(), default="invoice", help="Document type")
    parser.add_argument("--provider", choices=PROVIDERS.keys(), default="ollama", help="Where the model runs")
    parser.add_argument("--model", help="Model name (default depends on the provider)")
    parser.add_argument("--output", help="Optional path to save the JSON result")
    args = parser.parse_args()

    try:
        text = read_document(args.file)
        result = extract(text, args.type, model=args.model, provider=args.provider)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    json_output = result.model_dump_json(indent=2)
    print(json_output)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json_output, encoding="utf-8")


if __name__ == "__main__":
    main()