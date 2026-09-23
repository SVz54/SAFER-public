#!/usr/bin/env python3
from __future__ import annotations

"""
Run secure code generation from generation prompt records.

Usage:
    python generation/run_generation.py generation_prompts.jsonl generation_outputs.jsonl

Example:
    python generation/run_generation.py \
      generation/generation_prompts.jsonl \
      generation/generation_outputs.jsonl
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
FILESAFE_RE = re.compile(r"[^a-zA-Z0-9_.-]+")

GENERATION_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "assumptions": {"type": "string"},
        "generated_code": {"type": "string"},
        "security_rationale": {"type": "string"},
    },
    "required": ["assumptions", "generated_code", "security_rationale"],
    "additionalProperties": False,
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} in {path} must be a JSON object.")
            rows.append(obj)
    return rows


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_multiline(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def safe_filename(value: str) -> str:
    cleaned = FILESAFE_RE.sub("_", normalize_text(value))
    return cleaned.strip("._") or "record"


def validate_structured_generation(payload: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Structured generation payload is not an object.")

    expected = {"assumptions", "generated_code", "security_rationale"}
    keys = set(payload.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"Structured generation schema mismatch. missing={missing}, extra={extra}")

    assumptions = normalize_multiline(str(payload.get("assumptions", "")))
    generated_code = normalize_multiline(str(payload.get("generated_code", "")))
    security_rationale = normalize_multiline(str(payload.get("security_rationale", "")))

    if not generated_code:
        raise ValueError("Structured generation payload contains empty generated_code.")

    return {
        "assumptions": assumptions,
        "generated_code": generated_code,
        "security_rationale": security_rationale,
    }


def call_generation_with_retry(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    temperature: float,
) -> Tuple[str, Dict[str, str]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
                temperature=temperature,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "generation_output",
                        "schema": GENERATION_OUTPUT_SCHEMA,
                        "strict": True,
                    }
                },
            )
            output_text = getattr(response, "output_text", None)
            if output_text is None:
                raise ValueError("Model returned no output_text.")
            cleaned = str(output_text).strip()
            if not cleaned:
                raise ValueError("Model returned empty output_text.")
            parsed = json.loads(cleaned)
            structured = validate_structured_generation(parsed)
            return cleaned, structured
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            backoff = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.7))
            print(f"  retry {attempt}/{max_retries - 1} after error: {exc} (sleep {backoff:.1f}s)")
            time.sleep(backoff)

    assert last_error is not None
    raise last_error


def build_output_row(
    prompt_row: Dict[str, Any],
    *,
    model: str,
    generated_response_raw: str = "",
    assumptions: str = "",
    generated_code: str = "",
    security_rationale: str = "",
    code_sha256: str = "",
    error: str = "",
) -> Dict[str, Any]:
    return {
        "query_id": normalize_text(prompt_row.get("query_id", "")),
        "prompt": prompt_row.get("prompt", ""),
        "context_cwe_ids": prompt_row.get("context_cwe_ids", []),
        "num_context_items": prompt_row.get("num_context_items", 0),
        "model": model,
        "generated_response_raw": generated_response_raw,
        "assumptions": assumptions,
        "generated_code": generated_code,
        "security_rationale": security_rationale,
        "code_sha256": code_sha256,
        "error": error,
    }


def write_generation_artifacts(
    *,
    artifacts_dir: Path,
    query_id: str,
    generated_code: str,
    security_rationale: str,
    assumptions: str,
    code_extension: str,
) -> Dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(query_id)
    ext = code_extension if code_extension.startswith(".") else f".{code_extension}"

    code_path = artifacts_dir / f"{stem}.code{ext}"
    rationale_path = artifacts_dir / f"{stem}.rationale.txt"
    assumptions_path = artifacts_dir / f"{stem}.assumptions.txt"

    code_path.write_text(generated_code + ("\n" if generated_code and not generated_code.endswith("\n") else ""), encoding="utf-8")
    rationale_path.write_text(security_rationale + ("\n" if security_rationale and not security_rationale.endswith("\n") else ""), encoding="utf-8")
    assumptions_path.write_text(assumptions + ("\n" if assumptions and not assumptions.endswith("\n") else ""), encoding="utf-8")

    return {
        "code_path": str(code_path),
        "rationale_path": str(rationale_path),
        "assumptions_path": str(assumptions_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run secure code generation from generation prompt records.")
    parser.add_argument("generation_prompts_jsonl", type=Path, help="Input prompts JSONL")
    parser.add_argument("output_jsonl", type=Path, help="Output generated responses JSONL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--delay", type=float, default=0.4, help="Delay between API requests (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--start-index", type=int, default=1, help="1-based start index")
    parser.add_argument("--max-records", type=int, default=0, help="Max number of records to process (0 = all)")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Optional directory to write code/rationale files")
    parser.add_argument("--code-extension", type=str, default="txt", help="Code artifact extension, e.g. py, js, java")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.temperature < 0.0 or args.temperature > 2.0:
        raise ValueError("--temperature must be between 0.0 and 2.0")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    prompt_rows = load_jsonl(args.generation_prompts_jsonl)
    total = len(prompt_rows)
    if total == 0:
        print("No prompt rows found in input.")
        dump_jsonl(args.output_jsonl, [])
        return

    start = max(1, args.start_index)
    end = total if args.max_records <= 0 else min(total, start - 1 + args.max_records)

    client = OpenAI(api_key=api_key)
    output_rows: List[Dict[str, Any]] = []
    success_count = 0
    failed_count = 0

    for idx, prompt_row in enumerate(prompt_rows, start=1):
        if idx < start or idx > end:
            output_rows.append(
                build_output_row(
                    prompt_row,
                    model=args.model,
                    generated_response_raw="",
                    error="skipped",
                )
            )
            continue

        query_id = normalize_text(prompt_row.get("query_id", "")) or f"row-{idx}"
        system_prompt = str(prompt_row.get("system_prompt", "")).strip()
        user_prompt = str(prompt_row.get("user_prompt", "")).strip()

        print(f"[{idx}/{total}] generating for {query_id}")

        if not system_prompt or not user_prompt:
            failed_count += 1
            output_rows.append(
                build_output_row(
                    prompt_row,
                    model=args.model,
                    generated_response_raw="",
                    error="Missing system_prompt or user_prompt in input row.",
                )
            )
            print(f"  failed: {query_id}: missing prompt fields")
            continue

        try:
            generated_raw, structured = call_generation_with_retry(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_retries=max(1, args.max_retries),
                temperature=args.temperature,
            )

            code_sha256 = hashlib.sha256(structured["generated_code"].encode("utf-8")).hexdigest()
            output_row = build_output_row(
                prompt_row,
                model=args.model,
                generated_response_raw=generated_raw,
                assumptions=structured["assumptions"],
                generated_code=structured["generated_code"],
                security_rationale=structured["security_rationale"],
                code_sha256=code_sha256,
                error="",
            )

            if args.artifacts_dir is not None:
                artifact_paths = write_generation_artifacts(
                    artifacts_dir=args.artifacts_dir,
                    query_id=query_id,
                    generated_code=structured["generated_code"],
                    security_rationale=structured["security_rationale"],
                    assumptions=structured["assumptions"],
                    code_extension=args.code_extension,
                )
                output_row.update(artifact_paths)

            output_rows.append(
                output_row
            )
            success_count += 1
            print(f"  ok: {query_id}")
        except Exception as exc:
            failed_count += 1
            output_rows.append(
                build_output_row(
                    prompt_row,
                    model=args.model,
                    generated_response_raw="",
                    error=str(exc),
                )
            )
            print(f"  failed: {query_id}: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)

    dump_jsonl(args.output_jsonl, output_rows)
    print(
        f"Done. total={total} processed={max(0, end - start + 1)} "
        f"success={success_count} failed={failed_count} output={args.output_jsonl}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
