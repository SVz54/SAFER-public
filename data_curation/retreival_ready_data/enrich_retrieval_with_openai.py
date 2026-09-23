#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

RETRIEVAL_FIELDS = [
    "weakness_summary",
    "developer_risk_summary",
    "retrieval_keywords",
    "task_tags",
    "prompt_patterns",
    "secure_coding_guidance",
    "consequence_keywords",
    "example_signals",
    "retrieval_text",
]

SOURCE_FIELDS = [
    "cwe_id",
    "name",
    "abstraction",
    "status",
    "description_clean",
    "extended_description_clean",
    "alternate_terms",
    "child_of",
    "applicable_platforms",
    "likelihood_of_exploit",
    "demonstrative_examples",
    "observed_examples_selected",
    "potential_mitigations_structured",
    "common_consequences_structured",
]

LIST_RETRIEVAL_FIELDS = {
    "retrieval_keywords",
    "task_tags",
    "prompt_patterns",
    "secure_coding_guidance",
    "consequence_keywords",
    "example_signals",
}

SOURCE_ALIASES = {
    "description_clean": ["description_clean", "description"],
    "extended_description_clean": ["extended_description_clean", "extended_description"],
    "potential_mitigations_structured": ["potential_mitigations_structured", "potential_mitigations"],
    "common_consequences_structured": ["common_consequences_structured", "common_consequences"],
}

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "weakness_summary": {"type": "string"},
        "developer_risk_summary": {"type": "string"},
        "retrieval_keywords": {"type": "array", "items": {"type": "string"}},
        "task_tags": {"type": "array", "items": {"type": "string"}},
        "prompt_patterns": {"type": "array", "items": {"type": "string"}},
        "secure_coding_guidance": {"type": "array", "items": {"type": "string"}},
        "consequence_keywords": {"type": "array", "items": {"type": "string"}},
        "example_signals": {"type": "array", "items": {"type": "string"}},
        "retrieval_text": {"type": "string"},
    },
    "required": RETRIEVAL_FIELDS,
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are Stage 2 enrichment for a retrieval dataset used in a RAG pipeline for secure code generation.

Your job is to improve only the retrieval-facing fields for one CWE record so that the record is easier to retrieve when a user asks for code to build a software feature.

Core retrieval objective:
The end user usually asks for functionality, not for vulnerabilities.
They write prompts such as:
- create a login page
- build a file upload API
- parse XML input
- write a C function to process network packets
- create a profile update form
- build a password reset flow

Therefore, your output must help the retriever connect:
software-building prompt -> implied security risk -> relevant CWE guidance

Important behavior rules:
- Do not write like a security textbook or vulnerability database.
- Do not write like a security Q&A site.
- Prefer normal developer task language over taxonomy language.
- Prompt patterns must sound like realistic user requests for generating code, features, endpoints, parsers, handlers, forms, or services.
- Do not generate prompts such as:
  "How to prevent X"
  "Detect Y vulnerability"
  "Best practices for Z weakness"
unless the CWE truly maps to a user asking for a security-specific implementation task.
- retrieval_keywords must include both:
  1. weakness/security terms
  2. software feature and implementation terms likely to appear in user prompts
- task_tags must be short normalized engineering labels.
- secure_coding_guidance must be concise, actionable, and suitable to pass into generation.
- consequence_keywords must be short normalized outcomes, not long phrases.
- example_signals must be short implementation-level warning signs.

Grounding rules:
- Stay grounded in the provided source record.
- Do not hallucinate technologies, platforms, impacts, mitigations, or app contexts not supported by the record.
- Do not change source-derived fields.
- If a likely use case is not clearly supported by the source record, leave it out.

Output rules:
- Return JSON only.
- Match the schema exactly.
- Keep text concise but information-dense.
- retrieval_text must be a dense retrieval paragraph optimized for semantic search over code-generation prompts.
""".strip()

USER_PROMPT_TEMPLATE = """
Enrich retrieval fields for this CWE record.

Source-derived fields:
{source_payload}

Current baseline retrieval fields:
{baseline_payload}

When generating the retrieval fields, optimize for this use case:
A user asks an AI system to generate code for a software feature.
The user usually describes the feature, input, API, form, parser, upload flow, auth flow, handler, or service.
The user usually does NOT mention the CWE name or vulnerability explicitly.

Generate retrieval fields that make this CWE retrievable from realistic software-building prompts.

Additional field instructions:
- weakness_summary: 1 to 3 sentences explaining the weakness clearly.
- developer_risk_summary: explain when this weakness becomes relevant during implementation of real software features.
- retrieval_keywords: include both security terms and implementation / feature terms.
- task_tags: short normalized tags only.
- prompt_patterns: must sound like realistic code-generation prompts from developers, not security-study questions.
- secure_coding_guidance: concise implementation guidance that can be passed to a code generator.
- consequence_keywords: short normalized risk outcomes.
- example_signals: short implementation-level indicators drawn from examples or mitigations.
- retrieval_text: one dense paragraph connecting the weakness, likely developer prompt contexts, risks, and safe implementation guidance.

Return exactly these keys:
- weakness_summary
- developer_risk_summary
- retrieval_keywords
- task_tags
- prompt_patterns
- secure_coding_guidance
- consequence_keywords
- example_signals
- retrieval_text
""".strip()

_TEXT_LIMITS = {
    "weakness_summary": 700,
    "developer_risk_summary": 900,
    "retrieval_text": 4500,
}

_LIST_LIMITS = {
    "retrieval_keywords": (40, 120, False),
    "task_tags": (20, 80, True),
    "prompt_patterns": (20, 220, False),
    "secure_coding_guidance": (12, 320, False),
    "consequence_keywords": (20, 80, True),
    "example_signals": (20, 220, False),
}


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _trim_text(text: str, max_len: int) -> str:
    text = _normalize_ws(text)
    if len(text) <= max_len:
        return text
    clipped = text[:max_len].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:") + "..."


def _dedupe_str_list(values: List[str], lower: bool, max_items: int, max_item_len: int) -> List[str]:
    out: List[str] = []
    seen = set()

    for raw in values:
        cleaned = _normalize_ws(raw)
        if not cleaned:
            continue
        if len(cleaned) > max_item_len:
            cleaned = _trim_text(cleaned, max_item_len)

        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append(cleaned.lower() if lower else cleaned)
        if len(out) >= max_items:
            break

    return out


def _sanitize_for_prompt(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "<truncated>"
    if isinstance(value, str):
        return _trim_text(value, 1200)
    if isinstance(value, list):
        return [_sanitize_for_prompt(v, depth + 1) for v in value[:20]]
    if isinstance(value, dict):
        items = list(value.items())[:40]
        return {k: _sanitize_for_prompt(v, depth + 1) for k, v in items}
    return value


def _coerce_record_list(records: List[Any], source_label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{source_label} item {idx}: expected object, got {type(item).__name__}")
        out.append(item)
    return out


def load_records(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.lstrip()
    if not stripped:
        return []

    if stripped.startswith("["):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON array in {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected JSON array at top-level")
        return _coerce_record_list(payload, str(path))

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path} line {line_no}: expected JSON object, got {type(obj).__name__}")
        records.append(obj)
    return records


def write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pick_fields(record: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return {k: record.get(k) for k in keys}


def _first_non_none(record: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def build_source_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = pick_fields(record, SOURCE_FIELDS)
    for target_key, aliases in SOURCE_ALIASES.items():
        payload[target_key] = _first_non_none(record, aliases, payload.get(target_key))

    for key in SOURCE_FIELDS:
        value = payload.get(key)
        if key in {
            "alternate_terms",
            "child_of",
            "applicable_platforms",
            "demonstrative_examples",
            "observed_examples_selected",
            "potential_mitigations_structured",
            "common_consequences_structured",
        }:
            payload[key] = value if isinstance(value, list) else []
        else:
            payload[key] = _normalize_ws(str(value)) if value is not None else ""

    return payload


def build_baseline_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in RETRIEVAL_FIELDS:
        value = record.get(key)
        if key in LIST_RETRIEVAL_FIELDS:
            if isinstance(value, list):
                payload[key] = [str(v) for v in value if isinstance(v, str)]
            else:
                payload[key] = []
        else:
            payload[key] = _normalize_ws(str(value)) if value is not None else ""
    return payload


def validate_enrichment_shape(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Enrichment output is not a JSON object")

    payload_keys = set(payload.keys())
    expected_keys = set(RETRIEVAL_FIELDS)
    if payload_keys != expected_keys:
        missing = sorted(expected_keys - payload_keys)
        extra = sorted(payload_keys - expected_keys)
        raise ValueError(f"Schema mismatch. missing={missing}, extra={extra}")

    if not isinstance(payload["weakness_summary"], str):
        raise ValueError("weakness_summary must be string")
    if not isinstance(payload["developer_risk_summary"], str):
        raise ValueError("developer_risk_summary must be string")
    if not isinstance(payload["retrieval_text"], str):
        raise ValueError("retrieval_text must be string")

    for key in [
        "retrieval_keywords",
        "task_tags",
        "prompt_patterns",
        "secure_coding_guidance",
        "consequence_keywords",
        "example_signals",
    ]:
        value = payload[key]
        if not isinstance(value, list):
            raise ValueError(f"{key} must be array")
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{key} must contain only strings")


def normalize_enrichment(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload["weakness_summary"] = _trim_text(payload["weakness_summary"], _TEXT_LIMITS["weakness_summary"])
    payload["developer_risk_summary"] = _trim_text(
        payload["developer_risk_summary"],
        _TEXT_LIMITS["developer_risk_summary"],
    )
    payload["retrieval_text"] = _trim_text(payload["retrieval_text"], _TEXT_LIMITS["retrieval_text"])

    for key, (max_items, max_item_len, lower) in _LIST_LIMITS.items():
        payload[key] = _dedupe_str_list(payload[key], lower=lower, max_items=max_items, max_item_len=max_item_len)

    return payload


def build_user_prompt(record: Dict[str, Any]) -> str:
    source_payload = build_source_payload(record)
    baseline_payload = build_baseline_payload(record)

    source_payload = _sanitize_for_prompt(source_payload)
    baseline_payload = _sanitize_for_prompt(baseline_payload)

    return USER_PROMPT_TEMPLATE.format(
        source_payload=json.dumps(source_payload, ensure_ascii=False, indent=2),
        baseline_payload=json.dumps(baseline_payload, ensure_ascii=False, indent=2),
    )


def call_openai_with_retry(
    client: OpenAI,
    *,
    model: str,
    user_prompt: str,
    max_retries: int,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=user_prompt,
                temperature=0.2,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "retrieval_enrichment",
                        "schema": RESPONSE_SCHEMA,
                        "strict": True,
                    }
                },
            )

            if not getattr(response, "output_text", None):
                raise ValueError("Empty response.output_text from OpenAI")

            payload = json.loads(response.output_text)
            validate_enrichment_shape(payload)
            return normalize_enrichment(payload)

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            backoff = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 0.6))
            print(f"  retry {attempt}/{max_retries - 1} after error: {exc} (sleep {backoff:.1f}s)")
            time.sleep(backoff)

    assert last_error is not None
    raise last_error


def enrich_record(
    client: OpenAI,
    record: Dict[str, Any],
    *,
    model: str,
    max_retries: int,
) -> Tuple[Dict[str, Any], Optional[str]]:
    prompt = build_user_prompt(record)
    enriched = call_openai_with_retry(
        client,
        model=model,
        user_prompt=prompt,
        max_retries=max_retries,
    )

    merged = dict(record)
    for key in RETRIEVAL_FIELDS:
        merged[key] = enriched[key]
    return merged, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich retrieval records using OpenAI Responses API (supports JSONL and JSON array input)."
    )
    parser.add_argument("input_file", type=Path, help="Path to input JSONL or JSON array file")
    parser.add_argument("output_file", type=Path, help="Path to output JSONL or JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--delay", type=float, default=0.25, help="Delay between records (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max API retries per record")
    parser.add_argument("--start-index", type=int, default=1, help="1-based start index for processing")
    parser.add_argument("--max-records", type=int, default=0, help="Process at most N records (0 = all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    records = load_records(args.input_file)
    total = len(records)
    if total == 0:
        print("No records found in input file")
        write_records(args.output_file, [])
        return

    start = max(1, args.start_index)
    end = total if args.max_records <= 0 else min(total, start - 1 + args.max_records)

    client = OpenAI(api_key=api_key)

    out_records: List[Dict[str, Any]] = []
    success = 0
    failed = 0

    for idx, record in enumerate(records, start=1):
        if idx < start or idx > end:
            out_records.append(record)
            continue

        cwe_id = str(record.get("cwe_id") or f"record-{idx}")
        print(f"[{idx}/{total}] enriching {cwe_id}")

        try:
            enriched, _ = enrich_record(
                client,
                record,
                model=args.model,
                max_retries=max(1, args.max_retries),
            )
            out_records.append(enriched)
            success += 1
            print(f"  ok: {cwe_id}")
        except Exception as exc:
            out_records.append(record)
            failed += 1
            print(f"  failed: {cwe_id}: {exc}")

        if args.delay > 0:
            time.sleep(args.delay)

    write_records(args.output_file, out_records)
    print(
        f"Done. wrote={len(out_records)} success={success} failed={failed} "
        f"output={args.output_file}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)
