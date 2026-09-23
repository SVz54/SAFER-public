#!/usr/bin/env python3
from __future__ import annotations

"""
Build generation prompts from generation context rows.

Usage:
    python generation/build_generation_prompts.py generation_context.jsonl generation_prompts.jsonl

Example:
    python generation/build_generation_prompts.py \
      generation/generation_context.jsonl \
      generation/generation_prompts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


SYSTEM_PROMPT = """You are a secure software engineering assistant.

Your task is to generate implementation-ready code that is secure by design.
Use the provided security context as hard guidance.

Rules:
1. Do not ignore the listed CWE risks.
2. Apply concrete mitigations in the code and design.
3. Avoid insecure shortcuts, placeholder security, or unsafe defaults.
4. Keep the solution practical and runnable for developers.
5. If assumptions are necessary, state them briefly before the code.
"""


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def compact_string_list(values: Any, max_items: int) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = normalize_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


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


def format_context_block(
    retrieved_context: List[Dict[str, Any]],
    max_guidance: int,
    max_consequences: int,
    max_examples: int,
    include_task_tags: bool,
) -> str:
    if not retrieved_context:
        return "No CWE context retrieved."

    blocks: List[str] = []
    for item in retrieved_context:
        rank = item.get("rank", "")
        cwe_id = normalize_text(item.get("cwe_id", ""))
        name = normalize_text(item.get("name", ""))
        weakness_summary = normalize_text(item.get("weakness_summary", ""))
        developer_risk_summary = normalize_text(item.get("developer_risk_summary", ""))
        guidance = compact_string_list(item.get("secure_coding_guidance"), max_guidance)
        consequences = compact_string_list(item.get("consequence_keywords"), max_consequences)
        examples = compact_string_list(item.get("example_signals"), max_examples)
        task_tags = compact_string_list(item.get("task_tags"), 12) if include_task_tags else []

        lines = [
            f"[CWE Rank {rank}] {cwe_id} - {name}",
            f"Weakness summary: {weakness_summary}" if weakness_summary else "Weakness summary: (none)",
            f"Developer risk summary: {developer_risk_summary}" if developer_risk_summary else "Developer risk summary: (none)",
            "Secure coding guidance:",
        ]

        if guidance:
            lines.extend([f"- {g}" for g in guidance])
        else:
            lines.append("- (none)")

        lines.append("Consequence keywords: " + (", ".join(consequences) if consequences else "(none)"))
        lines.append("Example signals: " + (" | ".join(examples) if examples else "(none)"))

        if include_task_tags:
            lines.append("Task tags: " + (", ".join(task_tags) if task_tags else "(none)"))

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_user_prompt(
    *,
    prompt: str,
    retrieved_context: List[Dict[str, Any]],
    max_guidance: int,
    max_consequences: int,
    max_examples: int,
    include_task_tags: bool,
) -> str:
    context_block = format_context_block(
        retrieved_context,
        max_guidance=max_guidance,
        max_consequences=max_consequences,
        max_examples=max_examples,
        include_task_tags=include_task_tags,
    )

    return (
        "Developer task:\n"
        f"{prompt}\n\n"
        "Security context (retrieved CWE guidance):\n"
        f"{context_block}\n\n"
        "Instruction:\n"
        "Generate secure-by-design code for the developer task, explicitly applying the above CWE guidance.\n"
        "Return:\n"
        "1) Brief assumptions (if any)\n"
        "2) The code\n"
        "3) A short security rationale tied to the listed CWE context"
    )


def build_prompt_record(
    *,
    row: Dict[str, Any],
    max_guidance: int,
    max_consequences: int,
    max_examples: int,
    include_task_tags: bool,
) -> Dict[str, Any]:
    query_id = normalize_text(row.get("query_id", ""))
    prompt = normalize_text(row.get("prompt", ""))
    retrieved_context = row.get("retrieved_context", [])
    if not isinstance(retrieved_context, list):
        retrieved_context = []

    context_cwe_ids: List[str] = []
    seen = set()
    for item in retrieved_context:
        if not isinstance(item, dict):
            continue
        cwe_id = normalize_text(item.get("cwe_id", ""))
        if not cwe_id:
            continue
        key = cwe_id.lower()
        if key in seen:
            continue
        seen.add(key)
        context_cwe_ids.append(cwe_id)

    user_prompt = build_user_prompt(
        prompt=prompt,
        retrieved_context=retrieved_context,
        max_guidance=max_guidance,
        max_consequences=max_consequences,
        max_examples=max_examples,
        include_task_tags=include_task_tags,
    )

    return {
        "query_id": query_id,
        "prompt": prompt,
        "context_cwe_ids": context_cwe_ids,
        "num_context_items": len(retrieved_context),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready generation prompts from generation context.")
    parser.add_argument("generation_context_jsonl", type=Path, help="Input generation context JSONL")
    parser.add_argument("output_jsonl", type=Path, help="Output JSONL with generation prompts")
    parser.add_argument("--max-guidance", type=int, default=3, help="Max guidance lines per CWE in prompt context")
    parser.add_argument("--max-consequences", type=int, default=6, help="Max consequence keywords per CWE in prompt context")
    parser.add_argument("--max-examples", type=int, default=2, help="Max example signals per CWE in prompt context")
    parser.add_argument("--include-task-tags", action="store_true", help="Include task tags in prompt context")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = load_jsonl(args.generation_context_jsonl)
    prompt_rows = [
        build_prompt_record(
            row=row,
            max_guidance=max(1, args.max_guidance),
            max_consequences=max(1, args.max_consequences),
            max_examples=max(1, args.max_examples),
            include_task_tags=args.include_task_tags,
        )
        for row in rows
    ]

    dump_jsonl(args.output_jsonl, prompt_rows)
    print(f"Input context rows: {len(rows)}")
    print(f"Wrote generation prompts: {args.output_jsonl}")


if __name__ == "__main__":
    main()

