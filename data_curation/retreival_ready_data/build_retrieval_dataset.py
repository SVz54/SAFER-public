#!/usr/bin/env python3
"""
Build retrieval-ready CWE JSONL records from either:
1) raw MITRE CWE XML, or
2) the reduced cleaned XML produced by clean_cwe_xml.py

Output:
- JSONL file where each line is one retrieval-ready record
- Designed for embedding / hybrid retrieval / later generation filtering

Usage:
    python build_retrieval_dataset.py input.xml output.jsonl

Example:
    python build_retrieval_dataset.py cleaned_1435.xml retrieval_ready.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-\d+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-/.]{2,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ----------------------------
# Generic XML helpers
# ----------------------------

def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def children_by_name(elem: Optional[ET.Element], name: str) -> List[ET.Element]:
    if elem is None:
        return []
    return [child for child in list(elem) if local_name(child.tag) == name]


def first_child(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for child in list(elem):
        if local_name(child.tag) == name:
            return child
    return None


def normalize_space(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def iter_text_with_breaks(elem: ET.Element) -> Iterable[str]:
    block_tags = {"p", "div", "ul", "ol", "li", "br"}

    if elem.text:
        yield elem.text

    for child in list(elem):
        tag = local_name(child.tag)
        if tag in block_tags:
            yield "\n"

        yield from iter_text_with_breaks(child)

        if tag in block_tags:
            yield "\n"

        if child.tail:
            yield child.tail


def flatten_xml_text(elem: Optional[ET.Element]) -> str:
    if elem is None:
        return ""

    raw = "".join(iter_text_with_breaks(elem)).replace("\xa0", " ")
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    return raw.strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_space(text)
    if not text:
        return []
    return [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]


def first_n_sentences(text: str, n: int = 2) -> str:
    sents = split_sentences(text)
    return " ".join(sents[:n]).strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        item = normalize_space(item)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


# ----------------------------
# Scenario / keyword rules
# ----------------------------

TASK_RULES: List[Tuple[str, List[str], List[str], List[str]]] = [
    (
        "authentication",
        ["login", "signup", "sign up", "password", "credential", "session", "authentication", "account", "token"],
        [
            "create a login page",
            "build a signup API",
            "implement password reset",
            "create an authentication endpoint",
            "build session handling for logged-in users"
        ],
        ["authentication", "session-management", "credential-handling", "input-validation"]
    ),
    (
        "file-upload",
        ["upload", "multipart", "file", "filename", "attachment", "image upload", "document upload"],
        [
            "build a file upload endpoint",
            "create image upload functionality",
            "implement document upload API",
            "accept attachments from users",
            "save uploaded files to the server"
        ],
        ["file-upload", "input-validation", "filesystem-access"]
    ),
    (
        "file-download",
        ["download", "export", "read file", "open file", "serve file", "attachment download"],
        [
            "build a file download endpoint",
            "serve files from the server",
            "open a file by path",
            "export documents for users"
        ],
        ["file-download", "filesystem-access", "path-handling"]
    ),
    (
        "xml-processing",
        ["xml", "soap", "dtd", "entity", "parser", "parse xml", "xml input"],
        [
            "parse XML input from users",
            "build an XML import feature",
            "process SOAP/XML requests",
            "upload and parse XML documents"
        ],
        ["xml-processing", "parser-safety", "input-validation"]
    ),
    (
        "json-processing",
        ["json", "request body", "payload", "deserialize", "deserialization", "parse request"],
        [
            "parse JSON request bodies",
            "build an API that accepts JSON payloads",
            "deserialize untrusted input",
            "process incoming webhook payloads"
        ],
        ["json-processing", "request-handling", "input-validation", "deserialization"]
    ),
    (
        "path-handling",
        ["path", "directory", "file path", "filepath", "traversal", "../", "storage path", "archive extraction"],
        [
            "open a file using a user-supplied path",
            "extract uploaded archives",
            "read files from a requested path",
            "build a document import feature"
        ],
        ["path-handling", "filesystem-access", "input-validation"]
    ),
    (
        "user-content-rendering",
        ["html", "browser", "render", "display", "comment", "profile", "rich text", "markdown", "homepage", "echo"],
        [
            "build a comment system",
            "display user profiles",
            "render user-generated content",
            "create a message board",
            "show rich text submitted by users"
        ],
        ["user-content-rendering", "output-encoding", "web-ui", "input-validation"]
    ),
    (
        "database-querying",
        ["sql", "query", "database", "search", "filter", "lookup"],
        [
            "create a search endpoint",
            "build a database query API",
            "filter records based on user input",
            "look up accounts by user input"
        ],
        ["database-querying", "input-validation", "request-handling"]
    ),
    (
        "api-request-handling",
        ["api", "request", "query parameter", "header", "body", "endpoint", "parameter", "url", "http"],
        [
            "build a REST API endpoint",
            "handle query parameters from requests",
            "process request headers and body",
            "create a webhook endpoint"
        ],
        ["api-request-handling", "request-handling", "input-validation"]
    ),
    (
        "memory-safety",
        ["buffer", "overflow", "stack", "heap", "memory", "out-of-bounds", "pointer", "strlen", "strcpy", "gets"],
        [
            "write a parser in C",
            "copy input into a fixed-size buffer",
            "handle strings in C/C++ code",
            "process untrusted binary input"
        ],
        ["memory-safety", "native-code", "input-validation", "bounds-checking"]
    ),
]

CONSEQUENCE_NORMALIZATION = {
    "execute unauthorized code or commands": "code-execution",
    "execute unauthorized code": "code-execution",
    "execute commands": "code-execution",
    "bypass protection mechanism": "protection-bypass",
    "read memory": "data-exposure",
    "read files or directories": "data-exposure",
    "read application data": "data-exposure",
    "modify memory": "memory-corruption",
    "dos: crash, exit, or restart": "crash",
    "dos: resource consumption (cpu)": "resource-exhaustion",
    "dos: resource consumption (memory)": "resource-exhaustion",
    "varies by context": "undefined-behavior",
}


def keyword_in_text(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def infer_task_tags_and_patterns(full_text: str) -> Tuple[List[str], List[str], List[str]]:
    tags: List[str] = []
    prompt_patterns: List[str] = []
    keywords: List[str] = []

    for rule_name, triggers, patterns, rule_tags in TASK_RULES:
        matched = any(keyword_in_text(full_text, trig) for trig in triggers)
        if matched:
            tags.extend(rule_tags)
            prompt_patterns.extend(patterns)
            keywords.extend(triggers)
            keywords.append(rule_name.replace("-", " "))

    # Small fallback for very generic prompt alignment
    if "input" in full_text.lower():
        tags.extend(["input-validation", "request-handling"])
        prompt_patterns.extend([
            "validate incoming user input",
            "build a form that accepts user data",
            "create an endpoint that accepts untrusted input"
        ])
        keywords.extend(["input", "user input", "untrusted input", "validation"])

    return (
        dedupe_preserve_order(tags),
        dedupe_preserve_order(prompt_patterns),
        dedupe_preserve_order(keywords),
    )


def short_text_signal(text: str, max_len: int = 120) -> str:
    text = normalize_space(text)
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].strip()
    return cut + "..."


def extract_cve_year(reference: str) -> Optional[int]:
    m = CVE_YEAR_RE.search(reference or "")
    return int(m.group(1)) if m else None


def select_top_guidance(mitigations: List[Dict], limit: int = 6) -> List[str]:
    guidance: List[str] = []

    for mit in mitigations:
        desc = normalize_space(mit.get("description", ""))
        if not desc:
            continue
        # Keep only first 1-2 sentences for retrieval guidance.
        summary = first_n_sentences(desc, n=2)
        if summary:
            guidance.append(summary)

    # Some lightweight normalization
    cleaned: List[str] = []
    for item in guidance:
        item = item.replace('"accept known good"', "allowlist-based")
        cleaned.append(item)

    return dedupe_preserve_order(cleaned)[:limit]


def normalize_consequence_keywords(consequences: List[Dict]) -> List[str]:
    out: List[str] = []

    for cons in consequences:
        for impact in cons.get("impact", []):
            key = impact.lower()
            out.append(CONSEQUENCE_NORMALIZATION.get(key, impact.lower().replace(" ", "-")))
        note = cons.get("note", "").lower()
        if "xss" in note or "cross-site scripting" in note:
            out.append("xss")
        if "sql injection" in note:
            out.append("sql-injection")
        if "path traversal" in note or "directory traversal" in note:
            out.append("path-traversal")
        if "command" in note and "execution" in note:
            out.append("code-execution")

    return dedupe_preserve_order(out)


def compact_platforms(platforms: List[str]) -> List[str]:
    cleaned = []
    for p in platforms:
        p = normalize_space(p)
        if not p:
            continue
        cleaned.append(p)
    return dedupe_preserve_order(cleaned)


def bag_of_words_keywords(*texts: str, max_keywords: int = 16) -> List[str]:
    stop = {
        "the", "and", "for", "that", "with", "this", "when", "from", "into", "then",
        "using", "used", "use", "allow", "allows", "code", "data", "input", "output",
        "can", "may", "which", "such", "than", "where", "have", "has", "will",
        "within", "their", "there", "been", "being", "over", "does", "does not",
        "also", "must", "should", "example", "examples", "user", "application",
        "program", "software", "product", "weakness", "attack", "attacker", "security",
        "safe", "unsafe", "proper", "improper", "incorrect", "validation"
    }
    counter = Counter()
    for text in texts:
        for tok in TOKEN_RE.findall(text.lower()):
            if tok in stop or len(tok) < 3:
                continue
            if tok.isdigit():
                continue
            counter[tok] += 1
    return [tok for tok, _ in counter.most_common(max_keywords)]


# ----------------------------
# Parsing
# ----------------------------

def parse_alternate_terms(weakness: ET.Element) -> List[str]:
    result = []
    alt_root = first_child(weakness, "Alternate_Terms")
    for alt in children_by_name(alt_root, "Alternate_Term"):
        term = normalize_space(flatten_xml_text(first_child(alt, "Term")) or first_child(alt, "Term").text if first_child(alt, "Term") is not None else "")
        if not term:
            term = normalize_space((first_child(alt, "Term").text if first_child(alt, "Term") is not None else ""))
        if term:
            result.append(term)
    return dedupe_preserve_order(result)


def parse_child_of(weakness: ET.Element) -> List[str]:
    out = []
    child_of_root = first_child(weakness, "Child_Of")
    if child_of_root is not None:
        for parent in children_by_name(child_of_root, "Parent"):
            cwe_id = normalize_space(parent.attrib.get("CWE_ID", ""))
            if cwe_id:
                out.append(cwe_id)
        return dedupe_preserve_order(out)

    related = first_child(weakness, "Related_Weaknesses")
    if related is not None:
        for rel in children_by_name(related, "Related_Weakness"):
            if rel.attrib.get("Nature") == "ChildOf":
                cwe_id = normalize_space(rel.attrib.get("CWE_ID", ""))
                if cwe_id:
                    out.append(cwe_id)
    return dedupe_preserve_order(out)


def parse_platforms(weakness: ET.Element) -> List[str]:
    out = []
    root = first_child(weakness, "Applicable_Platforms")
    for child in list(root) if root is not None else []:
        tag = local_name(child.tag)
        attrs = {k: v for k, v in child.attrib.items() if normalize_space(v)}
        parts = [tag]
        if "Name" in attrs:
            parts.append(attrs["Name"])
        if "Class" in attrs:
            parts.append(attrs["Class"])
        if "Prevalence" in attrs:
            parts.append(attrs["Prevalence"])
        if normalize_space(child.text):
            parts.append(normalize_space(child.text))
        out.append(": ".join(parts))
    return compact_platforms(out)


def parse_examples(weakness: ET.Element) -> List[Dict]:
    out = []
    root = first_child(weakness, "Demonstrative_Examples")
    if root is None:
        return out

    for ex in children_by_name(root, "Demonstrative_Example"):
        item = {
            "intro_text": normalize_space(flatten_xml_text(first_child(ex, "Intro_Text"))),
            "bad_code": "",
            "good_code": "",
            "attack_code": "",
            "body_text": [],
        }

        for code in children_by_name(ex, "Example_Code"):
            nature = normalize_space(code.attrib.get("Nature", ""))
            code_text = flatten_xml_text(code)
            if not code_text:
                continue
            if nature.lower() == "bad":
                item["bad_code"] = code_text
            elif nature.lower() == "good":
                item["good_code"] = code_text
            elif nature.lower() == "attack":
                item["attack_code"] = code_text

        for body in children_by_name(ex, "Body_Text"):
            body_text = flatten_xml_text(body)
            if body_text:
                item["body_text"].append(body_text)

        if item["intro_text"] or item["bad_code"] or item["good_code"] or item["attack_code"] or item["body_text"]:
            out.append(item)
    return out


def parse_observed_examples(weakness: ET.Element) -> List[Dict]:
    out = []
    # cleaned output name
    root = first_child(weakness, "Observed_Examples_Selected")
    if root is None:
        root = first_child(weakness, "Observed_Examples")

    if root is None:
        return out

    for obs in children_by_name(root, "Observed_Example"):
        ref = normalize_space(flatten_xml_text(first_child(obs, "Reference")) or (first_child(obs, "Reference").text if first_child(obs, "Reference") is not None else ""))
        desc = normalize_space(flatten_xml_text(first_child(obs, "Description")))
        link = normalize_space(flatten_xml_text(first_child(obs, "Link")) or (first_child(obs, "Link").text if first_child(obs, "Link") is not None else ""))
        year = extract_cve_year(ref) or (
            int(obs.attrib["Year"]) if "Year" in obs.attrib and obs.attrib["Year"].isdigit() else None
        )
        if year is not None and year <= 2022:
            continue
        if not ref and not desc:
            continue
        out.append({
            "reference": ref,
            "year": year,
            "description": desc,
            "link": link
        })
    return out


def parse_mitigations(weakness: ET.Element) -> List[Dict]:
    out = []
    root = first_child(weakness, "Potential_Mitigations")
    if root is None:
        return out

    for mit in children_by_name(root, "Mitigation"):
        phases = [normalize_space(flatten_xml_text(p) or p.text) for p in children_by_name(mit, "Phase")]
        phases = [p for p in phases if p]
        strategy = normalize_space(flatten_xml_text(first_child(mit, "Strategy")) or (first_child(mit, "Strategy").text if first_child(mit, "Strategy") is not None else ""))
        desc = flatten_xml_text(first_child(mit, "Description"))
        effectiveness = normalize_space(flatten_xml_text(first_child(mit, "Effectiveness")) or (first_child(mit, "Effectiveness").text if first_child(mit, "Effectiveness") is not None else ""))
        notes = flatten_xml_text(first_child(mit, "Effectiveness_Notes"))
        if phases or strategy or desc or effectiveness or notes:
            out.append({
                "phase": dedupe_preserve_order(phases),
                "strategy": strategy,
                "description": desc,
                "effectiveness": effectiveness,
                "effectiveness_notes": notes
            })
    return out


def parse_consequences(weakness: ET.Element) -> List[Dict]:
    out = []
    root = first_child(weakness, "Common_Consequences")
    if root is None:
        return out

    for cons in children_by_name(root, "Consequence"):
        scopes = [normalize_space(flatten_xml_text(s) or s.text) for s in children_by_name(cons, "Scope")]
        impacts = [normalize_space(flatten_xml_text(i) or i.text) for i in children_by_name(cons, "Impact")]
        note = flatten_xml_text(first_child(cons, "Note"))
        if scopes or impacts or note:
            out.append({
                "scope": dedupe_preserve_order([s for s in scopes if s]),
                "impact": dedupe_preserve_order([i for i in impacts if i]),
                "note": note
            })
    return out


def parse_description(weakness: ET.Element, tag: str) -> str:
    node = first_child(weakness, tag)
    return normalize_space(flatten_xml_text(node) if node is not None else "")


# ----------------------------
# Deterministic enrichment
# ----------------------------

def build_weakness_summary(name: str, description: str, ext_description: str) -> str:
    parts = []
    if description:
        parts.append(first_n_sentences(description, 2))
    if ext_description:
        extra = first_n_sentences(ext_description, 1)
        if extra and extra.lower() not in " ".join(parts).lower():
            parts.append(extra)
    summary = " ".join(parts).strip()
    if not summary:
        summary = name
    return summary


def build_developer_risk_summary(full_text: str, tags: List[str]) -> str:
    phrases = []

    if "authentication" in tags:
        phrases.append("Relevant when generating login, signup, password reset, session, token, and account access flows.")
    if "file-upload" in tags:
        phrases.append("Relevant when generating file upload, attachment handling, image upload, or document import features.")
    if "file-download" in tags or "path-handling" in tags:
        phrases.append("Relevant when code reads, writes, serves, extracts, or resolves user-influenced file paths.")
    if "xml-processing" in tags:
        phrases.append("Relevant when parsing XML, SOAP, or other structured documents from untrusted sources.")
    if "json-processing" in tags or "api-request-handling" in tags:
        phrases.append("Relevant when building APIs, webhook handlers, request parsing logic, and request-body processing.")
    if "user-content-rendering" in tags:
        phrases.append("Relevant when displaying or rendering user-controlled content in a browser or UI.")
    if "memory-safety" in tags:
        phrases.append("Relevant when generating C/C++ code that copies, indexes, allocates, or parses untrusted data.")

    if not phrases:
        phrases.append("Relevant whenever generated code accepts, transforms, stores, renders, or acts on untrusted input.")

    # Add risk statement
    lower = full_text.lower()
    risks = []
    if "xss" in lower or "cross-site" in lower:
        risks.append("cross-site scripting")
    if "sql injection" in lower:
        risks.append("SQL injection")
    if "path traversal" in lower or "directory traversal" in lower:
        risks.append("path traversal")
    if "code execution" in lower or "execute unauthorized code" in lower:
        risks.append("code execution")
    if "crash" in lower or "overflow" in lower or "out-of-bounds" in lower:
        risks.append("crashes or memory corruption")
    if "resource consumption" in lower or "resource exhaustion" in lower:
        risks.append("resource exhaustion")
    if "read memory" in lower or "read files" in lower or "sensitive information" in lower:
        risks.append("data exposure")

    if risks:
        phrases.append("Poor handling here can lead to " + ", ".join(dedupe_preserve_order(risks[:5])) + ".")

    return " ".join(phrases).strip()


def build_example_signals(examples: List[Dict], observed_examples: List[Dict], limit: int = 8) -> List[str]:
    out: List[str] = []

    for ex in examples[:5]:
        if ex.get("intro_text"):
            out.append(short_text_signal(ex["intro_text"], 100))
        for body in ex.get("body_text", [])[:2]:
            out.append(short_text_signal(body, 100))

    for obs in observed_examples[:4]:
        if obs.get("description"):
            out.append(short_text_signal(obs["description"], 100))

    return dedupe_preserve_order(out)[:limit]


def build_observed_context_keywords(observed_examples: List[Dict]) -> List[str]:
    joined = " ".join(obs.get("description", "") for obs in observed_examples)
    candidates = [
        "browser", "mobile", "firewall", "router", "kernel", "llm", "management tool",
        "iot", "embedded", "web browser", "proxy server", "xml", "machine-learning"
    ]
    found = [c for c in candidates if c.lower() in joined.lower()]
    return dedupe_preserve_order(found)


def build_retrieval_keywords(
    name: str,
    alternate_terms: List[str],
    task_tags: List[str],
    scenario_keywords: List[str],
    consequence_keywords: List[str],
    *texts: str
) -> List[str]:
    items: List[str] = []
    items.extend([name])
    items.extend(alternate_terms)
    items.extend(task_tags)
    items.extend(scenario_keywords)
    items.extend(consequence_keywords)
    items.extend(bag_of_words_keywords(*texts, max_keywords=18))

    # Add phrase variants for common tags
    tag_to_phrase = {
        "input-validation": "input validation",
        "session-management": "session management",
        "credential-handling": "credential storage",
        "file-upload": "file upload",
        "file-download": "file download",
        "xml-processing": "xml parsing",
        "json-processing": "json payload",
        "request-handling": "request body",
        "api-request-handling": "api endpoint",
        "user-content-rendering": "render user content",
        "path-handling": "file path",
        "memory-safety": "memory safety",
        "database-querying": "database query",
        "parser-safety": "parser configuration",
        "filesystem-access": "filesystem access",
    }
    for tag in task_tags:
        if tag in tag_to_phrase:
            items.append(tag_to_phrase[tag])

    return dedupe_preserve_order(items)[:30]


def compose_retrieval_text(
    cwe_id: str,
    name: str,
    weakness_summary: str,
    developer_risk_summary: str,
    task_tags: List[str],
    prompt_patterns: List[str],
    secure_coding_guidance: List[str],
    consequence_keywords: List[str],
    applicable_platforms: List[str],
    example_signals: List[str],
) -> str:
    parts = [
        f"{cwe_id} {name}.",
        weakness_summary,
        developer_risk_summary,
    ]

    if task_tags:
        parts.append("Relevant task areas: " + ", ".join(task_tags[:8]) + ".")
    if prompt_patterns:
        parts.append("Typical prompts: " + "; ".join(prompt_patterns[:5]) + ".")
    if consequence_keywords:
        parts.append("Potential security outcomes: " + ", ".join(consequence_keywords[:8]) + ".")
    if applicable_platforms:
        parts.append("Applicable platforms/context: " + ", ".join(applicable_platforms[:6]) + ".")
    if secure_coding_guidance:
        parts.append("Secure coding guidance: " + " ".join(secure_coding_guidance[:5]))
    if example_signals:
        parts.append("Example signals: " + "; ".join(example_signals[:4]) + ".")

    return " ".join(p.strip() for p in parts if p.strip())


# ----------------------------
# Main record build
# ----------------------------

def build_record(weakness: ET.Element) -> Dict:
    cwe_id_raw = normalize_space(weakness.attrib.get("ID", ""))
    cwe_id = f"CWE-{cwe_id_raw}" if cwe_id_raw and not cwe_id_raw.startswith("CWE-") else cwe_id_raw
    name = normalize_space(weakness.attrib.get("Name", ""))
    abstraction = normalize_space(weakness.attrib.get("Abstraction", ""))
    status = normalize_space(weakness.attrib.get("Status", ""))

    description = parse_description(weakness, "Description")
    ext_description = parse_description(weakness, "Extended_Description")
    alternate_terms = parse_alternate_terms(weakness)
    child_of = parse_child_of(weakness)
    applicable_platforms = parse_platforms(weakness)
    likelihood = parse_description(weakness, "Likelihood_Of_Exploit")
    examples = parse_examples(weakness)
    observed_examples = parse_observed_examples(weakness)
    mitigations = parse_mitigations(weakness)
    consequences = parse_consequences(weakness)

    full_text = " ".join([
        name,
        description,
        ext_description,
        " ".join(alternate_terms),
        " ".join(m.get("description", "") for m in mitigations),
        " ".join(c.get("note", "") for c in consequences),
        " ".join(ex.get("intro_text", "") for ex in examples),
        " ".join(" ".join(ex.get("body_text", [])) for ex in examples),
        " ".join(obs.get("description", "") for obs in observed_examples),
    ])

    task_tags, prompt_patterns, scenario_keywords = infer_task_tags_and_patterns(full_text)
    weakness_summary = build_weakness_summary(name, description, ext_description)
    developer_risk_summary = build_developer_risk_summary(full_text, task_tags)
    secure_coding_guidance = select_top_guidance(mitigations, limit=6)
    consequence_keywords = normalize_consequence_keywords(consequences)
    example_signals = build_example_signals(examples, observed_examples, limit=8)
    observed_context_keywords = build_observed_context_keywords(observed_examples)

    retrieval_keywords = build_retrieval_keywords(
        name,
        alternate_terms,
        task_tags,
        scenario_keywords,
        consequence_keywords,
        full_text
    )

    retrieval_title = f"{cwe_id}: {name}".strip(": ")
    retrieval_text = compose_retrieval_text(
        cwe_id=cwe_id,
        name=name,
        weakness_summary=weakness_summary,
        developer_risk_summary=developer_risk_summary,
        task_tags=task_tags,
        prompt_patterns=prompt_patterns,
        secure_coding_guidance=secure_coding_guidance,
        consequence_keywords=consequence_keywords,
        applicable_platforms=applicable_platforms,
        example_signals=example_signals,
    )

    return {
        "doc_id": cwe_id,
        "cwe_id": cwe_id,
        "name": name,
        "abstraction": abstraction,
        "status": status,

        "description_clean": description,
        "extended_description_clean": ext_description,
        "alternate_terms": alternate_terms,
        "child_of": child_of,
        "applicable_platforms": applicable_platforms,
        "likelihood_of_exploit": likelihood,

        "demonstrative_examples": examples,
        "observed_examples_selected": observed_examples,
        "potential_mitigations_structured": mitigations,
        "common_consequences_structured": consequences,

        "weakness_summary": weakness_summary,
        "developer_risk_summary": developer_risk_summary,
        "retrieval_keywords": retrieval_keywords,
        "task_tags": task_tags,
        "prompt_patterns": prompt_patterns,
        "secure_coding_guidance": secure_coding_guidance,
        "consequence_keywords": consequence_keywords,
        "example_signals": example_signals,
        "observed_context_keywords": observed_context_keywords,

        "retrieval_title": retrieval_title,
        "retrieval_text": retrieval_text,
    }


def find_weakness_nodes(root: ET.Element) -> List[ET.Element]:
    out = []
    if local_name(root.tag) == "Weakness":
        return [root]
    for elem in root.iter():
        if local_name(elem.tag) == "Weakness":
            out.append(elem)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python build_retrieval_dataset.py input.xml output.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    tree = ET.parse(input_path)
    root = tree.getroot()
    weakness_nodes = find_weakness_nodes(root)

    with output_path.open("w", encoding="utf-8") as f:
        for weakness in weakness_nodes:
            record = build_record(weakness)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(weakness_nodes)} retrieval records to {output_path}")


if __name__ == "__main__":
    main()
