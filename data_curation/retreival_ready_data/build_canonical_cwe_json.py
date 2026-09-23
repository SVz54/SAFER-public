#!/usr/bin/env python3
"""
Convert cleaned CWE XML into canonical structured JSON or JSONL.

This script is intentionally extraction-only:
- no inferred fields
- no summaries
- no retrieval enrichment
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def normalize_space(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def iter_text_with_breaks(elem: ET.Element) -> Iterable[str]:
    block_tags = {"p", "div", "ul", "ol", "li", "br"}

    if elem.text:
        yield elem.text

    for child in list(elem):
        child_tag = local_name(child.tag)
        if child_tag in block_tags:
            yield "\n"

        yield from iter_text_with_breaks(child)

        if child_tag in block_tags:
            yield "\n"

        if child.tail:
            yield child.tail


def flatten_xml_text(elem: Optional[ET.Element], preserve_newlines: bool = False) -> str:
    if elem is None:
        return ""

    raw = "".join(iter_text_with_breaks(elem)).replace("\xa0", " ")
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = raw.strip()
    if not preserve_newlines:
        return normalize_space(raw)
    return raw


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


def cwe_id_with_prefix(raw_id: str) -> str:
    raw_id = normalize_space(raw_id)
    if not raw_id:
        return ""
    if raw_id.startswith("CWE-"):
        return raw_id
    return f"CWE-{raw_id}"


def parse_alternate_terms(weakness: ET.Element) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    alt_root = first_child(weakness, "Alternate_Terms")
    for alt in children_by_name(alt_root, "Alternate_Term"):
        term = flatten_xml_text(first_child(alt, "Term"))
        description = flatten_xml_text(first_child(alt, "Description"))
        if term or description:
            out.append(
                {
                    "term": term,
                    "description": description,
                }
            )
    return out


def parse_child_of(weakness: ET.Element) -> List[str]:
    out: List[str] = []
    child_root = first_child(weakness, "Child_Of")
    if child_root is not None:
        for parent in children_by_name(child_root, "Parent"):
            parent_id = cwe_id_with_prefix(parent.attrib.get("CWE_ID", ""))
            if parent_id:
                out.append(parent_id)
        return out

    related = first_child(weakness, "Related_Weaknesses")
    if related is not None:
        for rel in children_by_name(related, "Related_Weakness"):
            if rel.attrib.get("Nature") != "ChildOf":
                continue
            parent_id = cwe_id_with_prefix(rel.attrib.get("CWE_ID", ""))
            if parent_id:
                out.append(parent_id)
    return out


def parse_applicable_platforms(weakness: ET.Element) -> List[str]:
    out: List[str] = []
    platforms_root = first_child(weakness, "Applicable_Platforms")
    for item in list(platforms_root) if platforms_root is not None else []:
        tag = local_name(item.tag)
        parts = [tag]

        for key in sorted(item.attrib.keys()):
            value = normalize_space(item.attrib.get(key, ""))
            if value:
                parts.append(f"{key}={value}")

        text = normalize_space(item.text)
        if text:
            parts.append(f"text={text}")

        out.append("|".join(parts))
    return out


def parse_demonstrative_examples(weakness: ET.Element) -> List[Dict]:
    out: List[Dict] = []
    examples_root = first_child(weakness, "Demonstrative_Examples")
    if examples_root is None:
        return out

    for ex in children_by_name(examples_root, "Demonstrative_Example"):
        intro_text = flatten_xml_text(first_child(ex, "Intro_Text"))

        code_examples = []
        for code_node in children_by_name(ex, "Example_Code"):
            code_text = flatten_xml_text(code_node, preserve_newlines=True)
            language = normalize_space(code_node.attrib.get("Language", ""))
            nature = normalize_space(code_node.attrib.get("Nature", "")).lower()
            if code_text:
                code_examples.append(
                    {
                        "language": language,
                        "nature": nature,
                        "code": code_text,
                    }
                )

        body_text = [
            flatten_xml_text(body)
            for body in children_by_name(ex, "Body_Text")
            if flatten_xml_text(body)
        ]

        out.append(
            {
                "intro_text": intro_text,
                "code_examples": code_examples,
                "body_text": body_text,
            }
        )

    return out


def parse_potential_mitigations(weakness: ET.Element) -> List[Dict]:
    out: List[Dict] = []
    mit_root = first_child(weakness, "Potential_Mitigations")
    if mit_root is None:
        return out

    for mit in children_by_name(mit_root, "Mitigation"):
        phase = [flatten_xml_text(node) for node in children_by_name(mit, "Phase") if flatten_xml_text(node)]
        strategy = flatten_xml_text(first_child(mit, "Strategy"))
        description = flatten_xml_text(first_child(mit, "Description"), preserve_newlines=True)
        effectiveness = flatten_xml_text(first_child(mit, "Effectiveness"))
        effectiveness_notes = flatten_xml_text(first_child(mit, "Effectiveness_Notes"), preserve_newlines=True)

        out.append(
            {
                "phase": phase,
                "strategy": strategy,
                "description": description,
                "effectiveness": effectiveness,
                "effectiveness_notes": effectiveness_notes,
            }
        )
    return out


def parse_common_consequences(weakness: ET.Element) -> List[Dict]:
    out: List[Dict] = []
    cons_root = first_child(weakness, "Common_Consequences")
    if cons_root is None:
        return out

    for cons in children_by_name(cons_root, "Consequence"):
        scope = [flatten_xml_text(node) for node in children_by_name(cons, "Scope") if flatten_xml_text(node)]
        impact = [flatten_xml_text(node) for node in children_by_name(cons, "Impact") if flatten_xml_text(node)]
        note = flatten_xml_text(first_child(cons, "Note"), preserve_newlines=True)

        out.append(
            {
                "scope": scope,
                "impact": impact,
                "note": note,
            }
        )
    return out


def parse_weakness(weakness: ET.Element) -> Dict:
    return {
        "cwe_id": cwe_id_with_prefix(weakness.attrib.get("ID", "")),
        "name": normalize_space(weakness.attrib.get("Name", "")),
        "abstraction": normalize_space(weakness.attrib.get("Abstraction", "")),
        "status": normalize_space(weakness.attrib.get("Status", "")),
        "description": flatten_xml_text(first_child(weakness, "Description")),
        "extended_description": flatten_xml_text(first_child(weakness, "Extended_Description"), preserve_newlines=True),
        "alternate_terms": parse_alternate_terms(weakness),
        "child_of": parse_child_of(weakness),
        "applicable_platforms": parse_applicable_platforms(weakness),
        "likelihood_of_exploit": flatten_xml_text(first_child(weakness, "Likelihood_Of_Exploit")),
        "demonstrative_examples": parse_demonstrative_examples(weakness),
        "potential_mitigations": parse_potential_mitigations(weakness),
        "common_consequences": parse_common_consequences(weakness),
    }


def parse_weakness_nodes(root: ET.Element) -> List[ET.Element]:
    if local_name(root.tag) == "Weakness":
        return [root]
    return [node for node in root.iter() if local_name(node.tag) == "Weakness"]


def write_json_array(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert cleaned CWE XML into canonical structured JSON."
    )
    parser.add_argument("input_xml", help="Path to cleaned CWE XML file")
    parser.add_argument("output_file", help="Path to output JSON/JSONL file")
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL instead of a JSON array",
    )
    args = parser.parse_args()

    tree = ET.parse(args.input_xml)
    root = tree.getroot()
    weakness_nodes = parse_weakness_nodes(root)
    records = [parse_weakness(node) for node in weakness_nodes]

    output_path = Path(args.output_file)
    if args.jsonl:
        write_jsonl(output_path, records)
    else:
        write_json_array(output_path, records)

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
