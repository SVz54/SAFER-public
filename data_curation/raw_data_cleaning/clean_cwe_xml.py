#!/usr/bin/env python3
"""
Clean CWE XML into a reduced XML file for retrieval / generation preparation.

Keeps only:
1. Description
2. Extended_Description
3. Alternate_Terms
4. Child_Of (derived from Related_Weaknesses where Nature="ChildOf")
5. Applicable_Platforms
6. Likelihood_Of_Exploit
7. Demonstrative_Examples
8. Observed_Examples_Selected (only CVEs with year > 2022)
9. Potential_Mitigations
10. Common_Consequences

Usage:
    python clean_cwe_xml.py input.xml output_cleaned.xml
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from typing import Iterable, Optional

CWE_NS = "http://cwe.mitre.org/cwe-7"
XHTML_NS = "http://www.w3.org/1999/xhtml"
NS = {"cwe": CWE_NS, "xhtml": XHTML_NS}

ET.register_namespace("", CWE_NS)

CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-\d+", re.IGNORECASE)
OBSERVED_MIN_YEAR_EXCLUSIVE = 2022
KEEP_WEAKNESS_ATTRS = ("ID", "Name", "Abstraction", "Structure", "Status")


def local_name(tag: str) -> str:
    """Return the local name of an XML tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def namespace_uri(tag: str) -> str:
    """Extract namespace URI from a tag like '{uri}LocalName'."""
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def normalize_space(text: Optional[str]) -> str:
    """Collapse repeated whitespace and trim."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def iter_text_with_breaks(elem: ET.Element) -> Iterable[str]:
    """
    Recursively collect text while preserving some structure from XHTML tags.

    Inserts line breaks around block-level elements and <br>.
    """
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


def flatten_xml_text(elem: Optional[ET.Element]) -> str:
    """Convert an XML/XHTML subtree to readable plain text."""
    if elem is None:
        return ""

    raw = "".join(iter_text_with_breaks(elem))
    raw = raw.replace("\xa0", " ")

    # Clean up repeated whitespace while preserving paragraph-ish line breaks.
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
    raw = re.sub(r" *\n *", "\n", raw)

    return raw.strip()


def append_text_element(parent: ET.Element, tag: str, text: str) -> Optional[ET.Element]:
    """Append child only if text is non-empty."""
    text = normalize_space(text) if "\n" not in text else text.strip()
    if not text:
        return None
    child = ET.SubElement(parent, tag)
    child.text = text
    return child


def build_alternate_terms(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_alt = src_weakness.find("cwe:Alternate_Terms", ns)
    if src_alt is None:
        return

    dst_alt = ET.SubElement(dst_weakness, "Alternate_Terms")

    for term_node in src_alt.findall("cwe:Alternate_Term", ns):
        term_text = normalize_space(term_node.findtext("cwe:Term", default="", namespaces=ns))
        desc_text = flatten_xml_text(term_node.find("cwe:Description", ns))
        if not term_text and not desc_text:
            continue

        alt = ET.SubElement(dst_alt, "Alternate_Term")
        append_text_element(alt, "Term", term_text)
        if desc_text:
            append_text_element(alt, "Description", desc_text)

    if len(dst_alt) == 0:
        dst_weakness.remove(dst_alt)


def build_child_of(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_related = src_weakness.find("cwe:Related_Weaknesses", ns)
    if src_related is None:
        return

    child_nodes = []
    seen = set()

    for rel in src_related.findall("cwe:Related_Weakness", ns):
        if rel.get("Nature") != "ChildOf":
            continue

        cwe_id = rel.get("CWE_ID", "").strip()
        if not cwe_id or cwe_id in seen:
            continue

        seen.add(cwe_id)
        child_nodes.append(cwe_id)

    if not child_nodes:
        return

    dst_child_of = ET.SubElement(dst_weakness, "Child_Of")
    for cwe_id in child_nodes:
        parent = ET.SubElement(dst_child_of, "Parent")
        parent.set("CWE_ID", cwe_id)


def build_applicable_platforms(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_platforms = src_weakness.find("cwe:Applicable_Platforms", ns)
    if src_platforms is None:
        return

    dst_platforms = ET.SubElement(dst_weakness, "Applicable_Platforms")

    for child in list(src_platforms):
        tag = local_name(child.tag)
        out = ET.SubElement(dst_platforms, tag)

        # Keep attributes because they are meaningful here.
        for key, value in child.attrib.items():
            if value and value.strip():
                out.set(key, value.strip())

        text = normalize_space(child.text)
        if text:
            out.text = text

    if len(dst_platforms) == 0:
        dst_weakness.remove(dst_platforms)


def build_demonstrative_examples(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_examples = src_weakness.find("cwe:Demonstrative_Examples", ns)
    if src_examples is None:
        return

    dst_examples = ET.SubElement(dst_weakness, "Demonstrative_Examples")

    for ex in src_examples.findall("cwe:Demonstrative_Example", ns):
        dst_ex = ET.SubElement(dst_examples, "Demonstrative_Example")

        if ex.get("Demonstrative_Example_ID"):
            dst_ex.set("Demonstrative_Example_ID", ex.get("Demonstrative_Example_ID"))

        intro = normalize_space(ex.findtext("cwe:Intro_Text", default="", namespaces=ns))
        if intro:
            append_text_element(dst_ex, "Intro_Text", intro)

        for code_node in ex.findall("cwe:Example_Code", ns):
            code_text = flatten_xml_text(code_node)
            if not code_text:
                continue
            dst_code = ET.SubElement(dst_ex, "Example_Code")
            if code_node.get("Nature"):
                dst_code.set("Nature", code_node.get("Nature"))
            if code_node.get("Language"):
                dst_code.set("Language", code_node.get("Language"))
            dst_code.text = code_text

        for body_node in ex.findall("cwe:Body_Text", ns):
            body_text = flatten_xml_text(body_node)
            if body_text:
                append_text_element(dst_ex, "Body_Text", body_text)

        if len(dst_ex) == 0:
            dst_examples.remove(dst_ex)

    if len(dst_examples) == 0:
        dst_weakness.remove(dst_examples)


def extract_cve_year(reference_text: str) -> Optional[int]:
    match = CVE_YEAR_RE.search(reference_text or "")
    if not match:
        return None
    return int(match.group(1))


def build_observed_examples(
    src_weakness: ET.Element,
    dst_weakness: ET.Element,
    ns: dict[str, str],
    min_year_exclusive: int = OBSERVED_MIN_YEAR_EXCLUSIVE,
) -> None:
    src_obs = src_weakness.find("cwe:Observed_Examples", ns)
    if src_obs is None:
        return

    dst_obs = ET.SubElement(dst_weakness, "Observed_Examples_Selected")

    for obs in src_obs.findall("cwe:Observed_Example", ns):
        ref = normalize_space(obs.findtext("cwe:Reference", default="", namespaces=ns))
        year = extract_cve_year(ref)
        if year is None or year <= min_year_exclusive:
            continue

        desc = normalize_space(obs.findtext("cwe:Description", default="", namespaces=ns))
        link = normalize_space(obs.findtext("cwe:Link", default="", namespaces=ns))

        dst_item = ET.SubElement(dst_obs, "Observed_Example")
        dst_item.set("Year", str(year))
        append_text_element(dst_item, "Reference", ref)
        if desc:
            append_text_element(dst_item, "Description", desc)
        if link:
            append_text_element(dst_item, "Link", link)

    if len(dst_obs) == 0:
        dst_weakness.remove(dst_obs)


def build_potential_mitigations(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_mits = src_weakness.find("cwe:Potential_Mitigations", ns)
    if src_mits is None:
        return

    dst_mits = ET.SubElement(dst_weakness, "Potential_Mitigations")

    for mit in src_mits.findall("cwe:Mitigation", ns):
        dst_mit = ET.SubElement(dst_mits, "Mitigation")

        if mit.get("Mitigation_ID"):
            dst_mit.set("Mitigation_ID", mit.get("Mitigation_ID"))

        for phase_node in mit.findall("cwe:Phase", ns):
            phase = normalize_space(phase_node.text)
            if phase:
                append_text_element(dst_mit, "Phase", phase)

        strategy = normalize_space(mit.findtext("cwe:Strategy", default="", namespaces=ns))
        if strategy:
            append_text_element(dst_mit, "Strategy", strategy)

        desc_text = flatten_xml_text(mit.find("cwe:Description", ns))
        if desc_text:
            append_text_element(dst_mit, "Description", desc_text)

        effectiveness = normalize_space(mit.findtext("cwe:Effectiveness", default="", namespaces=ns))
        if effectiveness:
            append_text_element(dst_mit, "Effectiveness", effectiveness)

        notes_text = flatten_xml_text(mit.find("cwe:Effectiveness_Notes", ns))
        if notes_text:
            append_text_element(dst_mit, "Effectiveness_Notes", notes_text)

        if len(dst_mit) == 0:
            dst_mits.remove(dst_mit)

    if len(dst_mits) == 0:
        dst_weakness.remove(dst_mits)


def build_common_consequences(src_weakness: ET.Element, dst_weakness: ET.Element, ns: dict[str, str]) -> None:
    src_cc = src_weakness.find("cwe:Common_Consequences", ns)
    if src_cc is None:
        return

    dst_cc = ET.SubElement(dst_weakness, "Common_Consequences")

    for cons in src_cc.findall("cwe:Consequence", ns):
        dst_cons = ET.SubElement(dst_cc, "Consequence")

        for scope_node in cons.findall("cwe:Scope", ns):
            scope = normalize_space(scope_node.text)
            if scope:
                append_text_element(dst_cons, "Scope", scope)

        for impact_node in cons.findall("cwe:Impact", ns):
            impact = normalize_space(impact_node.text)
            if impact:
                append_text_element(dst_cons, "Impact", impact)

        note_text = flatten_xml_text(cons.find("cwe:Note", ns))
        if note_text:
            append_text_element(dst_cons, "Note", note_text)

        if len(dst_cons) == 0:
            dst_cc.remove(dst_cons)

    if len(dst_cc) == 0:
        dst_weakness.remove(dst_cc)


def clean_catalog(input_path: str, output_path: str) -> None:
    tree = ET.parse(input_path)
    root = tree.getroot()
    ns_uri = namespace_uri(root.tag) or CWE_NS
    ns = {"cwe": ns_uri, "xhtml": XHTML_NS}
    ET.register_namespace("", ns_uri)

    cleaned_root = ET.Element(root.tag, root.attrib)

    src_weaknesses = root.find("cwe:Weaknesses", ns)
    if src_weaknesses is None:
        raise ValueError("Could not find <Weaknesses> section in input XML.")

    cleaned_weaknesses = ET.SubElement(cleaned_root, "Weaknesses")

    for weakness in src_weaknesses.findall("cwe:Weakness", ns):
        dst_weakness = ET.SubElement(cleaned_weaknesses, "Weakness")

        # Keep core identifying attributes so the record remains usable.
        for attr in KEEP_WEAKNESS_ATTRS:
            value = weakness.get(attr)
            if value:
                dst_weakness.set(attr, value)

        description = normalize_space(weakness.findtext("cwe:Description", default="", namespaces=ns))
        if description:
            append_text_element(dst_weakness, "Description", description)

        ext_desc = flatten_xml_text(weakness.find("cwe:Extended_Description", ns))
        if ext_desc:
            append_text_element(dst_weakness, "Extended_Description", ext_desc)

        build_alternate_terms(weakness, dst_weakness, ns)
        build_child_of(weakness, dst_weakness, ns)
        build_applicable_platforms(weakness, dst_weakness, ns)

        likelihood = normalize_space(weakness.findtext("cwe:Likelihood_Of_Exploit", default="", namespaces=ns))
        if likelihood:
            append_text_element(dst_weakness, "Likelihood_Of_Exploit", likelihood)

        build_demonstrative_examples(weakness, dst_weakness, ns)
        build_observed_examples(
            weakness,
            dst_weakness,
            ns,
            min_year_exclusive=OBSERVED_MIN_YEAR_EXCLUSIVE,
        )
        build_potential_mitigations(weakness, dst_weakness, ns)
        build_common_consequences(weakness, dst_weakness, ns)

    cleaned_tree = ET.ElementTree(cleaned_root)
    ET.indent(cleaned_tree, space="  ")
    cleaned_tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python clean_cwe_xml.py input.xml output_cleaned.xml")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    try:
        clean_catalog(input_path, output_path)
    except FileNotFoundError as exc:
        print(f"Input file not found: {exc.filename}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as exc:
        print(f"Invalid XML in input file: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Cleaned XML written to: {output_path}")


if __name__ == "__main__":
    main()
