# Retrieval Enrichment Prompt Template

Use this later if you add an LLM enrichment stage after the deterministic builder.

## Goal
Transform one cleaned CWE record into a retrieval-oriented JSON object for secure code-generation RAG.

## Input fields
- cwe_id
- name
- description_clean
- extended_description_clean
- alternate_terms
- applicable_platforms
- likelihood_of_exploit
- demonstrative_examples
- observed_examples_selected
- potential_mitigations_structured
- common_consequences_structured

## Output fields
Return JSON only with:
- weakness_summary
- developer_risk_summary
- retrieval_keywords
- task_tags
- prompt_patterns
- secure_coding_guidance
- consequence_keywords
- example_signals
- observed_context_keywords
- retrieval_title
- retrieval_text

## Constraints
- Stay grounded in the source fields.
- Do not invent unrelated app scenarios.
- Write for prompt-to-security retrieval.
- Prefer developer task language over formal taxonomy language.
- Keep retrieval_text dense, concise, and useful for embeddings.
- Keep secure_coding_guidance action-oriented.
- Prompt patterns should resemble realistic code-generation prompts.
- Output valid JSON only.
