#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np
from openai import OpenAI

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+\-/.]{2,}")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_DIMENSIONS = os.getenv("EMBEDDING_DIMENSIONS")
PROMPT_CANDIDATES=["prompt","prompt_text","query","question"]
QUERY_ID_CANDIDATES=["query_id","id","prompt_id"]

def normalize_text(text: Any) -> str: return " ".join(str(text).split()).strip()
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: rows.append(json.loads(line))
    return rows
def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
def first_present(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] not in ("", None): return row[key]
    return None
def load_prompts(path: Path) -> List[Dict[str, Any]]:
    suffix=path.suffix.lower()
    if suffix==".jsonl": raw_rows=load_jsonl(path)
    elif suffix==".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f: raw_rows=list(csv.DictReader(f))
    else: raise ValueError("Prompt file must be .csv or .jsonl")
    prompts=[]
    for i,row in enumerate(raw_rows, start=1):
        prompt=normalize_text(first_present(row, PROMPT_CANDIDATES))
        if not prompt: raise ValueError(f"Row {i} missing prompt field. Tried: {PROMPT_CANDIDATES}")
        query_id=normalize_text(first_present(row, QUERY_ID_CANDIDATES) or f"Q{i:03d}")
        prompts.append({"query_id": query_id, "prompt": prompt, "raw_row": row})
    return prompts
def compact_list(values: Any) -> List[str]:
    if not isinstance(values, list): return []
    out=[]; seen=set()
    for value in values:
        if not isinstance(value, str): continue
        cleaned=normalize_text(value)
        if not cleaned: continue
        key=cleaned.lower()
        if key in seen: continue
        seen.add(key); out.append(cleaned)
    return out
def tokenize(text: str) -> List[str]: return [tok.lower() for tok in TOKEN_RE.findall(text)]
def build_lexical_document(record: Dict[str, Any]) -> str:
    parts=[]
    def add_text(value: str, repeat: int=1):
        value=normalize_text(value)
        if not value: return
        for _ in range(repeat): parts.append(value)
    def add_list(values: Any, repeat: int=1):
        for item in compact_list(values): add_text(item, repeat)
    add_text(record.get("name",""),3); add_text(record.get("retrieval_title",""),2)
    add_list(record.get("alternate_terms"),2); add_list(record.get("retrieval_keywords"),3)
    add_list(record.get("prompt_patterns"),3); add_list(record.get("task_tags"),2)
    add_list(record.get("consequence_keywords"),2); add_list(record.get("secure_coding_guidance"),1)
    add_text(record.get("weakness_summary",""),1); add_text(record.get("developer_risk_summary",""),1)
    add_text(record.get("retrieval_text",""),1)
    return "\n".join(parts)
@dataclass
class BM25Index:
    tokenized_docs: List[List[str]]
    doc_freqs: List[Dict[str,int]]
    idf: Dict[str,float]
    avgdl: float
    k1: float=1.5
    b: float=0.75
    @classmethod
    def build(cls, docs: List[str]) -> "BM25Index":
        tokenized_docs=[tokenize(doc) for doc in docs]; doc_freqs=[]; df={}; total_len=0
        for tokens in tokenized_docs:
            total_len += len(tokens); freqs={}
            for token in tokens: freqs[token]=freqs.get(token,0)+1
            doc_freqs.append(freqs)
            for token in freqs: df[token]=df.get(token,0)+1
        n_docs=max(len(tokenized_docs),1); avgdl=total_len/n_docs if n_docs else 0.0
        idf={term: math.log(1+(n_docs-freq+0.5)/(freq+0.5)) for term,freq in df.items()}
        return cls(tokenized_docs, doc_freqs, idf, avgdl)
    def score(self, query: str) -> np.ndarray:
        q_tokens=tokenize(query); scores=np.zeros(len(self.tokenized_docs), dtype=np.float32)
        for i,freqs in enumerate(self.doc_freqs):
            doc_len=len(self.tokenized_docs[i]) or 1; score=0.0
            for term in q_tokens:
                if term not in freqs: continue
                tf=freqs[term]; idf=self.idf.get(term,0.0)
                denom=tf + self.k1*(1-self.b + self.b*doc_len/max(self.avgdl,1e-9))
                score += idf*(tf*(self.k1+1))/denom
            scores[i]=score
        return scores
def minmax_scale(scores: np.ndarray) -> np.ndarray:
    if scores.size==0: return scores
    min_v=float(scores.min()); max_v=float(scores.max())
    if abs(max_v-min_v)<1e-12: return np.zeros_like(scores)
    return (scores-min_v)/(max_v-min_v)
def embed_query(client: OpenAI, query: str) -> np.ndarray:
    kwargs={"model": DEFAULT_MODEL, "input": query}
    if DEFAULT_DIMENSIONS: kwargs["dimensions"]=int(DEFAULT_DIMENSIONS)
    response=client.embeddings.create(**kwargs)
    vec=np.array(response.data[0].embedding, dtype=np.float32); norm=np.linalg.norm(vec)
    return vec if norm==0 else vec/norm
def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("prompt_file", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--method", choices=["vector","bm25","hybrid"], default="hybrid")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--vector-weight", type=float, default=0.7)
    parser.add_argument("--bm25-weight", type=float, default=0.3)
    args=parser.parse_args()
    if args.method=="hybrid" and abs((args.vector_weight+args.bm25_weight)-1.0)>1e-9:
        raise ValueError("vector-weight and bm25-weight must sum to 1.0")
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key: raise RuntimeError("OPENAI_API_KEY is not set.")
    metadata=load_jsonl(args.index_dir/"metadata.jsonl")
    embeddings=np.load(args.index_dir/"embeddings.npy")
    prompts=load_prompts(args.prompt_file)
    bm25=BM25Index.build([build_lexical_document(record) for record in metadata])
    client=OpenAI(api_key=api_key); run_name=args.run_name or f"{args.method}_run"
    rows=[]
    for idx,item in enumerate(prompts, start=1):
        prompt=item["prompt"]; q_vec=embed_query(client, prompt)
        vector_scores=embeddings @ q_vec; bm25_scores=bm25.score(prompt)
        if args.method=="vector":
            ranked_idx=np.argsort(-vector_scores); combined_scores=vector_scores
        elif args.method=="bm25":
            ranked_idx=np.argsort(-bm25_scores); combined_scores=bm25_scores
        else:
            vector_scaled=minmax_scale(vector_scores); bm25_scaled=minmax_scale(bm25_scores)
            combined_scores=args.vector_weight*vector_scaled + args.bm25_weight*bm25_scaled
            ranked_idx=np.argsort(-combined_scores)
        ranked_results=[]
        for rank,rec_idx in enumerate(ranked_idx[:args.top_k], start=1):
            record=metadata[int(rec_idx)]
            ranked_results.append({"rank": rank, "cwe_id": record.get("cwe_id"), "name": record.get("name"),
                                   "retrieval_title": record.get("retrieval_title"),
                                   "score": float(combined_scores[rec_idx]), "vector_score": float(vector_scores[rec_idx]),
                                   "bm25_score": float(bm25_scores[rec_idx]), "task_tags": record.get("task_tags",[])})
        rows.append({"query_id": item["query_id"], "prompt": prompt, "method": args.method, "run_name": run_name,
                     "config": {"top_k": args.top_k, "embedding_model": DEFAULT_MODEL,
                                "embedding_dimensions": int(DEFAULT_DIMENSIONS) if DEFAULT_DIMENSIONS else None,
                                "vector_weight": args.vector_weight if args.method=="hybrid" else None,
                                "bm25_weight": args.bm25_weight if args.method=="hybrid" else None},
                     "ranked_results": ranked_results})
        print(f"Retrieved {idx}/{len(prompts)} prompts")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_jsonl, rows)
    print(f"Wrote retrieval results to: {args.output_jsonl}")
if __name__=="__main__":
    main()
