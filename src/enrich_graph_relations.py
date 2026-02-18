"""
Single-Pass Integrated Enrichment: Extract internal + cross-jurisdiction relations from graph chunks.
No cross_jurisdiction_regulations.json — outputs directly to *_relations.json.
Senior Graph Data Engineer: Ling Long
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# Cross-ref naming: ensures seamless Neo4j MERGE
CROSS_REF_PREFIX_KR = "KR_Article_"
CROSS_REF_PREFIX_EU = "EU_Article_"

RELATION_TYPES = [
    "DEFINES", "ESTABLISHES", "IMPOSES", "ENCOMPASSES", "SUPPLEMENTS",
    "MANDATED_FOR", "APPLIES_TO", "INCLUDES", "IS_A", "DETAILS_SCOPE",
    "DETAILS_DEFINITION_OF", "PENALIZES_WITH", "JUSTIFIES_LOGIC",
    "LEADS_TO", "TRIGGERS", "ALIGNS_WITH", "CONTRADICTS", "REFERENCES",
]


class SingleRelation(BaseModel):
    type: str = Field(description="Relationship type")
    target_node_type: str = Field(description="Article, Concept, Stakeholder, etc.")
    target_node_name: str = Field(description="Target node identifier")
    description: str = Field(description="Brief description of the relationship")
    metadata: Literal["INTERNAL", "CROSS_JURISDICTION"] = Field(
        default="INTERNAL",
        description="INTERNAL = same jurisdiction; CROSS_JURISDICTION = KR<->EU link",
    )


class ArticleRelations(BaseModel):
    relations: List[SingleRelation] = Field(default_factory=list)


SYSTEM_PROMPT_KR = """You are a legal knowledge engineer extracting structured relationships from the Korea AI Core Law.

You are aware of the EU AI Act. If this KR article references, aligns with, supplements, or contradicts a specific EU AI Act provision, extract a CROSS_JURISDICTION relationship with:
- target_node_type: "Article"
- target_node_name: "EU_Article_N" (e.g., EU_Article_5 for EU AI Act Article 5)
- metadata: "CROSS_JURISDICTION"

For internal (KR-only) relations, use metadata: "INTERNAL" and standard target_node_type (Concept, Stakeholder, Domain, etc.).

Relation types: DEFINES, ESTABLISHES, IMPOSES, ENCOMPASSES, SUPPLEMENTS, MANDATED_FOR, APPLIES_TO, ALIGNS_WITH, CONTRADICTS, REFERENCES, and others as appropriate.

Output valid JSON with a "relations" array. Each relation: type, target_node_type, target_node_name, description, metadata."""

SYSTEM_PROMPT_EU = """You are a legal knowledge engineer extracting structured relationships from the EU AI Act.

You are aware of the Korea AI Core Law. If this EU provision references, aligns with, supplements, or contradicts a specific KR AI Core Law article, extract a CROSS_JURISDICTION relationship with:
- target_node_type: "Article"
- target_node_name: "KR_Article_N" (e.g., KR_Article_5 for Korea AI Act Article 5)
- metadata: "CROSS_JURISDICTION"

For internal (EU-only) relations, use metadata: "INTERNAL" and standard target_node_type (Concept, Stakeholder, Regulation, etc.).

Relation types: DEFINES, ESTABLISHES, IMPOSES, ENCOMPASSES, SUPPLEMENTS, MANDATED_FOR, APPLIES_TO, ALIGNS_WITH, CONTRADICTS, REFERENCES, and others as appropriate.

Output valid JSON with a "relations" array. Each relation: type, target_node_type, target_node_name, description, metadata."""


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_llm():
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )


def _normalize_cross_ref_name(name: str, source: str) -> str:
    """Ensure cross-ref uses KR_Article_N or EU_Article_N convention."""
    s = (name or "").strip()
    if not s:
        return s
    s_upper = s.upper()
    if source == "Korea AI Law" and (s_upper.startswith("EU_ARTICLE_") or "EU " in s_upper[:10]):
        return s
    if source == "EU AI Act" and (s_upper.startswith("KR_ARTICLE_") or "KR " in s_upper[:10]):
        return s
    if source == "Korea AI Law" and ("EU" in s_upper or "ARTICLE" in s_upper):
        import re
        m = re.search(r"ARTICLE\s*(\d+)", s_upper, re.I)
        if m:
            return f"EU_Article_{m.group(1)}"
    if source == "EU AI Act" and ("KR" in s_upper or "KOREA" in s_upper or "ARTICLE" in s_upper):
        import re
        m = re.search(r"ARTICLE\s*(\d+)", s_upper, re.I)
        if m:
            return f"KR_Article_{m.group(1)}"
    return s


def _parse_llm_response(text: str) -> List[Dict]:
    """Parse LLM response into list of relation dicts."""
    try:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = next((i for i, l in enumerate(lines) if "```" in l and i > 0), len(lines))
            text = "\n".join(lines[1:start])
        data = json.loads(text)
        rels = data.get("relations") or data.get("relation") or data
        if isinstance(rels, dict):
            rels = [rels]
        out = []
        for r in rels:
            if isinstance(r, dict) and r.get("type"):
                out.append({
                    "type": r.get("type", "RELATES_TO"),
                    "target_node_type": r.get("target_node_type", "Concept"),
                    "target_node_name": (r.get("target_node_name") or "").strip()[:500],
                    "description": (r.get("description") or "")[:2000],
                    "metadata": r.get("metadata", "INTERNAL"),
                })
        return out
    except Exception:
        return []


async def extract_relations_async(
    llm, text: str, item_id: str, source: str, system_prompt: str
) -> List[Dict]:
    """Call LLM to extract relations from text."""
    user = f"Extract legal relationships from this text. Source item: {item_id}\n\nText:\n{text[:12000]}"
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user)]
        msg = llm.invoke(messages)
        content = msg.content if hasattr(msg, "content") else str(msg)
        rels = _parse_llm_response(content)
        for r in rels:
            r["target_node_name"] = _normalize_cross_ref_name(r["target_node_name"], source)
        return rels
    except Exception as e:
        print(f"  [WARN] LLM error for {item_id}: {e}")
        return []


def collect_items(data: dict) -> List[Dict]:
    """Collect all article/rationale/annex items with id and full_text."""
    out = []
    for section in ("articles", "rationale", "annexes", "addenda"):
        for item in data.get(section) or []:
            nid = item.get("id")
            if nid:
                out.append({
                    "section": section,
                    "id": nid,
                    "full_text": item.get("full_text") or "",
                    "item": item,
                })
    return out


async def run_enrichment(
    chunk_path: Path,
    out_path: Path,
    source: str,
    system_prompt: str,
    batch_size: int = 5,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Single-pass integrated enrichment."""
    data = load_json(chunk_path)
    items = collect_items(data)
    llm = get_llm()
    checkpoint: Dict[str, List[Dict]] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = load_json(checkpoint_path)
        except Exception:
            checkpoint = {}

    processed = 0
    total_relations = 0
    cross_count = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(items, desc=f"Enriching {source}", unit="item")
    except ImportError:
        pbar = items

    for i, rec in enumerate(pbar):
        nid = rec["id"]
        key = f"{rec['section']}::{nid}"
        if key in checkpoint:
            rels = checkpoint[key]
        else:
            text = rec["full_text"]
            if not text.strip():
                rels = []
            else:
                rels = await extract_relations_async(
                    llm, text, nid, source, system_prompt
                )
            checkpoint[key] = rels
            if checkpoint_path:
                save_json(checkpoint_path, checkpoint)

        rec["item"]["relations"] = rels
        processed += 1
        total_relations += len(rels)
        cross_count += sum(1 for r in rels if r.get("metadata") == "CROSS_JURISDICTION")

        if (i + 1) % batch_size == 0:
            save_json(out_path, data)
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(rels=total_relations, cross=cross_count)

    save_json(out_path, data)
    if checkpoint_path:
        save_json(checkpoint_path, checkpoint)

    return {"processed": processed, "total_relations": total_relations, "cross_jurisdiction": cross_count}


def main():
    import argparse
    processed = _project_root / "data" / "processed"
    parser = argparse.ArgumentParser(description="Single-pass integrated relation enrichment")
    parser.add_argument("--kr", action="store_true", help="Process Korea AI Law only")
    parser.add_argument("--eu", action="store_true", help="Process EU AI Act only")
    parser.add_argument("--batch", type=int, default=5, help="Batch size for checkpointing")
    args = parser.parse_args()
    do_kr = args.kr or not args.eu
    do_eu = args.eu or not args.kr

    async def run():
        results = []
        if do_kr:
            r = await run_enrichment(
                processed / "AICoreLaw_graph_chunk.json",
                processed / "AICoreLaw_relations.json",
                "Korea AI Law",
                SYSTEM_PROMPT_KR,
                batch_size=args.batch,
                checkpoint_path=processed / "enrich_checkpoint_KR.json",
            )
            results.append(("Korea AI Law", r))
            print(f"KR: {r['processed']} items, {r['total_relations']} relations ({r['cross_jurisdiction']} cross)")
        if do_eu:
            r = await run_enrichment(
                processed / "AIAct_graph_chunk.json",
                processed / "AIAct_relations.json",
                "EU AI Act",
                SYSTEM_PROMPT_EU,
                batch_size=args.batch,
                checkpoint_path=processed / "enrich_checkpoint_EU.json",
            )
            results.append(("EU AI Act", r))
            print(f"EU: {r['processed']} items, {r['total_relations']} relations ({r['cross_jurisdiction']} cross)")
        return results

    asyncio.run(run())
    print("Done. Relations saved to AICoreLaw_relations.json and AIAct_relations.json")


if __name__ == "__main__":
    main()
