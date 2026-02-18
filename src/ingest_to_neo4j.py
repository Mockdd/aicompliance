"""
Neo4j ingestion for legal graph: structural nodes (Article, Rationale, Annex, Addenda),
Chunks with vector embeddings, HAS_CHUNK relations, and semantic relations.
Idempotent MERGE; schema export to data/processed/graph_schema_summary.json.

Relation: (Article|Rationale|Annex|Addenda)-[:HAS_CHUNK]->(Chunk)
- AI Act: Article, Rationale, Annex each have multiple Chunks with embeddings.
- AICoreLaw: Article, Addenda each have multiple Chunks with embeddings.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# ---------------------------------------------------------------------------
# Configuration (rename labels here for flexibility)
# ---------------------------------------------------------------------------
CONFIG = {
    "LABEL_ARTICLE": "Article",
    "LABEL_RATIONALE": "Rationale",
    "LABEL_ANNEX": "Annex",
    "LABEL_ADDENDA": "Article",  # Addenda as Article; or set "Addenda"
    "LABEL_CHUNK": "Chunk",
    "REL_HAS_CHUNK": "HAS_CHUNK",  # (Article|Rationale|Annex|Addenda)-[:HAS_CHUNK]->(Chunk)
    "REL_REGULATION_CONTAINS": "INCLUDES",  # (Regulation)-[:INCLUDES]->(Article|Rationale|Annex|Addenda)
    "VECTOR_INDEX_NAME": "legal_chunk_vector",
    "VECTOR_DIMENSIONS": 1536,
    "VECTOR_SIMILARITY": "cosine",
    "CLEAR_DB_BY_SOURCE": False,
    "BATCH_SIZE": 80,
    "STRUCT_BATCH_SIZE": 40,  # Smaller: structural nodes have large full_text; avoids Neo4j request size limit
}

# JSON section key -> config label key
STRUCTURAL_SECTIONS = [
    ("articles", "LABEL_ARTICLE"),
    ("rationale", "LABEL_RATIONALE"),
    ("annexes", "LABEL_ANNEX"),
    ("addenda", "LABEL_ADDENDA"),
]

NAME_BASED_LABELS = frozenset({
    "Concept", "Domain", "Stakeholder", "RiskCategory", "Support", "Sanction",
    "Requirement", "UsageCriterion", "TechCriterion", "Regulation", "LegalSource",
})
ID_BASED_LABELS = frozenset({"Article", "Rationale", "Annex", "Chunk"})


def get_driver():
    uri = os.environ.get("NEO4J_URI", "").strip() or os.environ.get("NEO4J", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip() or os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or uri.startswith("xxx."):
        raise SystemExit(
            "Neo4j connection missing. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env"
        )
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(user, password))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rel_type(rel_type: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", rel_type or "RELATED_TO") or "RELATED_TO"


def _chunked(batch: List[Any], size: int):
    for i in range(0, len(batch), size):
        yield batch[i : i + size]


# ---------------------------------------------------------------------------
# Constraints & Vector Index
# ---------------------------------------------------------------------------
def ensure_constraints_and_index(driver, config: dict) -> None:
    with driver.session() as session:
        struct_labels = [
            config["LABEL_ARTICLE"], config["LABEL_RATIONALE"], config["LABEL_ANNEX"],
            config["LABEL_CHUNK"],
        ]
        if config.get("LABEL_ADDENDA") and config["LABEL_ADDENDA"] not in struct_labels:
            struct_labels.append(config["LABEL_ADDENDA"])
        for label in struct_labels:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE",
            )
        for label in NAME_BASED_LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE",
            )
        chunk_label = config["LABEL_CHUNK"]
        idx_name = config["VECTOR_INDEX_NAME"]
        dim = config["VECTOR_DIMENSIONS"]
        sim = config["VECTOR_SIMILARITY"]
        try:
            session.run(f"""
                CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
                FOR (c:{chunk_label}) ON (c.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dim},
                        `vector.similarity_function`: '{sim}'
                    }}
                }}
            """)
        except Exception as e:
            print(f"  [Vector index] Note: {e}. Ensure Neo4j 5.13+ for vector index.")


def clear_by_source(driver, source_file: str) -> None:
    with driver.session() as session:
        count_result = session.run("MATCH (n) WHERE n.source_file = $sf RETURN count(n) as c", sf=source_file)
        rec = count_result.single()
        n_count = rec["c"] if rec else 0
        if n_count:
            session.run("MATCH (n) WHERE n.source_file = $sf DETACH DELETE n", sf=source_file)
            print(f"  Cleared {n_count} nodes with source_file = {source_file!r}")


def ingest_file(driver, data: dict, source_file: str, config: dict) -> None:
    ts = _ts()
    LART = config["LABEL_ARTICLE"]
    LRAT = config["LABEL_RATIONALE"]
    LANX = config["LABEL_ANNEX"]
    LADD = config["LABEL_ADDENDA"]
    LCHK = config["LABEL_CHUNK"]
    R_HAS = config["REL_HAS_CHUNK"]
    batch_size = config.get("BATCH_SIZE", 100)
    struct_batch_size = config.get("STRUCT_BATCH_SIZE", 40)
    section_labels = [(s, config[k]) for s, k in STRUCTURAL_SECTIONS]
    regulation_name = (data.get("source") or "").strip() or source_file  # "EU AI Act", "Korea AI Law"
    R_CONTAINS = config.get("REL_REGULATION_CONTAINS", "INCLUDES")

    def _node_id(raw_id: str) -> str:
        return f"{regulation_name}::{raw_id}"

    with driver.session() as session:
        # ---- Regulation (top-level law) ----
        session.run(
            "MERGE (r:Regulation { name: $name }) SET r.source_file = $sf, r.updated_at = $ts",
            name=regulation_name, sf=source_file, ts=ts,
        )

        # ---- Step 1: Structural nodes (Article, Rationale, Annex, Addenda) ----
        struct_batch: List[Dict[str, Any]] = []
        section_counts: Dict[str, int] = {}
        for section, label_key in STRUCTURAL_SECTIONS:
            label = config[label_key]
            items = data.get(section) or []
            section_counts[section] = 0
            for item in items:
                nid = item.get("id")
                if nid is None or (isinstance(nid, str) and not nid.strip()):
                    continue
                section_counts[section] += 1
                meta = item.get("metadata") or {}
                full_text = item.get("full_text") or ""
                struct_batch.append({
                    "id": _node_id(nid),
                    "display_id": nid,
                    "label": label,
                    "regulation_name": regulation_name,
                    "source_file": source_file,
                    "updated_at": ts,
                    "full_text": full_text,
                    "metadata": json.dumps(meta),
                    "title": meta.get("title") or item.get("title") or "",
                })
        print(f"  [{source_file}] Regulation {regulation_name!r} sections: {section_counts} -> {len(struct_batch)} structural nodes")

        for label in (LART, LRAT, LANX, LADD):
            batch = [b for b in struct_batch if b["label"] == label]
            if not batch:
                continue
            for chunk in _chunked(batch, struct_batch_size):
                if not chunk:
                    continue
                session.run(f"""
                    UNWIND $batch AS row
                    MERGE (n:{label} {{ id: row.id }})
                    SET n.source_file = row.source_file, n.display_id = row.display_id, n.updated_at = row.updated_at,
                        n.full_text = row.full_text, n.metadata = row.metadata, n.title = row.title
                """, batch=chunk)
            # Link each structural node to its Regulation
            for chunk in _chunked(batch, struct_batch_size):
                session.run(f"""
                    UNWIND $batch AS row
                    MATCH (reg:Regulation {{ name: row.regulation_name }})
                    MATCH (n:{label} {{ id: row.id }})
                    MERGE (reg)-[:{R_CONTAINS}]->(n)
                """, batch=chunk)

        # ---- Step 2: Chunks (with embedding) & HAS_CHUNK ----
        chunk_batch: List[Dict[str, Any]] = []
        has_chunk_batch: List[Dict[str, Any]] = []
        for section, parent_label in section_labels:
            for item in data.get(section) or []:
                parent_id = item.get("id")
                if not parent_id:
                    continue
                parent_node_id = _node_id(parent_id)
                for ch in item.get("chunks") or []:
                    cid = ch.get("chunk_id")
                    text = ch.get("text") or ""
                    emb = ch.get("embedding")
                    if not cid:
                        continue
                    chunk_node_id = _node_id(cid)
                    chunk_batch.append({
                        "id": chunk_node_id,
                        "display_id": cid,
                        "text": text,
                        "embedding": emb,
                        "source_file": source_file,
                        "updated_at": ts,
                    })
                    has_chunk_batch.append({
                        "parent_id": parent_node_id,
                        "parent_label": parent_label,
                        "chunk_id": chunk_node_id,
                    })

        if chunk_batch:
            for c in _chunked(chunk_batch, batch_size):
                session.run(f"""
                    UNWIND $batch AS row
                    MERGE (c:{LCHK} {{ id: row.id }})
                    SET c.source_file = row.source_file, c.updated_at = row.updated_at,
                        c.text = row.text,
                        c.embedding = CASE WHEN row.embedding IS NOT NULL AND size(row.embedding) > 0 THEN row.embedding ELSE c.embedding END
                """, batch=c)
            for plabel in (LART, LRAT, LANX, LADD):
                batch = [{"pid": r["parent_id"], "cid": r["chunk_id"]} for r in has_chunk_batch if r["parent_label"] == plabel]
                for c in _chunked(batch, batch_size):
                    if not c:
                        continue
                    session.run(f"""
                        UNWIND $batch AS row
                        MATCH (p:{plabel} {{ id: row.pid }})
                        MATCH (c:{LCHK} {{ id: row.cid }})
                        MERGE (p)-[:{R_HAS}]->(c)
                    """, batch=c)

        # ---- Step 3: Semantic relations ----
        rel_batches_id: Dict[str, List[Dict[str, Any]]] = {}
        rel_batches_name: Dict[str, List[Dict[str, Any]]] = {}
        for section, parent_label in section_labels:
            for item in data.get(section) or []:
                parent_id = item.get("id")
                if not parent_id:
                    continue
                parent_node_id = _node_id(parent_id)
                for rel in item.get("relations") or []:
                    rtype = _safe_rel_type(rel.get("type") or "RELATES_TO")
                    target_type = (rel.get("target_node_type") or "Concept").strip()
                    target_name = (rel.get("target_node_name") or "").strip()
                    desc = (rel.get("description") or "")[:2000]
                    if not target_name:
                        continue
                    if target_type in ID_BASED_LABELS:
                        key = (parent_label, rtype)
                        rel_batches_id.setdefault(key, []).append({
                            "src_id": parent_node_id, "tgt_id": _node_id(target_name), "desc": desc,
                            "sf": source_file, "ts": ts, "target_type": target_type,
                        })
                    else:
                        tlabel = target_type if target_type in NAME_BASED_LABELS else "Concept"
                        key = (parent_label, rtype, tlabel)
                        rel_batches_name.setdefault(key, []).append({
                            "src_id": parent_node_id, "tgt_name": target_name, "desc": desc,
                            "sf": source_file, "ts": ts,
                        })
        for (parent_label, rtype), batch in rel_batches_id.items():
            if not batch:
                continue
            by_tgt: Dict[str, List[Dict[str, Any]]] = {}
            for row in batch:
                by_tgt.setdefault(row["target_type"], []).append(row)
            for target_type, rows in by_tgt.items():
                for chunk in _chunked(rows, batch_size):
                    session.run(f"""
                        UNWIND $batch AS row
                        MATCH (src:{parent_label} {{ id: row.src_id }})
                        MERGE (tgt:{target_type} {{ id: row.tgt_id }})
                        ON CREATE SET tgt.source_file = row.sf, tgt.updated_at = row.ts
                        ON MATCH SET tgt.updated_at = row.ts
                        MERGE (src)-[r:{rtype}]->(tgt)
                        SET r.description = row.desc, r.source_file = row.sf, r.updated_at = row.ts
                    """, batch=chunk)
        for (parent_label, rtype, tlabel), batch in rel_batches_name.items():
            if not batch:
                continue
            for chunk in _chunked(batch, batch_size):
                session.run(f"""
                    UNWIND $batch AS row
                    MATCH (src:{parent_label} {{ id: row.src_id }})
                    MERGE (tgt:{tlabel} {{ name: row.tgt_name }})
                    ON CREATE SET tgt.source_file = row.sf, tgt.updated_at = row.ts
                    ON MATCH SET tgt.updated_at = row.ts
                    MERGE (src)-[r:{rtype}]->(tgt)
                    SET r.description = row.desc, r.source_file = row.sf, r.updated_at = row.ts
                """, batch=chunk)


def export_schema_summary(driver, out_path: Path) -> None:
    summary: Dict[str, Any] = {
        "exported_at": _ts(),
        "node_labels": {},
        "relationship_types": {},
        "node_counts": {},
    }
    with driver.session() as session:
        r = session.run(
            "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName "
            "WITH nodeLabels, collect(propertyName) AS keys RETURN nodeLabels, keys"
        )
        for rec in r:
            labels = rec["nodeLabels"]
            keys = list(rec["keys"]) if rec.get("keys") else []
            key = "::".join(labels) if labels else "Unknown"
            summary["node_labels"][key] = keys
        r = session.run(
            "CALL db.schema.relTypeProperties() YIELD relType, propertyName "
            "WITH relType, collect(propertyName) AS keys RETURN relType, keys"
        )
        for rec in r:
            summary["relationship_types"][rec["relType"]] = list(rec.get("keys") or [])
        r = session.run("CALL db.labels() YIELD label RETURN label")
        for rec in r:
            lab = rec["label"]
            c = session.run(f"MATCH (n:{lab}) RETURN count(n) as c").single()
            summary["node_counts"][lab] = c["c"] if c else 0
    save_json(out_path, summary)
    print(f"Schema summary written to {out_path}")


def main() -> None:
    import argparse
    processed = _project_root / "data" / "processed"
    parser = argparse.ArgumentParser(description="Ingest legal graph JSON into Neo4j.")
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=[
            processed / "AICoreLaw_relations_ready.json",
            processed / "AIAct_relations_ready.json",
        ],
        help="Input *_relations_ready.json files",
    )
    parser.add_argument("--clear-by-source", action="store_true", help="Delete nodes with matching source_file before ingest")
    parser.add_argument("--schema-only", action="store_true", help="Only export schema summary")
    args = parser.parse_args()

    config = dict(CONFIG)
    config["CLEAR_DB_BY_SOURCE"] = args.clear_by_source

    driver = get_driver()
    try:
        ensure_constraints_and_index(driver, config)
        if args.schema_only:
            export_schema_summary(driver, processed / "graph_schema_summary.json")
            return
        for path in args.files:
            if not path.is_absolute():
                path = _project_root / path
            if not path.exists():
                print(f"Skipping (not found): {path}")
                continue
            data = load_json(path)
            source_file = path.name
            if config["CLEAR_DB_BY_SOURCE"]:
                clear_by_source(driver, source_file)
            ingest_file(driver, data, source_file, config)
            print(f"Ingested: {path.name}")
        export_schema_summary(driver, processed / "graph_schema_summary.json")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
