"""
Legal Knowledge Graph Audit Script
Validates integrity of integrated graph (KR AI Core Law & EU AI Act).
Deep Diagnostic Engine + Self-Healing: Audit, Diagnose, and Heal phantoms.
Senior Graph Data Engineer: Ling Long
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

JURISDICTION_MAP = {"Korea AI Law": "KR", "EU AI Act": "EU"}
UNKNOWN_JURISDICTION = "OTHER"

# Regulation name -> source JSON file
REGULATION_TO_JSON = {
    "Korea AI Law": "AICoreLaw_relations_ready.json",
    "EU AI Act": "AIAct_relations_ready.json",
}

# Root-cause classification codes
CAUSE_SOURCE_MISSING = "SOURCE_MISSING"
CAUSE_CHUNK_EMPTY = "CHUNK_EMPTY"
CAUSE_EMBEDDING_MISSING = "EMBEDDING_MISSING"
CAUSE_INGESTION_FAILURE = "INGESTION_FAILURE"
CAUSE_EXTERNAL_REF = "EXTERNAL_REF_NO_CONTENT"

# Self-healing: label for external refs we keep but tag
REFERENCE_ONLY_LABEL = "ReferenceOnly"

# Future-proof: Cypher snippet for ingest_to_neo4j.py to avoid creating phantoms.
# Run this logic when creating semantic-relation targets: if target_node_name is
# comma-separated (e.g. "Articles 9, 10, 11"), split and MERGE each ID separately
# instead of creating one node with that composite name.
CYPHER_SNIPPET_FOR_INGEST = """
-- In ingest_to_neo4j.py (semantic relations): when target_node_name is comma-separated
-- (e.g. "Articles 9, 10, 11"), split into ["Article 9", "Article 10", "Article 11"]
-- and MERGE each as separate node + create one relation per split. Do not create one node
-- with the composite name. Use same pattern as audit_graph_data._split_phantom_display_id().
"""


def get_driver():
    uri = os.environ.get("NEO4J_URI", "").strip() or os.environ.get("NEO4J", "").strip()
    user = os.environ.get("NEO4J_USERNAME", "").strip() or os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or uri.startswith("xxx."):
        raise SystemExit("Neo4j connection missing. Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env")
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


def _is_external_ref(display_id: str) -> bool:
    """Detect if the ID refers to an external regulation (Directive, other Regulation, TEU)."""
    if not display_id:
        return False
    s = display_id.lower()
    return any(x in s for x in [
        "directive", "regulation (eu)", "regulation (ec)", "regulation (eec)",
        "teu", "tfeu", "no 182/2011", "no 765/2008", "no 1025/2012", "no 2019/1020",
        "2014/90/eu", "2016/797", "2016/680", "of directive", "of regulation",
    ])


def _guess_section(display_id: str, label: str) -> str:
    """Guess JSON section key from display_id/label."""
    if not display_id:
        return "articles"
    d = display_id.lower()
    if "recital" in d or label == "Rationale":
        return "rationale"
    if "annex" in d or label == "Annex":
        return "annexes"
    if "addenda" in d:
        return "addenda"
    return "articles"


def _build_json_index(processed_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """Build lookup: (regulation_name, display_id) -> item from _ready.json files."""
    index: Dict[Tuple[str, str], Dict] = {}
    for reg_name, fname in REGULATION_TO_JSON.items():
        path = processed_dir / fname
        if not path.exists():
            continue
        data = load_json(path)
        for section in ("articles", "rationale", "annexes", "addenda"):
            for item in data.get(section) or []:
                nid = item.get("id")
                if nid:
                    index[(reg_name, nid)] = {"section": section, "item": item}
    return index


def _classify_missing_chunk(
    reg_name: str, display_id: str, label: str, json_index: Dict[Tuple[str, str], Dict]
) -> Tuple[str, str]:
    """
    Classify root cause for missing HAS_CHUNK.
    Returns (cause_code, detail).
    """
    if _is_external_ref(display_id):
        return (CAUSE_EXTERNAL_REF, "External reference; no source content expected")
    key = (reg_name, display_id)
    if key not in json_index:
        return (CAUSE_SOURCE_MISSING, "Not found in source JSON")
    entry = json_index.get(key)
    if not entry:
        return (CAUSE_SOURCE_MISSING, "Not found in source JSON")
    item = entry["item"]
    chunks = item.get("chunks") or []
    if not chunks:
        return (CAUSE_CHUNK_EMPTY, "chunks list is empty (parsing/min_length issue)")
    has_embedding = any(
        bool(c.get("embedding")) and isinstance(c.get("embedding"), list) and len(c.get("embedding", [])) > 0
        for c in chunks
    )
    if not has_embedding:
        return (CAUSE_EMBEDDING_MISSING, f"{len(chunks)} chunk(s) exist but no embedding")
    return (CAUSE_INGESTION_FAILURE, "JSON has chunks+embeddings; likely ID mapping or ingest logic issue")


def run_deep_diagnostic(
    session, processed_dir: Path, structural_without_chunk: List[Dict]
) -> Dict[str, Any]:
    """Deep trace: cross-reference Neo4j missing nodes with source JSON; classify root cause."""
    json_index = _build_json_index(processed_dir)
    results: List[Dict] = []
    classification_counts: Dict[str, int] = {}

    try:
        from tqdm import tqdm
        iterator = tqdm(structural_without_chunk, desc="Diagnosing missing HAS_CHUNK", unit="node")
    except ImportError:
        iterator = structural_without_chunk

    for node in iterator:
        nid = node.get("id") or ""
        display_id = node.get("display_id") or nid.split("::", 1)[-1] if "::" in nid else nid
        reg_name = nid.split("::", 1)[0] if "::" in nid else ""
        label = node.get("label") or "Article"
        cause, detail = _classify_missing_chunk(reg_name, display_id, label, json_index)
        classification_counts[cause] = classification_counts.get(cause, 0) + 1
        results.append({
            "id": nid, "display_id": display_id, "label": label, "regulation": reg_name,
            "cause": cause, "detail": detail,
        })

    return {
        "total_diagnosed": len(results),
        "classification_counts": classification_counts,
        "results": results,
        "recommendations": _get_recommendations(classification_counts),
    }


def _get_recommendations(classification_counts: Dict[str, int]) -> List[str]:
    """Self-healing action plan based on root-cause distribution."""
    recs: List[str] = []
    if classification_counts.get(CAUSE_SOURCE_MISSING, 0) > 0:
        recs.append(
            "SOURCE_MISSING: These nodes (often external refs created by relations) are not in source JSON. "
            "Expected for references like 'Article 8 of Directive 2014/90/EU'. No action if intentional."
        )
    if classification_counts.get(CAUSE_CHUNK_EMPTY, 0) > 0:
        recs.append(
            "CHUNK_EMPTY: Run parsing pipeline with lower min_length or adjust RecursiveCharacterTextSplitter. "
            "Check ingest_EU_legal.py / ingest_KR_legal.py chunking parameters."
        )
    if classification_counts.get(CAUSE_EMBEDDING_MISSING, 0) > 0:
        recs.append(
            "EMBEDDING_MISSING: Run add_embeddings.py (or equivalent) to populate chunk embeddings before ingest."
        )
    if classification_counts.get(CAUSE_INGESTION_FAILURE, 0) > 0:
        recs.append(
            "INGESTION_FAILURE: Data is correct in JSON but HAS_CHUNK missing in Neo4j. "
            "Check ingest_to_neo4j.py: ID mapping (composite id vs chunk_id), HAS_CHUNK MERGE logic."
        )
    if classification_counts.get(CAUSE_EXTERNAL_REF, 0) > 0:
        recs.append(
            "EXTERNAL_REF_NO_CONTENT: External references; no source content. Not critical for RAG."
        )
    return recs


def _split_phantom_display_id(display_id: str, label: str) -> List[str]:
    """
    Split composite phantom IDs like 'Articles 9, 10, 11' into ['Article 9', 'Article 10', 'Article 11'].
    Returns list of normalized display IDs for entity resolution.
    """
    if not display_id or not display_id.strip():
        return []
    raw = display_id.strip()
    parts = [p.strip() for p in re.split(r",|\s+and\s+", raw, flags=re.I) if p.strip()]
    if not parts:
        return [raw]
    normalized: List[str] = []
    prefix = "Article"
    if raw.lower().startswith("recital"):
        prefix = "Recital"
    elif raw.lower().startswith("annex"):
        prefix = "Annex"
    first = parts[0]
    if re.match(r"^(?:Article|Articles)s?\s+\d+", first, re.I):
        prefix = "Article" if "Article" in first[:10] else prefix
    elif re.match(r"^Recitals?\s+\d+", first, re.I):
        prefix = "Recital"
    for i, p in enumerate(parts):
        if re.match(r"^\d+$", p):
            normalized.append(f"{prefix} {p}")
        elif re.match(r"^(?:Article|Articles)s?\s+\d+", p, re.I):
            normalized.append(re.sub(r"^Articles?\s+", "Article ", p, flags=re.I))
        elif re.match(r"^Recitals?\s+\d+", p, re.I):
            normalized.append(re.sub(r"^Recitals?\s+", "Recital ", p, flags=re.I))
        else:
            normalized.append(p)
    return normalized if len(normalized) > 1 else [raw]


def run_self_healing(session, deep_diagnostic: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrated Healing Engine: re-link SOURCE_MISSING phantoms to real nodes, tag EXTERNAL_REF, cleanup.
    """
    results_list = deep_diagnostic.get("results", [])
    phantoms_source = [r for r in results_list if r.get("cause") == CAUSE_SOURCE_MISSING]
    phantoms_external = [r for r in results_list if r.get("cause") == CAUSE_EXTERNAL_REF]

    total_phantoms = len(phantoms_source) + len(phantoms_external)
    relinked = 0
    bridged_ids: List[str] = []
    remaining_ghosts: List[Dict] = []

    # 1) Tag EXTERNAL_REF nodes with :ReferenceOnly (do not delete)
    for r in phantoms_external:
        phantom_id = r.get("id")
        if not phantom_id:
            continue
        try:
            session.run(
                "MATCH (n) WHERE n.id = $id SET n:ReferenceOnly RETURN count(n) AS c",
                id=phantom_id,
            )
        except Exception:
            pass

    # 2) SOURCE_MISSING: split, find real nodes, migrate relationships, delete phantom
    for r in phantoms_source:
        phantom_id = r.get("id")
        display_id = r.get("display_id") or (phantom_id.split("::", 1)[-1] if "::" in phantom_id else phantom_id)
        reg_name = phantom_id.split("::", 1)[0] if "::" in phantom_id else ""
        label = r.get("label") or "Article"

        split_ids = _split_phantom_display_id(display_id, label)
        real_node_ids: List[str] = []
        for sid in split_ids:
            candidate_id = f"{reg_name}::{sid}" if reg_name else sid
            rec = session.run(
                "MATCH (n) WHERE n.id = $id RETURN n.id AS id",
                id=candidate_id,
            ).single()
            if rec and rec.get("id"):
                real_node_ids.append(rec["id"])

        if not real_node_ids:
            remaining_ghosts.append(r)
            continue

        # Fetch all relationships (outgoing and incoming) with type and properties
        out_rels = session.run("""
            MATCH (phantom)-[r]->(other) WHERE phantom.id = $pid
            RETURN type(r) AS rel_type, properties(r) AS props, elementId(other) AS other_eid
        """, pid=phantom_id)
        in_rels = session.run("""
            MATCH (other)-[r]->(phantom) WHERE phantom.id = $pid
            RETURN type(r) AS rel_type, properties(r) AS props, elementId(other) AS other_eid
        """, pid=phantom_id)
        out_list = [(rec["rel_type"], rec["props"] or {}, rec["other_eid"]) for rec in out_rels]
        in_list = [(rec["rel_type"], rec["props"] or {}, rec["other_eid"]) for rec in in_rels]

        def safe_type(t: str) -> str:
            return re.sub(r"[^A-Za-z0-9_]", "_", t or "RELATED_TO") or "RELATED_TO"

        for real_id in real_node_ids:
            for rel_type, props, other_eid in out_list:
                rt = safe_type(rel_type)
                session.run(
                    f"MATCH (other) WHERE elementId(other) = $oeid MATCH (real) WHERE real.id = $rid "
                    f"CREATE (real)-[r2:{rt}]->(other) SET r2 = $props",
                    oeid=other_eid, rid=real_id, props=props,
                )
            for rel_type, props, other_eid in in_list:
                rt = safe_type(rel_type)
                session.run(
                    f"MATCH (other) WHERE elementId(other) = $oeid MATCH (real) WHERE real.id = $rid "
                    f"CREATE (other)-[r2:{rt}]->(real) SET r2 = $props",
                    oeid=other_eid, rid=real_id, props=props,
                )

        # Delete phantom and its relationships
        session.run("MATCH (n) WHERE n.id = $id DETACH DELETE n", id=phantom_id)
        relinked += 1
        bridged_ids.append(f"'{display_id}' -> {', '.join(real_node_ids)}")

    return {
        "total_phantoms_detected": total_phantoms,
        "successfully_relinked": relinked,
        "remaining_ghost_nodes": len(remaining_ghosts),
        "bridged_ids": bridged_ids,
        "remaining_ghost_details": remaining_ghosts[:50],
        "external_ref_tagged_count": len(phantoms_external),
        "cypher_snippet_for_ingest": CYPHER_SNIPPET_FOR_INGEST.strip(),
    }


def _tabulate_rows(rows: List[List[Any]], headers: List[str]) -> str:
    if not rows:
        return "  (none)"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(f"{str(h):{w}}" for h, w in zip(headers, widths)), sep]
    for row in rows:
        lines.append("  ".join(f"{str(c):{w}}" for c, w in zip(row, widths)))
    return "\n".join(lines)


def run_schema_compliance(session, schema_path: Path) -> Dict[str, Any]:
    schema = load_json(schema_path) if schema_path.exists() else {}
    summary = schema.get("node_labels", {})
    rel_types_schema = schema.get("relationship_types", {})
    schema_labels = set(summary.keys())
    schema_rel_types = set(k.strip("`:") for k in rel_types_schema.keys())
    result = session.run("CALL db.labels() YIELD label RETURN label")
    actual_labels = {r["label"] for r in result}
    result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    actual_rel_types = {r["relationshipType"] for r in result}
    undeclared_labels = actual_labels - schema_labels
    missing_labels = schema_labels - actual_labels
    undeclared_rel_types = actual_rel_types - schema_rel_types
    missing_rel_types = schema_rel_types - actual_rel_types
    return {
        "schema_file": str(schema_path),
        "schema_exported_at": schema.get("exported_at"),
        "actual_labels": sorted(actual_labels),
        "schema_labels": sorted(schema_labels),
        "undeclared_labels": sorted(undeclared_labels),
        "missing_labels": sorted(missing_labels),
        "undeclared_rel_types": sorted(undeclared_rel_types),
        "missing_rel_types": sorted(missing_rel_types),
        "compliant": len(undeclared_labels) == 0 and len(missing_labels) == 0,
    }


def run_statistical_summary(session) -> Dict[str, Any]:
    result = session.run("MATCH (r:Regulation) RETURN r.name AS name ORDER BY r.name")
    regulations = [r["name"] for r in result]
    node_counts = {"KR": 0, "EU": 0, "OTHER": 0}
    result = session.run("""
        MATCH (r:Regulation)-[:INCLUDES]->(n)
        RETURN r.name AS reg_name, count(n) AS c
    """)
    for rec in result:
        jur = JURISDICTION_MAP.get(rec["reg_name"], UNKNOWN_JURISDICTION)
        node_counts[jur] = node_counts.get(jur, 0) + rec["c"]
    result = session.run("MATCH (n) WHERE NOT ()-[:INCLUDES]->(n) RETURN count(n) AS c")
    node_counts["UNLINKED"] = result.single()["c"]
    total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    result = session.run("""
        MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS c ORDER BY c DESC
    """)
    rel_distribution = {r["rel_type"]: r["c"] for r in result}
    result = session.run("""
        MATCH (n)
        WITH n, count { (n)--() } AS degree
        WHERE degree > 5
        RETURN labels(n)[0] AS label,
               coalesce(n.name, n.display_id, n.id, toString(id(n))) AS identifier,
               degree
        ORDER BY degree DESC
        LIMIT 20
    """)
    high_density = [{"label": r["label"], "identifier": r["identifier"], "degree": r["degree"]} for r in result]
    return {
        "regulations": regulations,
        "jurisdiction_node_counts": node_counts,
        "total_nodes": total_nodes,
        "total_relationships": total_rels,
        "relationship_distribution": rel_distribution,
        "high_density_nodes": high_density,
    }


def run_cross_jurisdiction_analysis(session) -> Dict[str, Any]:
    result = session.run("""
        MATCH (c:Concept) WHERE c.name IS NOT NULL
        WITH c, c.name AS concept_name
        OPTIONAL MATCH (c)<--(kr) WHERE kr.id STARTS WITH 'Korea AI Law::'
        WITH c, concept_name, count(DISTINCT kr) AS kr_links
        OPTIONAL MATCH (c)<--(eu) WHERE eu.id STARTS WITH 'EU AI Act::'
        WITH concept_name, kr_links, count(DISTINCT eu) AS eu_links
        WHERE kr_links > 0 AND eu_links > 0
        RETURN concept_name, kr_links, eu_links ORDER BY kr_links + eu_links DESC
    """)
    shared_concepts = [{"name": r["concept_name"], "kr_links": r["kr_links"], "eu_links": r["eu_links"]} for r in result]
    result = session.run("""
        MATCH (d:Domain) WHERE d.name IS NOT NULL
        WITH d, d.name AS domain_name
        OPTIONAL MATCH (d)<--(kr) WHERE kr.id STARTS WITH 'Korea AI Law::'
        WITH d, domain_name, count(DISTINCT kr) AS kr_links
        OPTIONAL MATCH (d)<--(eu) WHERE eu.id STARTS WITH 'EU AI Act::'
        WITH domain_name, kr_links, count(DISTINCT eu) AS eu_links
        WHERE kr_links > 0 AND eu_links > 0
        RETURN domain_name, kr_links, eu_links ORDER BY kr_links + eu_links DESC
    """)
    shared_domains = [{"name": r["domain_name"], "kr_links": r["kr_links"], "eu_links": r["eu_links"]} for r in result]
    result = session.run("""
        MATCH (s:Stakeholder) WHERE s.name IS NOT NULL
        WITH s, s.name AS stakeholder_name
        OPTIONAL MATCH (s)<--(kr) WHERE kr.id STARTS WITH 'Korea AI Law::'
        WITH s, stakeholder_name, count(DISTINCT kr) AS kr_links
        OPTIONAL MATCH (s)<--(eu) WHERE eu.id STARTS WITH 'EU AI Act::'
        WITH stakeholder_name, kr_links, count(DISTINCT eu) AS eu_links
        WHERE kr_links > 0 AND eu_links > 0
        RETURN stakeholder_name, kr_links, eu_links ORDER BY kr_links + eu_links DESC
    """)
    shared_stakeholders = [{"name": r["stakeholder_name"], "kr_links": r["kr_links"], "eu_links": r["eu_links"]} for r in result]
    return {
        "shared_concepts_count": len(shared_concepts),
        "shared_concepts": shared_concepts[:50],
        "shared_domains_count": len(shared_domains),
        "shared_domains": shared_domains[:30],
        "shared_stakeholders_count": len(shared_stakeholders),
        "shared_stakeholders": shared_stakeholders[:30],
    }


def run_integrity_checks(session) -> Dict[str, Any]:
    result = session.run("""
        MATCH (n) WHERE NOT (n)--()
        RETURN labels(n)[0] AS label, count(n) AS c ORDER BY c DESC
    """)
    orphans_by_label = [{"label": r["label"], "count": r["c"]} for r in result]
    # All structural nodes without HAS_CHUNK (Article, Rationale, Annex)
    result = session.run("""
        MATCH (n)
        WHERE (n:Article OR n:Rationale OR n:Annex) AND NOT (n)-[:HAS_CHUNK]->()
        RETURN labels(n)[0] AS label, n.id AS id, n.display_id AS display_id
    """)
    structural_without_chunk = [
        {"label": r["label"], "id": r["id"], "display_id": r.get("display_id")} for r in result
    ]
    result = session.run("""
        MATCH (a:Article) WHERE NOT (a)-[:HAS_CHUNK]->() RETURN count(a) AS c
    """)
    articles_without_chunk_count = result.single()["c"]
    return {
        "orphan_nodes_by_label": orphans_by_label,
        "orphan_total": sum(o["count"] for o in orphans_by_label),
        "articles_without_has_chunk_count": articles_without_chunk_count,
        "structural_without_chunk": structural_without_chunk,
        "articles_without_has_chunk_sample": structural_without_chunk[:20],
    }


def run_audit(driver, schema_path: Path, out_dir: Path) -> Dict[str, Any]:
    report = {
        "audit_timestamp": _ts(),
        "schema_compliance": {},
        "statistical_summary": {},
        "cross_jurisdiction": {},
        "integrity": {},
        "deep_diagnostic": {},
    }
    with driver.session() as session:
        report["schema_compliance"] = run_schema_compliance(session, schema_path)
        report["statistical_summary"] = run_statistical_summary(session)
        report["cross_jurisdiction"] = run_cross_jurisdiction_analysis(session)
        report["integrity"] = run_integrity_checks(session)
        structural = report["integrity"].get("structural_without_chunk", [])
        if structural:
            report["deep_diagnostic"] = run_deep_diagnostic(session, out_dir, structural)
            if report["deep_diagnostic"].get("results"):
                report["self_healing"] = run_self_healing(session, report["deep_diagnostic"])
    return report


def print_report(report: Dict[str, Any]) -> None:
    ts = report.get("audit_timestamp", "")
    print("\n" + "=" * 70)
    print("  LEGAL KNOWLEDGE GRAPH AUDIT REPORT")
    print(f"  Timestamp: {ts}")
    print("=" * 70)
    sc = report.get("schema_compliance", {})
    print("\n--- 1. SCHEMA COMPLIANCE ---")
    print(f"  Schema file: {sc.get('schema_file', 'N/A')}")
    print(f"  Compliant: {sc.get('compliant', False)}")
    if sc.get("undeclared_labels"):
        print(f"  Undeclared labels: {sc['undeclared_labels']}")
    if sc.get("missing_labels"):
        print(f"  Missing labels: {sc['missing_labels']}")
    if sc.get("undeclared_rel_types"):
        print(f"  Undeclared relationship types: {sc['undeclared_rel_types']}")
    ss = report.get("statistical_summary", {})
    print("\n--- 2. STATISTICAL SUMMARY ---")
    print(f"  Total nodes: {ss.get('total_nodes', 0)}")
    print(f"  Total relationships: {ss.get('total_relationships', 0)}")
    print(f"  Nodes by jurisdiction: {ss.get('jurisdiction_node_counts', {})}")
    print("\n  Relationship type distribution:")
    for rt, cnt in list(ss.get("relationship_distribution", {}).items())[:25]:
        print(f"    {rt}: {cnt}")
    print("\n  High-density nodes (top 10):")
    rows = [[h["label"], (h["identifier"] or "")[:50], h["degree"]] for h in ss.get("high_density_nodes", [])[:10]]
    print(_tabulate_rows(rows, ["Label", "Identifier", "Degree"]))
    cj = report.get("cross_jurisdiction", {})
    print("\n--- 3. CROSS-JURISDICTION BRIDGE CHECK ---")
    print(f"  Shared Concept nodes (KR & EU): {cj.get('shared_concepts_count', 0)}")
    print(f"  Shared Domain nodes: {cj.get('shared_domains_count', 0)}")
    print(f"  Shared Stakeholder nodes: {cj.get('shared_stakeholders_count', 0)}")
    if cj.get("shared_concepts"):
        print("  Sample shared concepts:")
        for c in cj["shared_concepts"][:10]:
            print(f"    - {c['name']} (KR: {c['kr_links']}, EU: {c['eu_links']})")
    ig = report.get("integrity", {})
    print("\n--- 4. INTEGRITY & ORPHAN SEARCH ---")
    print(f"  Orphan nodes total: {ig.get('orphan_total', 0)}")
    if ig.get("orphan_nodes_by_label"):
        for o in ig["orphan_nodes_by_label"]:
            print(f"    {o['label']}: {o['count']}")
    print(f"  Structural nodes without HAS_CHUNK: {ig.get('articles_without_has_chunk_count', 0)}")
    if ig.get("articles_without_has_chunk_sample"):
        for a in ig["articles_without_has_chunk_sample"][:10]:
            print(f"    - {a.get('display_id') or a.get('id')}")

    dd = report.get("deep_diagnostic", {})
    if dd:
        print("\n--- 5. DEEP DIAGNOSTIC (Root Cause) ---")
        print(f"  Total diagnosed: {dd.get('total_diagnosed', 0)}")
        for cause, cnt in sorted(dd.get("classification_counts", {}).items(), key=lambda x: -x[1]):
            print(f"    {cause}: {cnt}")
        print("  Action plan:")
        for r in dd.get("recommendations", []):
            print(f"    -> {r}")
        print("  Sample by cause:")
        by_cause: Dict[str, List[Dict]] = {}
        for r in dd.get("results", []):
            by_cause.setdefault(r["cause"], []).append(r)
        for cause in (CAUSE_INGESTION_FAILURE, CAUSE_EMBEDDING_MISSING, CAUSE_CHUNK_EMPTY, CAUSE_SOURCE_MISSING, CAUSE_EXTERNAL_REF):
            if cause in by_cause:
                samples = by_cause[cause][:5]
                print(f"    {cause}: {[s['display_id'] for s in samples]}")

    sh = report.get("self_healing", {})
    if sh:
        print("\n--- 6. SELF-HEALING (Fixer) ---")
        print(f"  Total Phantoms Detected: {sh.get('total_phantoms_detected', 0)}")
        print(f"  Successfully Re-linked: {sh.get('successfully_relinked', 0)}")
        print(f"  Remaining Ghost Nodes: {sh.get('remaining_ghost_nodes', 0)}")
        print(f"  External refs tagged :ReferenceOnly: {sh.get('external_ref_tagged_count', 0)}")
        if sh.get("bridged_ids"):
            print("  Bridged IDs:")
            for b in sh["bridged_ids"][:20]:
                print(f"    - {b}")
        if sh.get("cypher_snippet_for_ingest"):
            print("  Future-proof (ingest_to_neo4j):")
            print("    " + sh["cypher_snippet_for_ingest"].replace("\n", "\n    "))
    print("\n" + "=" * 70)


def main() -> None:
    schema_path = _project_root / "data" / "processed" / "graph_schema_summary.json"
    out_dir = _project_root / "data" / "processed"
    driver = get_driver()
    try:
        report = run_audit(driver, schema_path, out_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"audit_report_{timestamp}.json"
        save_json(out_path, report)
        print(f"\nAudit report saved to: {out_path}")
        print_report(report)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
