"""Memory grooming: category正規化, entity再抽出, GraphDB再構築, 孤立Entity掃除."""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyarrow as pa

from engram.db import get_table, delete_records, insert_records, TABLE_NAME

logger = logging.getLogger(__name__)

# Phase 1: Category 正規化マッピング
CATEGORY_MAP = {
    # 統合対象
    "troubleshooting": "debugging",
    "backend": "implementation",
    "system-design": "architecture",
    "api-design": "architecture",
    "cloud": "infrastructure",
    "planning": "workflow",
    "documentation": "workflow",
    # そのまま維持
    "debugging": "debugging",
    "architecture": "architecture",
    "design-decision": "design-decision",
    "implementation": "implementation",
    "performance": "performance",
    "configuration": "configuration",
    "workflow": "workflow",
    "convention": "convention",
    "testing": "testing",
    "tooling": "tooling",
    "database": "database",
    "security": "security",
    "operations": "operations",
    "ai-integration": "ai-integration",
    "observability": "observability",
    "infrastructure": "infrastructure",
    "frontend": "frontend",
}


def _load_all_records(db_path: str) -> pa.Table:
    """VectorDB の全レコードを PyArrow Table として取得する。"""
    table = get_table(db_path)
    return table.to_arrow()


def _record_to_dict(arrow_table: pa.Table, idx: int) -> Dict[str, Any]:
    """PyArrow Table の1行を辞書に変換する。"""
    result = {}
    for col_name in arrow_table.column_names:
        val = arrow_table.column(col_name)[idx].as_py()
        result[col_name] = val
    return result


# ──────────────────────────────────────────────
# Phase 1: Category 正規化
# ──────────────────────────────────────────────

def analyze_categories(db_path: str) -> Dict[str, Any]:
    """Phase 1 dry-run: 正規化対象の category を分析する。"""
    arrow = _load_all_records(db_path)
    categories = [arrow.column("category")[i].as_py() for i in range(len(arrow))]

    to_rename: Dict[str, Tuple[str, int]] = {}  # old -> (new, count)
    unknown: Dict[str, int] = {}

    for cat in categories:
        if cat in CATEGORY_MAP:
            new_cat = CATEGORY_MAP[cat]
            if cat != new_cat:
                to_rename.setdefault(cat, (new_cat, 0))
                to_rename[cat] = (new_cat, to_rename[cat][1] + 1)
        else:
            unknown[cat] = unknown.get(cat, 0) + 1

    return {
        "total": len(categories),
        "to_rename": {k: {"to": v[0], "count": v[1]} for k, v in to_rename.items()},
        "unknown": unknown,
    }


def normalize_categories(db_path: str) -> Dict[str, int]:
    """Phase 1: category を正規マッピングに従って書き換える。

    Returns:
        {"renamed": N, "skipped_unknown": N}
    """
    table = get_table(db_path)
    arrow = table.to_arrow()

    renamed = 0
    skipped_unknown = 0

    for i in range(len(arrow)):
        rec = _record_to_dict(arrow, i)
        old_cat = rec.get("category", "")
        if old_cat not in CATEGORY_MAP:
            logger.warning("Unknown category '%s' for id=%s, skipping", old_cat, rec.get("id"))
            skipped_unknown += 1
            continue

        new_cat = CATEGORY_MAP[old_cat]
        if old_cat == new_cat:
            continue

        # delete + re-insert で更新
        record_id = rec["id"]
        rec["category"] = new_cat
        del rec["vector"]  # 再埋め込み不要、既存vectorを使う

        # vectorを元データから取得
        vec = arrow.column("vector")[i].as_py()
        rec["vector"] = vec

        delete_records([record_id], db_path)
        insert_records([rec], db_path)
        renamed += 1

    return {"renamed": renamed, "skipped_unknown": skipped_unknown}


# ──────────────────────────────────────────────
# Phase 2: Entity/Relation 再抽出
# ──────────────────────────────────────────────

def analyze_entities(db_path: str) -> Dict[str, Any]:
    """Phase 2 dry-run: entities_json の状態を分析する。"""
    arrow = _load_all_records(db_path)
    total = len(arrow)
    empty = 0

    for i in range(total):
        ej = arrow.column("entities_json")[i].as_py()
        try:
            entities = json.loads(ej) if ej else []
        except (json.JSONDecodeError, TypeError):
            entities = []
        if not entities:
            empty += 1

    return {"total": total, "empty_entities": empty, "to_re_extract": total}


def re_extract_entities(
    db_path: str,
    llm_fn: Callable,
    batch_size: int = 5,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    """Phase 2: 全メモリの entities/relations を LLM で再抽出する。

    Args:
        db_path: LanceDB パス
        llm_fn: messages を受け取り文字列を返す LLM 関数
        batch_size: 1回の LLM 呼び出しで処理するメモリ件数
        progress_fn: (processed, total) を受け取るコールバック

    Returns:
        {"processed": N, "updated": N, "errors": N}
    """
    from engram.prompts_groom import build_entity_extraction_prompt

    table = get_table(db_path)
    arrow = table.to_arrow()
    total = len(arrow)

    stats = {"processed": 0, "updated": 0, "errors": 0}

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_mems = []

        for i in range(batch_start, batch_end):
            rec = _record_to_dict(arrow, i)
            batch_mems.append(rec)

        # LLM 呼び出し
        messages = build_entity_extraction_prompt(batch_mems)
        try:
            response = llm_fn(messages)
            extractions = _parse_extraction_response(response, batch_mems)
        except Exception as e:
            logger.error("LLM call failed for batch %d-%d: %s", batch_start, batch_end, e)
            stats["errors"] += batch_end - batch_start
            stats["processed"] += batch_end - batch_start
            if progress_fn:
                progress_fn(stats["processed"], total)
            continue

        # 結果を VectorDB に書き戻す
        for batch_idx, (mem, extraction) in enumerate(zip(batch_mems, extractions)):
            if extraction is None:
                stats["errors"] += 1
                stats["processed"] += 1
                continue

            new_entities = extraction.get("entities", [])
            new_relations = extraction.get("relations", [])

            record_id = mem["id"]
            mem["entities_json"] = json.dumps(new_entities, ensure_ascii=False)
            mem["relations_json"] = json.dumps(new_relations, ensure_ascii=False)

            # vector は既存のものを維持
            vec = arrow.column("vector")[batch_start + batch_idx].as_py()
            mem["vector"] = vec

            try:
                delete_records([record_id], db_path)
                insert_records([mem], db_path)
                stats["updated"] += 1
            except Exception as e:
                logger.error("Failed to update record %s: %s", record_id, e)
                stats["errors"] += 1

            stats["processed"] += 1

        if progress_fn:
            progress_fn(stats["processed"], total)

    return stats


def _parse_extraction_response(
    response: str,
    batch_mems: List[Dict],
) -> List[Optional[Dict]]:
    """LLM レスポンスからJSON配列を抽出し、各メモリにマッチさせる。"""
    # JSON配列を探す
    start = response.find("[")
    if start == -1:
        raise ValueError("No JSON array found in response")

    depth = 0
    for i in range(start, len(response)):
        if response[i] == "[":
            depth += 1
        elif response[i] == "]":
            depth -= 1
            if depth == 0:
                parsed = json.loads(response[start : i + 1])
                break
    else:
        raise ValueError("Incomplete JSON array in response")

    # id でマッチング
    id_to_extraction = {item["id"]: item for item in parsed if "id" in item}

    result: List[Optional[Dict]] = []
    for mem in batch_mems:
        mem_id = mem.get("id", "")
        extraction = id_to_extraction.get(mem_id)
        if extraction is None:
            # 順序ベースのフォールバック
            idx = batch_mems.index(mem)
            if idx < len(parsed):
                extraction = parsed[idx]
        result.append(extraction)

    return result


# ──────────────────────────────────────────────
# Phase 3: GraphDB 再構築
# ──────────────────────────────────────────────

def rebuild_graph(
    db_path: str,
    graph_path: str,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    """Phase 3: VectorDB をソースとして GraphDB を完全再構築する。

    Returns:
        {"total": N, "synced": N, "errors": N}
    """
    import datetime
    from engram.graph import sync_to_graph, close_graph_db

    # キャッシュをクリア（ロック解放のため先に閉じる）
    close_graph_db(graph_path)

    # 既存 GraphDB をバックアップ＆削除
    if os.path.exists(graph_path):
        backup_path = graph_path + ".bak"
        if os.path.exists(backup_path):
            if os.path.isdir(backup_path):
                shutil.rmtree(backup_path)
            else:
                os.remove(backup_path)
        if os.path.isdir(graph_path):
            shutil.copytree(graph_path, backup_path)
            shutil.rmtree(graph_path)
        else:
            shutil.copy2(graph_path, backup_path)
            os.remove(graph_path)
        logger.info("Backed up old graph to %s", backup_path)

    arrow = _load_all_records(db_path)
    total = len(arrow)
    stats = {"total": total, "synced": 0, "errors": 0}

    for i in range(total):
        rec = _record_to_dict(arrow, i)
        memory_id = rec.get("id", "")

        try:
            entities = json.loads(rec.get("entities_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            entities = []

        try:
            relations = json.loads(rec.get("relations_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            relations = []

        if not entities:
            stats["synced"] += 1
            if progress_fn:
                progress_fn(i + 1, total)
            continue

        ts = rec.get("timestamp")
        if ts is None:
            ts = datetime.datetime.now()
        elif isinstance(ts, str):
            ts = datetime.datetime.fromisoformat(ts)

        try:
            sync_to_graph(
                memory_id=memory_id,
                event=rec.get("event", ""),
                category=rec.get("category", ""),
                timestamp=ts,
                entities=entities,
                relations=relations,
                graph_path=graph_path,
            )
            stats["synced"] += 1
        except Exception as e:
            logger.error("Graph sync failed for %s: %s", memory_id, e)
            stats["errors"] += 1

        if progress_fn:
            progress_fn(i + 1, total)

    return stats


# ──────────────────────────────────────────────
# Phase 4: 孤立 Entity 掃除
# ──────────────────────────────────────────────

def cleanup_orphan_entities(graph_path: str) -> Dict[str, int]:
    """Phase 4: mention_count <= 0 の孤立 Entity を削除する。

    Returns:
        {"deleted": N}
    """
    from engram.graph import get_graph_db, get_connection

    get_graph_db(graph_path)
    conn = get_connection(graph_path)

    # 孤立エンティティの数を数える
    result = conn.execute(
        "MATCH (e:Entity) WHERE e.mention_count <= 0 RETURN count(e)"
    )
    count = 0
    if result.has_next():
        count = result.get_next()[0]

    if count > 0:
        # RELATES_TO エッジも掃除（孤立エンティティ間のエッジ）
        conn.execute(
            "MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity) "
            "WHERE e1.mention_count <= 0 OR e2.mention_count <= 0 "
            "DELETE r"
        )
        conn.execute(
            "MATCH (e:Entity) WHERE e.mention_count <= 0 DELETE e"
        )

    return {"deleted": count}
