"""ae-groom のテスト."""

import json
import os
import pytest

from engram.groom import (
    CATEGORY_MAP,
    analyze_categories,
    normalize_categories,
    analyze_entities,
    re_extract_entities,
    rebuild_graph,
    cleanup_orphan_entities,
    _parse_extraction_response,
)
from engram.prompts_groom import build_entity_extraction_prompt


# === テストヘルパー ===

def _insert_test_memories(db_path, memories):
    """テスト用メモリをDBに挿入する。"""
    from engram.db import insert_records
    from engram.embedder import embed_text

    records = []
    for mem in memories:
        tags_str = " ".join(mem.get("tags", []) or [])
        text = f"{mem['event']} {mem.get('context', '')} {tags_str}"
        records.append({
            "id": mem["id"],
            "vector": embed_text(text),
            "event": mem["event"],
            "context": mem.get("context", ""),
            "core_lessons": mem.get("core_lessons", ""),
            "category": mem.get("category", "debugging"),
            "tags": mem.get("tags", []),
            "related_files": mem.get("related_files", []),
            "session_id": mem.get("session_id", "test"),
            "timestamp": None,
            "entities_json": mem.get("entities_json", "[]"),
            "relations_json": mem.get("relations_json", "[]"),
            "occurrence_count": mem.get("occurrence_count", 1),
        })
    insert_records(records, db_path)


def _make_id(suffix: str) -> str:
    """テスト用の64文字hex IDを生成。"""
    return suffix.ljust(64, "0")


# === Phase 1: Category 正規化 ===

class TestCategoryMap:
    """CATEGORY_MAP の整合性テスト。"""

    def test_all_values_are_valid_categories(self):
        valid = set(CATEGORY_MAP.values())
        for val in CATEGORY_MAP.values():
            assert val in valid

    def test_identity_mappings_exist(self):
        """維持カテゴリは自分自身にマッピングされる。"""
        identity_cats = [
            "debugging", "architecture", "implementation", "frontend",
            "configuration", "workflow", "testing", "security",
        ]
        for cat in identity_cats:
            assert CATEGORY_MAP[cat] == cat

    def test_rename_mappings(self):
        assert CATEGORY_MAP["troubleshooting"] == "debugging"
        assert CATEGORY_MAP["backend"] == "implementation"
        assert CATEGORY_MAP["system-design"] == "architecture"
        assert CATEGORY_MAP["frontend"] == "frontend"


class TestAnalyzeCategories:
    def test_detects_rename_targets(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "category": "troubleshooting"},
            {"id": _make_id("b"), "event": "ev2", "category": "debugging"},
            {"id": _make_id("c"), "event": "ev3", "category": "backend"},
        ])
        result = analyze_categories(tmp_db_path)
        assert result["total"] == 3
        assert "troubleshooting" in result["to_rename"]
        assert result["to_rename"]["troubleshooting"]["to"] == "debugging"
        assert result["to_rename"]["troubleshooting"]["count"] == 1
        assert "backend" in result["to_rename"]

    def test_detects_unknown_categories(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "category": "unknown_cat"},
        ])
        result = analyze_categories(tmp_db_path)
        assert "unknown_cat" in result["unknown"]

    def test_no_renames_needed(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "category": "debugging"},
            {"id": _make_id("b"), "event": "ev2", "category": "architecture"},
        ])
        result = analyze_categories(tmp_db_path)
        assert len(result["to_rename"]) == 0


class TestNormalizeCategories:
    def test_renames_categories(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "category": "troubleshooting"},
            {"id": _make_id("b"), "event": "ev2", "category": "backend"},
            {"id": _make_id("c"), "event": "ev3", "category": "debugging"},
        ])
        result = normalize_categories(tmp_db_path)
        assert result["renamed"] == 2
        assert result["skipped_unknown"] == 0

        # 結果を検証
        from engram.db import get_table
        df = get_table(tmp_db_path).to_pandas()
        categories = set(df["category"])
        assert "troubleshooting" not in categories
        assert "backend" not in categories
        assert "debugging" in categories
        assert "implementation" in categories

    def test_skips_unknown_categories(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "category": "unknown_cat"},
        ])
        result = normalize_categories(tmp_db_path)
        assert result["renamed"] == 0
        assert result["skipped_unknown"] == 1

    def test_preserves_other_fields(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {
                "id": _make_id("a"),
                "event": "important event",
                "context": "some context",
                "core_lessons": "lesson",
                "category": "troubleshooting",
                "tags": ["tag1", "tag2"],
                "related_files": ["file.py"],
            },
        ])
        normalize_categories(tmp_db_path)

        from engram.db import get_table
        df = get_table(tmp_db_path).to_pandas()
        row = df.iloc[0]
        assert row["event"] == "important event"
        assert row["context"] == "some context"
        assert row["category"] == "debugging"
        assert "tag1" in row["tags"]


# === Phase 2: Entity 再抽出 ===

class TestAnalyzeEntities:
    def test_counts_empty_entities(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "entities_json": "[]"},
            {"id": _make_id("b"), "event": "ev2", "entities_json": '["Next.js"]'},
            {"id": _make_id("c"), "event": "ev3", "entities_json": ""},
        ])
        result = analyze_entities(tmp_db_path)
        assert result["total"] == 3
        assert result["empty_entities"] == 2
        assert result["to_re_extract"] == 3


class TestParseExtractionResponse:
    def test_parses_valid_response(self):
        batch = [{"id": "abc"}, {"id": "def"}]
        response = json.dumps([
            {"id": "abc", "entities": ["Next.js"], "relations": []},
            {"id": "def", "entities": ["React"], "relations": [{"source": "React", "target": "Next.js", "type": "USES"}]},
        ])
        result = _parse_extraction_response(response, batch)
        assert len(result) == 2
        assert result[0]["entities"] == ["Next.js"]
        assert result[1]["entities"] == ["React"]

    def test_parses_response_with_markdown_fences(self):
        batch = [{"id": "abc"}]
        response = '```json\n[{"id": "abc", "entities": ["LanceDB"], "relations": []}]\n```'
        result = _parse_extraction_response(response, batch)
        assert result[0]["entities"] == ["LanceDB"]

    def test_fallback_to_index_when_id_mismatch(self):
        batch = [{"id": "abc"}]
        response = '[{"id": "wrong", "entities": ["X"], "relations": []}]'
        result = _parse_extraction_response(response, batch)
        assert result[0]["entities"] == ["X"]

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError, match="No JSON"):
            _parse_extraction_response("no json here", [])


class TestReExtractEntities:
    def test_updates_entities(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "Next.jsでSSR", "entities_json": "[]"},
        ])

        def mock_llm(messages):
            return json.dumps([
                {"id": _make_id("a"), "entities": ["Next.js", "SSR"], "relations": []}
            ])

        result = re_extract_entities(tmp_db_path, llm_fn=mock_llm, batch_size=5)
        assert result["updated"] == 1
        assert result["errors"] == 0

        from engram.db import get_table
        df = get_table(tmp_db_path).to_pandas()
        entities = json.loads(df.iloc[0]["entities_json"])
        assert "Next.js" in entities
        assert "SSR" in entities

    def test_handles_llm_error(self, tmp_db_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "entities_json": "[]"},
        ])

        def mock_llm_error(messages):
            raise RuntimeError("LLM timeout")

        result = re_extract_entities(tmp_db_path, llm_fn=mock_llm_error, batch_size=5)
        assert result["errors"] == 1
        assert result["updated"] == 0


# === Phase 3: GraphDB 再構築 ===

class TestRebuildGraph:
    def test_rebuilds_from_vectordb(self, tmp_db_path, tmp_graph_path):
        _insert_test_memories(tmp_db_path, [
            {
                "id": _make_id("a"),
                "event": "ev1",
                "entities_json": '["Next.js", "React"]',
                "relations_json": '[{"source": "Next.js", "target": "React", "type": "USES"}]',
            },
            {
                "id": _make_id("b"),
                "event": "ev2",
                "entities_json": '["LanceDB"]',
                "relations_json": "[]",
            },
        ])

        result = rebuild_graph(tmp_db_path, tmp_graph_path)
        assert result["synced"] == 2
        assert result["errors"] == 0

        from engram.graph import get_graph_stats
        stats = get_graph_stats(tmp_graph_path)
        assert stats["memory_count"] == 2
        assert stats["entity_count"] == 3  # Next.js, React, LanceDB

    def test_skips_empty_entities(self, tmp_db_path, tmp_graph_path):
        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "entities_json": "[]"},
        ])

        result = rebuild_graph(tmp_db_path, tmp_graph_path)
        assert result["synced"] == 1  # counted as synced (skip)
        assert result["errors"] == 0

        from engram.graph import get_graph_stats
        stats = get_graph_stats(tmp_graph_path)
        assert stats["memory_count"] == 0  # no memory node created

    def test_backs_up_old_graph(self, tmp_db_path, tmp_graph_path):
        # 先に初期グラフを作成
        from engram.graph import get_graph_db
        get_graph_db(tmp_graph_path)

        _insert_test_memories(tmp_db_path, [
            {"id": _make_id("a"), "event": "ev1", "entities_json": '["X"]'},
        ])

        rebuild_graph(tmp_db_path, tmp_graph_path)
        assert os.path.exists(tmp_graph_path + ".bak")


# === Phase 4: 孤立 Entity 掃除 ===

class TestCleanupOrphanEntities:
    def test_deletes_zero_mention_entities(self, tmp_db_path, tmp_graph_path):
        from engram.graph import get_graph_db, get_connection
        import datetime

        get_graph_db(tmp_graph_path)
        conn = get_connection(tmp_graph_path)

        # Entity with mention_count=0
        conn.execute(
            "CREATE (e:Entity {name: 'orphan', first_seen: $ts, last_seen: $ts, mention_count: 0})",
            {"ts": datetime.datetime.now()},
        )
        # Entity with mention_count=1
        conn.execute(
            "CREATE (e:Entity {name: 'used', first_seen: $ts, last_seen: $ts, mention_count: 1})",
            {"ts": datetime.datetime.now()},
        )

        result = cleanup_orphan_entities(tmp_graph_path)
        assert result["deleted"] == 1

        # Verify 'used' still exists
        r = conn.execute("MATCH (e:Entity {name: 'used'}) RETURN count(e)")
        assert r.get_next()[0] == 1

        # Verify 'orphan' is gone
        r = conn.execute("MATCH (e:Entity {name: 'orphan'}) RETURN count(e)")
        assert r.get_next()[0] == 0

    def test_no_orphans(self, tmp_graph_path):
        from engram.graph import get_graph_db
        get_graph_db(tmp_graph_path)

        result = cleanup_orphan_entities(tmp_graph_path)
        assert result["deleted"] == 0


# === Prompts ===

class TestBuildEntityExtractionPrompt:
    def test_returns_system_and_user_messages(self):
        memories = [
            {"id": "abc", "event": "ev1", "context": "ctx", "core_lessons": "lesson",
             "category": "debugging", "tags": ["t1"], "related_files": ["f.py"]},
        ]
        messages = build_entity_extraction_prompt(memories)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "abc" in messages[1]["content"]
        assert "ev1" in messages[1]["content"]

    def test_handles_multiple_memories(self):
        memories = [
            {"id": "a", "event": "ev1", "context": "", "core_lessons": "",
             "category": "debug", "tags": [], "related_files": []},
            {"id": "b", "event": "ev2", "context": "", "core_lessons": "",
             "category": "arch", "tags": [], "related_files": []},
        ]
        messages = build_entity_extraction_prompt(memories)
        assert "メモリ 1" in messages[1]["content"]
        assert "メモリ 2" in messages[1]["content"]
