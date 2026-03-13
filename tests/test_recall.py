"""
ae-recall: 記憶検索CLIのテストスペック

BDD Scenarios:
  1. セマンティック検索: クエリに意味的に近い記憶が上位に返される
  2. Top-K制御: --limit オプションで返却件数を制御できる
  3. カテゴリフィルタ: --category で特定カテゴリのみに絞り込める
  4. 出力フォーマット: --format json でJSON、--format markdown でMarkdownを出力
  5. 空DB: 記憶が0件の場合、空結果を正常に返す
  6. 類似度スコア: 各結果にcosine similarityスコアが付与される
"""

import json
import os
import shutil
import tempfile

import pytest


# === Fixtures ===

@pytest.fixture
def tmp_db_path():
    path = os.path.join(tempfile.mkdtemp(), "test_engram_db")
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def populated_db(tmp_db_path):
    """テスト用に5件の記憶が入ったDBを準備する（engram.dbレイヤーを直接使用）"""
    import datetime
    from engram.db import get_table, insert_records

    records = [
        {
            "id": "id_session_001_nextjs",
            "event": "Next.jsのApp RouterでuseEffectがサーバーコンポーネントで動かない",
            "context": "サーバーコンポーネント内でブラウザAPIを使おうとした",
            "core_lessons": "'use client'ディレクティブを追加すること",
            "category": "bug_fix",
            "tags": ["Next.js", "React", "Server Components"],
            "related_files": ["app/page.tsx"],
            "session_id": "session_001",
            "timestamp": datetime.datetime.now(),
            "entities_json": "[]",
            "relations_json": "[]",
        },
        {
            "id": "id_session_002_goroutine",
            "event": "Goのgoroutineでチャネルのデッドロックが発生",
            "context": "送信側と受信側のチャネルサイズが不一致",
            "core_lessons": "Delveのattach機能でgoroutineの状態を直接確認する",
            "category": "debugging",
            "tags": ["Go", "goroutine", "deadlock", "Delve"],
            "related_files": ["cmd/worker/main.go"],
            "session_id": "session_002",
            "timestamp": datetime.datetime.now(),
            "entities_json": "[]",
            "relations_json": "[]",
        },
        {
            "id": "id_session_003_n1",
            "event": "RailsのN+1クエリでページロードが10秒超過",
            "context": "has_many関連のポリモーフィック関連付け",
            "core_lessons": "polymorphic関連付けではeager_loadを使う",
            "category": "performance",
            "tags": ["Rails", "ActiveRecord", "N+1"],
            "related_files": ["app/models/comment.rb"],
            "session_id": "session_003",
            "timestamp": datetime.datetime.now(),
            "entities_json": "[]",
            "relations_json": "[]",
        },
        {
            "id": "id_session_004_typescript",
            "event": "TypeScriptのstrictモードでOptional Chainingの型ガードが効かない",
            "context": "undefined | nullのユニオン型でのナローイング",
            "core_lessons": "明示的なif文で型ガードする方が安全",
            "category": "architecture",
            "tags": ["TypeScript", "strict", "type-guard"],
            "related_files": ["src/utils/validator.ts"],
            "session_id": "session_004",
            "timestamp": datetime.datetime.now(),
            "entities_json": "[]",
            "relations_json": "[]",
        },
        {
            "id": "id_session_005_cors",
            "event": "Ollama APIをブラウザから叩くとCORSエラー",
            "context": "Next.jsのクライアントコンポーネントからfetch",
            "core_lessons": "Route Handlerを経由させる",
            "category": "architecture",
            "tags": ["Next.js", "Ollama", "CORS"],
            "related_files": ["app/api/chat/route.ts"],
            "session_id": "session_005",
            "timestamp": datetime.datetime.now(),
            "entities_json": "[]",
            "relations_json": "[]",
        },
    ]

    insert_records(records, db_path=tmp_db_path)
    return tmp_db_path


# === BDD Scenario 1: セマンティック検索 ===

class TestSemanticSearch:
    def test_returns_relevant_results_for_query(self, populated_db):
        """クエリに意味的に近い記憶が返される"""
        from engram.recall import search_memories

        results = search_memories("Reactのサーバーコンポーネントでエラー", db_path=populated_db)

        assert len(results) > 0
        # Next.jsのuseEffect問題が最上位にくるはず
        top_result = results[0]
        assert "Next.js" in top_result["event"] or "サーバーコンポーネント" in top_result["event"]

    def test_go_deadlock_query_finds_goroutine_issue(self, populated_db):
        """Goのデッドロックに関するクエリでgoroutine問題がヒットする"""
        from engram.recall import search_memories

        results = search_memories("Goでデッドロックが起きた", db_path=populated_db)

        events = [r["event"] for r in results[:2]]
        assert any("goroutine" in e or "デッドロック" in e for e in events)

    def test_indirect_query_finds_related_memory(self, populated_db):
        """直接一致しないクエリでも意味的に関連する記憶がヒットする"""
        from engram.recall import search_memories

        # "データベースクエリが遅い" → N+1問題がヒットするはず
        results = search_memories("データベースクエリが遅い", db_path=populated_db)

        events = [r["event"] for r in results[:3]]
        assert any("N+1" in e or "ページロード" in e for e in events)


# === BDD Scenario 2: Top-K制御 ===

class TestTopKControl:
    def test_default_limit_returns_up_to_5(self, populated_db):
        """デフォルトでは最大5件返す"""
        from engram.recall import search_memories

        results = search_memories("プログラミング", db_path=populated_db)
        assert len(results) <= 5

    def test_custom_limit(self, populated_db):
        """limit指定で返却件数を制御できる"""
        from engram.recall import search_memories

        results = search_memories("プログラミング", db_path=populated_db, limit=2)
        assert len(results) <= 2


# === BDD Scenario 3: カテゴリフィルタ ===

class TestCategoryFilter:
    def test_filter_by_category(self, populated_db):
        """カテゴリを指定して絞り込める"""
        from engram.recall import search_memories

        results = search_memories(
            "エラー", db_path=populated_db, category="architecture"
        )

        for r in results:
            assert r["category"] == "architecture"

    def test_filter_with_no_matching_category_returns_empty(self, populated_db):
        """一致するカテゴリがない場合は空リストを返す"""
        from engram.recall import search_memories

        results = search_memories(
            "テスト", db_path=populated_db, category="nonexistent_category"
        )

        assert results == []


# === BDD Scenario 4: 出力フォーマット ===

class TestOutputFormat:
    def test_json_format(self, populated_db):
        """format_output('json') でJSON文字列を返す"""
        from engram.recall import search_memories, format_output

        results = search_memories("CORSエラー", db_path=populated_db)
        output = format_output(results, fmt="json")

        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_markdown_format(self, populated_db):
        """format_output('markdown') でMarkdown文字列を返す（##見出しと記憶フィールドの存在確認）"""
        from engram.recall import search_memories, format_output

        results = search_memories("CORSエラー", db_path=populated_db)
        output = format_output(results, fmt="markdown")

        # ## 見出しが含まれること
        assert "##" in output, "Markdownに##見出しが含まれること"
        # 記憶フィールドのラベルが含まれること
        assert any(
            field in output for field in ["event", "context", "core_lessons", "category"]
        ), "Markdownに記憶フィールド（event/context/core_lessons/category）のいずれかが含まれること"
        assert "CORS" in output


# === BDD Scenario 4.5: vectorフィールド非公開 ===

class TestVectorFieldExclusion:
    def test_search_results_do_not_contain_vector_field(self, populated_db):
        """search_memoriesの返却dictにvectorフィールドが含まれない"""
        from engram.recall import search_memories

        results = search_memories("CORSエラー", db_path=populated_db)

        assert len(results) > 0
        for r in results:
            assert "vector" not in r, "返却dictにvectorフィールドが含まれてはならない"


# === BDD Scenario 5: 空DB ===

class TestEmptyDatabase:
    def test_search_on_empty_db_returns_empty(self, tmp_db_path):
        """記憶が0件のDBを検索しても空リストで正常終了する"""
        from engram.recall import search_memories

        results = search_memories("何でも", db_path=tmp_db_path)
        assert results == []


# === BDD Scenario 6: 類似度スコア ===

class TestSimilarityScore:
    def test_results_include_score(self, populated_db):
        """各結果にsimilarityスコアが付与される"""
        from engram.recall import search_memories

        results = search_memories("CORSエラー", db_path=populated_db)

        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_results_sorted_by_score_descending(self, populated_db):
        """結果はスコア降順でソートされている"""
        from engram.recall import search_memories

        results = search_memories("CORSエラー", db_path=populated_db)

        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)
