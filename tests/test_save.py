"""
ae-save: 記憶保存CLIのテストスペック

BDD Scenarios:
  1. INSERT: 新規記憶をJSON配列で渡すとLanceDBに保存される
  2. UPDATE: target_id指定で既存記憶を上書き更新できる
  3. UPSERT冪等性: 同一IDで再実行しても重複せず上書きされる
  4. SKIP: action=SKIPの場合、DB操作を行わず正常終了する
  5. バッチ保存: 複数記憶を含むJSON配列を一括保存できる
  6. バリデーション: 不正なJSONや必須フィールド欠損時にエラーを返す
  7. entities/relations: V1ではJSON文字列として保持される
  8. timestamp: INSERT後にtimestampが正しく設定される
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
def sample_insert_payload():
    return [
        {
            "action": "INSERT",
            "target_id": None,
            "payload": {
                "event": "Ollama APIを叩くとCORSエラー",
                "context": "Next.jsのクライアントから直接フェッチ",
                "core_lessons": "Next.jsのRoute Handlerを経由させること",
                "category": "architecture",
                "tags": ["Next.js", "Ollama", "CORS"],
                "related_files": ["app/api/chat/route.ts"],
                "session_id": "session_20260313_1000",
            },
            "entities": ["Next.js", "Ollama API", "Route Handler"],
            "relations": [
                {"source": "Next.js", "target": "Route Handler", "type": "USES"}
            ],
        }
    ]


@pytest.fixture
def sample_skip_payload():
    return [
        {
            "action": "SKIP",
            "reason": "まだCORSエラーの原因を調査中であり解決していないため。",
        }
    ]


# === BDD Scenario 1: INSERT ===

class TestInsert:
    def test_insert_creates_record_in_lancedb(self, tmp_db_path, sample_insert_payload):
        """新規記憶をINSERTすると、LanceDBに1件保存される"""
        from engram.save import save_memories
        from engram.db import get_table

        result = save_memories(sample_insert_payload, db_path=tmp_db_path)

        assert result["inserted"] == 1
        assert result["updated"] == 0
        assert result["skipped"] == 0

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        assert len(records) == 1
        assert records.iloc[0]["event"] == "Ollama APIを叩くとCORSエラー"
        assert records.iloc[0]["category"] == "architecture"

    def test_insert_generates_deterministic_id(self, tmp_db_path, sample_insert_payload):
        """IDはsession_id + eventから決定論的に生成される"""
        from engram.save import save_memories, generate_memory_id
        from engram.db import get_table

        save_memories(sample_insert_payload, db_path=tmp_db_path)

        expected_id = generate_memory_id(
            session_id="session_20260313_1000",
            event="Ollama APIを叩くとCORSエラー",
        )

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        assert records.iloc[0]["id"] == expected_id

    def test_insert_stores_embedding_vector(self, tmp_db_path, sample_insert_payload):
        """保存されたレコードにembeddingベクトルが含まれる"""
        from engram.save import save_memories
        from engram.db import get_table

        save_memories(sample_insert_payload, db_path=tmp_db_path)

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        vector = records.iloc[0]["vector"]
        assert len(vector) == 384  # paraphrase-multilingual-MiniLM-L12-v2

    def test_insert_stores_entities_relations_as_json_string(
        self, tmp_db_path, sample_insert_payload
    ):
        """V1ではentities/relationsはJSON文字列として保持される"""
        from engram.save import save_memories
        from engram.db import get_table

        save_memories(sample_insert_payload, db_path=tmp_db_path)

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        entities = json.loads(records.iloc[0]["entities_json"])
        relations = json.loads(records.iloc[0]["relations_json"])
        assert "Next.js" in entities
        assert relations[0]["type"] == "USES"

    def test_insert_sets_timestamp(self, tmp_db_path, sample_insert_payload):
        """INSERT後のレコードにtimestampが正しく設定される"""
        import pandas as pd
        from engram.save import save_memories
        from engram.db import get_table

        save_memories(sample_insert_payload, db_path=tmp_db_path)

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        ts = records.iloc[0]["timestamp"]
        # timestampがNoneでなく、かつpandasのTimestampまたはdatetime互換であること
        assert ts is not None
        assert pd.notna(ts)


# === BDD Scenario 2: UPDATE ===

class TestUpdate:
    def test_update_overwrites_existing_record(self, tmp_db_path, sample_insert_payload):
        """target_id指定のUPDATEで既存記憶が上書きされる"""
        from engram.save import save_memories, generate_memory_id
        from engram.db import get_table

        # まずINSERT
        save_memories(sample_insert_payload, db_path=tmp_db_path)

        existing_id = generate_memory_id(
            session_id="session_20260313_1000",
            event="Ollama APIを叩くとCORSエラー",
        )

        # UPDATEで上書き
        update_payload = [
            {
                "action": "UPDATE",
                "target_id": existing_id,
                "payload": {
                    "event": "Ollama APIを叩くとCORSエラー",
                    "context": "Next.jsのクライアントから直接フェッチ",
                    "core_lessons": "Route Handler経由 + OLLAMA_ORIGINS環境変数の設定も必要",
                    "category": "architecture",
                    "tags": ["Next.js", "Ollama", "CORS", "env"],
                    "related_files": ["app/api/chat/route.ts", ".env.local"],
                    "session_id": "session_20260313_1000",
                },
                "entities": [],
                "relations": [],
            }
        ]

        result = save_memories(update_payload, db_path=tmp_db_path)
        assert result["updated"] == 1

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        assert len(records) == 1  # 件数は増えない
        assert "OLLAMA_ORIGINS" in records.iloc[0]["core_lessons"]
        # UPDATE後もレコードのidがtarget_idと一致していること
        assert records.iloc[0]["id"] == existing_id

    def test_update_with_nonexistent_target_id_raises_error(self, tmp_db_path):
        """存在しないtarget_idでUPDATEするとSaveValidationErrorになる"""
        from engram.save import save_memories, SaveValidationError

        update_payload = [
            {
                "action": "UPDATE",
                "target_id": "nonexistent_id_that_does_not_exist",
                "payload": {
                    "event": "存在しない記憶の上書き",
                    "context": "テスト",
                    "core_lessons": "エラーになるはず",
                    "category": "debugging",
                    "tags": [],
                    "related_files": [],
                    "session_id": "session_test",
                },
                "entities": [],
                "relations": [],
            }
        ]

        with pytest.raises(SaveValidationError) as exc_info:
            save_memories(update_payload, db_path=tmp_db_path)
        assert exc_info.value.error_code == "TARGET_NOT_FOUND"


# === BDD Scenario 3: UPSERT冪等性 ===

class TestUpsertIdempotency:
    def test_same_insert_twice_does_not_duplicate(
        self, tmp_db_path, sample_insert_payload
    ):
        """同一内容のINSERTを2回実行しても1件のみ存在する"""
        from engram.save import save_memories
        from engram.db import get_table

        save_memories(sample_insert_payload, db_path=tmp_db_path)
        save_memories(sample_insert_payload, db_path=tmp_db_path)

        table = get_table(db_path=tmp_db_path)
        records = table.to_pandas()
        assert len(records) == 1


# === BDD Scenario 4: SKIP ===

class TestSkip:
    def test_skip_does_not_modify_db(self, tmp_db_path, sample_skip_payload):
        """action=SKIPの場合、DBに変更を加えず正常終了する"""
        from engram.save import save_memories
        from engram.db import get_table

        result = save_memories(sample_skip_payload, db_path=tmp_db_path)

        assert result["skipped"] == 1
        assert result["inserted"] == 0
        assert result["updated"] == 0

    def test_skip_mixed_with_insert(self, tmp_db_path):
        """SKIP + INSERTが混在する配列でも正しく処理される"""
        from engram.save import save_memories
        from engram.db import get_table

        mixed_payload = [
            {"action": "SKIP", "reason": "作業中"},
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": "テスト記憶",
                    "context": "テスト",
                    "core_lessons": "テスト教訓",
                    "category": "debugging",
                    "tags": ["test"],
                    "related_files": [],
                    "session_id": "session_test",
                },
                "entities": [],
                "relations": [],
            },
        ]

        result = save_memories(mixed_payload, db_path=tmp_db_path)
        assert result["skipped"] == 1
        assert result["inserted"] == 1

        table = get_table(db_path=tmp_db_path)
        assert len(table.to_pandas()) == 1


# === BDD Scenario 5: バッチ保存 ===

class TestBatchSave:
    def test_multiple_inserts_in_single_call(self, tmp_db_path):
        """複数INSERTを含むJSON配列を一括保存できる"""
        from engram.save import save_memories
        from engram.db import get_table

        batch_payload = [
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": f"テストイベント{i}",
                    "context": f"コンテキスト{i}",
                    "core_lessons": f"教訓{i}",
                    "category": "debugging",
                    "tags": [f"tag{i}"],
                    "related_files": [],
                    "session_id": f"session_{i}",
                },
                "entities": [],
                "relations": [],
            }
            for i in range(5)
        ]

        result = save_memories(batch_payload, db_path=tmp_db_path)
        assert result["inserted"] == 5

        table = get_table(db_path=tmp_db_path)
        assert len(table.to_pandas()) == 5


# === BDD Scenario 6: バリデーション ===

class TestValidation:
    def test_invalid_json_raises_error(self, tmp_db_path):
        """不正な構造のペイロードはバリデーションエラーになり、error_codeが設定される"""
        from engram.save import save_memories, SaveValidationError

        with pytest.raises(SaveValidationError) as exc_info:
            save_memories([{"invalid": "data"}], db_path=tmp_db_path)
        assert exc_info.value.error_code == "INVALID_SCHEMA"

    def test_missing_required_field_raises_error(self, tmp_db_path):
        """payloadの必須フィールドが欠けている場合バリデーションエラーになり、error_codeが設定される"""
        from engram.save import save_memories, SaveValidationError

        incomplete_payload = [
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": "テスト",
                    # context, core_lessons 等が欠落
                },
                "entities": [],
                "relations": [],
            }
        ]

        with pytest.raises(SaveValidationError) as exc_info:
            save_memories(incomplete_payload, db_path=tmp_db_path)
        assert exc_info.value.error_code == "MISSING_FIELD"

    def test_invalid_action_raises_error(self, tmp_db_path):
        """未知のactionはバリデーションエラーになり、error_codeが設定される"""
        from engram.save import save_memories, SaveValidationError

        with pytest.raises(SaveValidationError) as exc_info:
            save_memories(
                [{"action": "DELETE", "payload": {}}], db_path=tmp_db_path
            )
        assert exc_info.value.error_code == "INVALID_ACTION"
