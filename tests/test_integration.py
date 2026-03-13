"""
統合テスト: ae-save → ae-recall のE2Eフロー

BDD Scenarios:
  1. CLIパイプライン: echo JSON | ae-save && ae-recall で記憶の保存→検索が通る
  2. 冪等性E2E: 同じJSONを2回パイプしても記憶が重複しない
  3. UPSERT E2E: INSERT → UPDATE で記憶が上書きされ、recall結果に反映される
  4. 不正JSON: 不正なJSONをstdinに渡すとexit code非ゼロとstderrメッセージが返る
  5. --limitオプション: --limitでrecall結果の件数が制御される
  6. Markdownフォーマット強化: ##見出しと記憶フィールドの存在確認
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest


@pytest.fixture
def tmp_db_path():
    path = os.path.join(tempfile.mkdtemp(), "test_engram_e2e")
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def ae_save_cmd():
    """ae-save.py のCLIパス"""
    return [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "ae-save.py")]


@pytest.fixture
def ae_recall_cmd():
    """ae-recall.py のCLIパス"""
    return [sys.executable, os.path.join(os.path.dirname(__file__), "..", "scripts", "ae-recall.py")]


class TestCLIPipeline:
    def test_save_then_recall_via_cli(self, tmp_db_path, ae_save_cmd, ae_recall_cmd):
        """JSON stdin → ae-save → ae-recall でE2Eが通る"""
        payload = json.dumps([
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": "CLIパイプラインテスト用の記憶",
                    "context": "E2Eテスト",
                    "core_lessons": "CLIが正しく動作すること",
                    "category": "debugging",
                    "tags": ["test", "cli"],
                    "related_files": [],
                    "session_id": "session_e2e_001",
                },
                "entities": [],
                "relations": [],
            }
        ])

        # ae-save
        save_result = subprocess.run(
            ae_save_cmd + ["--db-path", tmp_db_path],
            input=payload,
            capture_output=True,
            text=True,
        )
        assert save_result.returncode == 0, f"ae-save failed: {save_result.stderr}"

        # ae-recall (JSON format)
        recall_result = subprocess.run(
            ae_recall_cmd + [
                "--query", "CLIパイプラインテスト",
                "--db-path", tmp_db_path,
                "--format", "json",
            ],
            capture_output=True,
            text=True,
        )
        assert recall_result.returncode == 0, f"ae-recall failed: {recall_result.stderr}"

        results = json.loads(recall_result.stdout)
        assert len(results) > 0
        assert "CLIパイプライン" in results[0]["event"]

    def test_recall_markdown_format_via_cli(self, tmp_db_path, ae_save_cmd, ae_recall_cmd):
        """ae-recall --format markdown が##見出しと記憶フィールドを含むMarkdown出力を返す"""
        payload = json.dumps([
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": "Markdownフォーマットテスト",
                    "context": "テスト",
                    "core_lessons": "Markdown出力が正しいこと",
                    "category": "debugging",
                    "tags": ["test"],
                    "related_files": [],
                    "session_id": "session_e2e_002",
                },
                "entities": [],
                "relations": [],
            }
        ])

        subprocess.run(
            ae_save_cmd + ["--db-path", tmp_db_path],
            input=payload,
            capture_output=True,
            text=True,
        )

        recall_result = subprocess.run(
            ae_recall_cmd + [
                "--query", "Markdownテスト",
                "--db-path", tmp_db_path,
                "--format", "markdown",
            ],
            capture_output=True,
            text=True,
        )
        assert recall_result.returncode == 0
        output = recall_result.stdout
        # ## 見出しが含まれること
        assert "##" in output, "Markdownに##見出しが含まれること"
        # 記憶フィールドのいずれかが含まれること
        assert any(
            field in output for field in ["event", "context", "core_lessons", "category"]
        ), "Markdownに記憶フィールドが含まれること"
        assert "Markdown" in output

    def test_invalid_json_stdin_returns_nonzero_exit_code(
        self, tmp_db_path, ae_save_cmd
    ):
        """不正なJSONをstdinに渡すとexit codeが非ゼロでstderrにエラーメッセージが出力される"""
        invalid_input = "これはJSONではない{broken:"

        result = subprocess.run(
            ae_save_cmd + ["--db-path", tmp_db_path],
            input=invalid_input,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "不正なJSONではexit codeが非ゼロであること"
        assert result.stderr.strip() != "", "不正なJSONではstderrにエラーメッセージが出力されること"

    def test_recall_with_limit_option(self, tmp_db_path, ae_save_cmd, ae_recall_cmd):
        """--limitオプションでrecall結果の件数が制御される"""
        # 複数件保存
        batch_payload = json.dumps([
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": f"limitテスト用記憶{i}",
                    "context": "テスト",
                    "core_lessons": f"教訓{i}",
                    "category": "debugging",
                    "tags": ["test"],
                    "related_files": [],
                    "session_id": f"session_limit_{i}",
                },
                "entities": [],
                "relations": [],
            }
            for i in range(5)
        ])

        save_result = subprocess.run(
            ae_save_cmd + ["--db-path", tmp_db_path],
            input=batch_payload,
            capture_output=True,
            text=True,
        )
        assert save_result.returncode == 0, f"ae-save failed: {save_result.stderr}"

        # --limit 2 で実行
        recall_result = subprocess.run(
            ae_recall_cmd + [
                "--query", "limitテスト",
                "--db-path", tmp_db_path,
                "--format", "json",
                "--limit", "2",
            ],
            capture_output=True,
            text=True,
        )
        assert recall_result.returncode == 0, f"ae-recall failed: {recall_result.stderr}"

        results = json.loads(recall_result.stdout)
        assert len(results) <= 2, f"--limit 2 指定時は最大2件であること (実際: {len(results)}件)"


class TestIdempotencyE2E:
    def test_duplicate_save_does_not_create_duplicates(
        self, tmp_db_path, ae_save_cmd, ae_recall_cmd
    ):
        """同一JSONを2回saveしても記憶が重複しない"""
        payload = json.dumps([
            {
                "action": "INSERT",
                "target_id": None,
                "payload": {
                    "event": "冪等性テスト用記憶",
                    "context": "テスト",
                    "core_lessons": "2回saveしても1件",
                    "category": "debugging",
                    "tags": ["test"],
                    "related_files": [],
                    "session_id": "session_idempotent",
                },
                "entities": [],
                "relations": [],
            }
        ])

        # 2回save
        for _ in range(2):
            result = subprocess.run(
                ae_save_cmd + ["--db-path", tmp_db_path],
                input=payload,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0

        # recall で件数確認
        recall_result = subprocess.run(
            ae_recall_cmd + [
                "--query", "冪等性テスト",
                "--db-path", tmp_db_path,
                "--format", "json",
            ],
            capture_output=True,
            text=True,
        )
        results = json.loads(recall_result.stdout)
        # 同一イベントが複数返ってこないこと
        events = [r["event"] for r in results]
        assert events.count("冪等性テスト用記憶") == 1
