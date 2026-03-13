"""
cursor.json 管理モジュールのテストスペック

BDD Scenarios:
  1. 初期状態: cursor.jsonが存在しない場合でも正常動作する
  2. GET: 存在するファイルのカーソル情報を取得できる
  3. UPDATE: カーソル情報を新規作成・更新できる
  4. REMOVE: 特定ファイルのカーソルを削除できる
  5. LIST: 全エントリを取得できる
  6. 永続化: ディスク上のcursor.jsonに正しくJSON保存される
  7. 堅牢性: 不正なJSONファイルや同時アクセスに対して安全にフォールバックする
"""

import json
import os

import pytest


# === BDD Scenario 1: 初期状態 ===

class TestCursorInitialization:
    def test_new_cursor_manager_without_existing_file(self, tmp_cursor_path):
        """cursor.jsonが存在しない状態でCursorManagerを生成してもエラーにならない"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        assert cm.list_cursors() == {}

    def test_cursor_file_not_created_until_first_write(self, tmp_cursor_path):
        """初期化だけではファイルは生成されない（遅延書き込み）"""
        from engram.cursor import CursorManager

        CursorManager(tmp_cursor_path)
        assert not os.path.exists(tmp_cursor_path)

    def test_loads_existing_cursor_file(self, tmp_cursor_path):
        """既存のcursor.jsonがあれば読み込む"""
        from engram.cursor import CursorManager

        existing_data = {
            "session_log.txt": {
                "last_read_line": 100,
                "last_checked_mtime": 1741824000.0,
            }
        }
        with open(tmp_cursor_path, "w") as f:
            json.dump(existing_data, f)

        cm = CursorManager(tmp_cursor_path)
        assert cm.list_cursors() == existing_data


# === BDD Scenario 2: GET ===

class TestGetCursor:
    def test_get_existing_cursor(self, tmp_cursor_path):
        """存在するファイルのカーソルを取得できる"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=42, last_checked_mtime=1741824000.0)

        cursor = cm.get_cursor("log.txt")
        assert cursor["last_read_line"] == 42
        assert cursor["last_checked_mtime"] == 1741824000.0

    def test_get_nonexistent_cursor_returns_default(self, tmp_cursor_path):
        """存在しないファイル名でget_cursorするとデフォルト値を返す"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cursor = cm.get_cursor("nonexistent.txt")

        assert cursor["last_read_line"] == 0
        assert cursor["last_checked_mtime"] == 0.0

    def test_get_cursor_returns_dict_with_expected_keys(self, tmp_cursor_path):
        """返却dictにlast_read_lineとlast_checked_mtimeが含まれる"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=1, last_checked_mtime=1.0)

        cursor = cm.get_cursor("log.txt")
        assert "last_read_line" in cursor
        assert "last_checked_mtime" in cursor


# === BDD Scenario 3: UPDATE ===

class TestUpdateCursor:
    def test_update_creates_new_entry(self, tmp_cursor_path):
        """存在しないファイルに対するupdateで新規エントリが作成される"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("new_log.txt", last_read_line=10, last_checked_mtime=1000.0)

        cursor = cm.get_cursor("new_log.txt")
        assert cursor["last_read_line"] == 10
        assert cursor["last_checked_mtime"] == 1000.0

    def test_update_overwrites_existing_entry(self, tmp_cursor_path):
        """既存エントリをupdateすると値が上書きされる"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=10, last_checked_mtime=1000.0)
        cm.update_cursor("log.txt", last_read_line=50, last_checked_mtime=2000.0)

        cursor = cm.get_cursor("log.txt")
        assert cursor["last_read_line"] == 50
        assert cursor["last_checked_mtime"] == 2000.0

    def test_update_does_not_affect_other_entries(self, tmp_cursor_path):
        """あるファイルのupdateが他のファイルのエントリに影響しない"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log_a.txt", last_read_line=10, last_checked_mtime=1000.0)
        cm.update_cursor("log_b.txt", last_read_line=20, last_checked_mtime=2000.0)

        cm.update_cursor("log_a.txt", last_read_line=99, last_checked_mtime=9999.0)

        cursor_b = cm.get_cursor("log_b.txt")
        assert cursor_b["last_read_line"] == 20
        assert cursor_b["last_checked_mtime"] == 2000.0


# === BDD Scenario 4: REMOVE ===

class TestRemoveCursor:
    def test_remove_existing_entry(self, tmp_cursor_path):
        """存在するエントリを削除できる"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=10, last_checked_mtime=1000.0)
        cm.remove_cursor("log.txt")

        assert cm.get_cursor("log.txt")["last_read_line"] == 0

    def test_remove_nonexistent_entry_does_not_raise(self, tmp_cursor_path):
        """存在しないエントリの削除はエラーにならない"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.remove_cursor("nonexistent.txt")  # 例外が出なければOK

    def test_remove_persists_to_disk(self, tmp_cursor_path):
        """削除後にディスク上のcursor.jsonからもエントリが消えている"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=10, last_checked_mtime=1000.0)
        cm.remove_cursor("log.txt")

        # 別インスタンスで読み直し
        cm2 = CursorManager(tmp_cursor_path)
        assert "log.txt" not in cm2.list_cursors()


# === BDD Scenario 5: LIST ===

class TestListCursors:
    def test_list_empty(self, tmp_cursor_path):
        """エントリが0件の場合、空dictを返す"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        assert cm.list_cursors() == {}

    def test_list_multiple_entries(self, tmp_cursor_path):
        """複数エントリがある場合、全件を返す"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("a.txt", last_read_line=1, last_checked_mtime=100.0)
        cm.update_cursor("b.txt", last_read_line=2, last_checked_mtime=200.0)
        cm.update_cursor("c.txt", last_read_line=3, last_checked_mtime=300.0)

        cursors = cm.list_cursors()
        assert len(cursors) == 3
        assert set(cursors.keys()) == {"a.txt", "b.txt", "c.txt"}

    def test_list_returns_copy_not_reference(self, tmp_cursor_path):
        """list_cursorsの返却値を変更しても内部状態に影響しない"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=1, last_checked_mtime=100.0)

        cursors = cm.list_cursors()
        cursors["log.txt"]["last_read_line"] = 9999

        assert cm.get_cursor("log.txt")["last_read_line"] == 1


# === BDD Scenario 6: 永続化 ===

class TestPersistence:
    def test_update_persists_to_disk(self, tmp_cursor_path):
        """updateの結果がディスクに書き出される"""
        from engram.cursor import CursorManager

        cm = CursorManager(tmp_cursor_path)
        cm.update_cursor("log.txt", last_read_line=42, last_checked_mtime=1741824000.0)

        assert os.path.exists(tmp_cursor_path)

        with open(tmp_cursor_path) as f:
            data = json.load(f)
        assert data["log.txt"]["last_read_line"] == 42

    def test_data_survives_reinitialization(self, tmp_cursor_path):
        """CursorManagerを再生成しても以前のデータが読み込める"""
        from engram.cursor import CursorManager

        cm1 = CursorManager(tmp_cursor_path)
        cm1.update_cursor("log.txt", last_read_line=42, last_checked_mtime=1741824000.0)

        cm2 = CursorManager(tmp_cursor_path)
        cursor = cm2.get_cursor("log.txt")
        assert cursor["last_read_line"] == 42


# === BDD Scenario 7: 堅牢性 ===

class TestRobustness:
    def test_corrupted_cursor_file_falls_back_to_empty(self, tmp_cursor_path):
        """cursor.jsonが壊れている場合、空状態で初期化される（データロスよりクラッシュ回避を優先）"""
        from engram.cursor import CursorManager

        with open(tmp_cursor_path, "w") as f:
            f.write("{invalid json content!!!")

        cm = CursorManager(tmp_cursor_path)
        assert cm.list_cursors() == {}

    def test_cursor_file_in_nonexistent_directory(self, tmp_path):
        """親ディレクトリが存在しない場合、自動作成される"""
        from engram.cursor import CursorManager

        nested_path = str(tmp_path / "deep" / "nested" / "cursor.json")
        cm = CursorManager(nested_path)
        cm.update_cursor("log.txt", last_read_line=1, last_checked_mtime=1.0)

        assert os.path.exists(nested_path)
