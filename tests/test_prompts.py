"""
Agent 2用プロンプト生成のテストスペック

BDD Scenarios:
  1. 基本構造: messages配列がOpenAI Chat Completions形式で生成される
  2. システムプロンプト: roleがsystemのメッセージにAgent 2の指示が含まれる
  3. 差分テキスト注入: ユーザーメッセージに差分テキストが含まれる
  4. 既存記憶の注入: 関連する既存記憶がコンテキストとして渡される
  5. 出力スキーマ指示: プロンプトにJSON出力スキーマの仕様が含まれる
  6. エッジケース: 空差分、大量テキスト、既存記憶なし
"""

import json

import pytest


# === BDD Scenario 1: 基本構造 ===

class TestMessageStructure:
    def test_returns_list_of_messages(self):
        """build_extraction_promptがlist[dict]を返す"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="some log text", existing_memories=[])

        assert isinstance(messages, list)
        assert len(messages) >= 2  # 最低でもsystem + user

    def test_messages_have_role_and_content(self):
        """各メッセージにroleとcontentフィールドが含まれる"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="some log text", existing_memories=[])

        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_first_message_is_system(self):
        """最初のメッセージのroleがsystemである"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="some log text", existing_memories=[])

        assert messages[0]["role"] == "system"


# === BDD Scenario 2: システムプロンプト ===

class TestSystemPrompt:
    def test_system_prompt_describes_agent2_role(self):
        """システムプロンプトにAgent 2（砂金掘り）の役割説明が含まれる"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        system_content = messages[0]["content"]

        # 抽出・意味付けに関するキーワードが含まれること
        assert any(
            keyword in system_content
            for keyword in ["抽出", "記憶", "knowledge", "extract", "memory"]
        )

    def test_system_prompt_defines_actions(self):
        """システムプロンプトにSKIP/INSERT/UPDATEの3アクションが定義されている"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        system_content = messages[0]["content"]

        assert "SKIP" in system_content
        assert "INSERT" in system_content
        assert "UPDATE" in system_content

    def test_system_prompt_explains_skip_criteria(self):
        """システムプロンプトに「作業中・格闘中」の場合SKIPする旨が記載されている"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        system_content = messages[0]["content"]

        # SKIPの判断基準に関する記述があること
        assert any(
            keyword in system_content
            for keyword in ["作業中", "格闘中", "途中", "未解決", "in progress"]
        )


# === BDD Scenario 3: 差分テキスト注入 ===

class TestDiffTextInjection:
    def test_diff_text_appears_in_user_message(self):
        """差分テキストがuserロールのメッセージに含まれる"""
        from engram.prompts import build_extraction_prompt

        diff = "ここにセッションログの差分が入る\nCORSエラーが発生\n解決方法を発見"
        messages = build_extraction_prompt(diff_text=diff, existing_memories=[])

        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) >= 1

        user_content = " ".join(m["content"] for m in user_messages)
        assert "CORSエラーが発生" in user_content

    def test_diff_text_is_not_in_system_message(self):
        """差分テキスト自体はsystemメッセージには入らない（変数的な内容はuserに）"""
        from engram.prompts import build_extraction_prompt

        unique_marker = "UNIQUE_DIFF_MARKER_12345"
        messages = build_extraction_prompt(diff_text=unique_marker, existing_memories=[])

        system_content = messages[0]["content"]
        assert unique_marker not in system_content


# === BDD Scenario 4: 既存記憶の注入 ===

class TestExistingMemoriesInjection:
    def test_existing_memories_included_when_provided(self):
        """既存記憶がある場合、プロンプトに含まれる"""
        from engram.prompts import build_extraction_prompt

        existing = [
            {
                "id": "abc123",
                "event": "CORSエラーの解決",
                "core_lessons": "Route Handlerを使う",
                "score": 0.85,
            }
        ]
        messages = build_extraction_prompt(diff_text="log text", existing_memories=existing)

        all_content = " ".join(m["content"] for m in messages)
        assert "CORSエラーの解決" in all_content

    def test_existing_memory_ids_available_for_update(self):
        """既存記憶のIDがUPDATE用に参照可能な形で含まれる"""
        from engram.prompts import build_extraction_prompt

        existing = [
            {
                "id": "target_id_for_update",
                "event": "既存記憶",
                "core_lessons": "既存教訓",
                "score": 0.90,
            }
        ]
        messages = build_extraction_prompt(diff_text="log text", existing_memories=existing)

        all_content = " ".join(m["content"] for m in messages)
        assert "target_id_for_update" in all_content

    def test_no_existing_memories_still_valid(self):
        """既存記憶が空リストでも正常にプロンプトが生成される"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])

        assert len(messages) >= 2  # system + user


# === BDD Scenario 5: 出力スキーマ指示 ===

class TestOutputSchemaInstruction:
    def test_output_format_specifies_json_array(self):
        """出力形式としてJSON配列が指定されている"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        all_content = " ".join(m["content"] for m in messages)

        assert "JSON" in all_content

    def test_output_schema_includes_required_fields(self):
        """出力スキーマにaction, payload等の必須フィールドが言及されている"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        all_content = " ".join(m["content"] for m in messages)

        assert "action" in all_content
        assert "payload" in all_content

    def test_output_schema_includes_payload_fields(self):
        """payloadの内部フィールド（event, context, core_lessons等）が言及されている"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="log text", existing_memories=[])
        all_content = " ".join(m["content"] for m in messages)

        for field in ["event", "context", "core_lessons", "category", "tags"]:
            assert field in all_content, f"'{field}'がプロンプトに含まれていない"


# === BDD Scenario 6: エッジケース ===

class TestPromptEdgeCases:
    def test_empty_diff_text(self):
        """空の差分テキストでもエラーにならない"""
        from engram.prompts import build_extraction_prompt

        messages = build_extraction_prompt(diff_text="", existing_memories=[])
        assert isinstance(messages, list)
        assert len(messages) >= 2

    def test_very_long_diff_text(self):
        """非常に長い差分テキストでもエラーにならない"""
        from engram.prompts import build_extraction_prompt

        long_text = "ログ行\n" * 10000
        messages = build_extraction_prompt(diff_text=long_text, existing_memories=[])
        assert isinstance(messages, list)

    def test_diff_text_truncated_to_max_diff_lines(self):
        """MAX_DIFF_LINESを超える差分は末尾N行のみがプロンプトに含まれる"""
        from engram.prompts import build_extraction_prompt, MAX_DIFF_LINES

        # MAX_DIFF_LINES + 100行のテキストを生成
        # 先頭行は捨てられ、末尾がMAX_DIFF_LINESに収まることを検証
        early_marker = "THIS_IS_EARLY_LINE_MARKER"
        late_marker = "THIS_IS_LATE_LINE_MARKER"
        lines = [f"line_{i}" for i in range(MAX_DIFF_LINES + 100)]
        lines[0] = early_marker          # 先頭行（切り捨てられるべき）
        lines[-1] = late_marker          # 末尾行（保持されるべき）
        long_text = "\n".join(lines)

        messages = build_extraction_prompt(diff_text=long_text, existing_memories=[])
        all_content = " ".join(m["content"] for m in messages)

        # 末尾行は含まれる
        assert late_marker in all_content
        # 先頭行は切り捨てられている
        assert early_marker not in all_content

    def test_many_existing_memories(self):
        """大量の既存記憶が渡されてもエラーにならない"""
        from engram.prompts import build_extraction_prompt

        existing = [
            {"id": f"id_{i}", "event": f"event_{i}", "core_lessons": f"lesson_{i}", "score": 0.5}
            for i in range(50)
        ]
        messages = build_extraction_prompt(diff_text="log text", existing_memories=existing)
        assert isinstance(messages, list)
