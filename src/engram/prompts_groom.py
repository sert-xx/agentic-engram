"""ae-groom 用 LLM プロンプト（カテゴリ分類 / Entity再抽出）."""

from __future__ import annotations

import json
from typing import Dict, List


def build_category_classification_prompt(
    memories: List[Dict],
    canonical_categories: List[str],
) -> List[Dict[str, str]]:
    """未知カテゴリのメモリを正規カテゴリに分類するプロンプトを構築する。

    Args:
        memories: カテゴリ分類対象のメモリ辞書リスト（各要素に id, event, context, core_lessons, category）
        canonical_categories: 正規カテゴリのリスト

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    categories_list = "\n".join(f"- `{c}`" for c in canonical_categories)

    system_content = f"""\
あなたは開発知見の記憶を適切なカテゴリに分類する専門家です。

以下の正規カテゴリ一覧から、各メモリに最も適切なカテゴリを1つ選んでください。

## 正規カテゴリ
{categories_list}

## ルール
- 必ず上記の正規カテゴリから1つを選ぶこと（新しいカテゴリを作らない）
- メモリの event, context, core_lessons の内容を総合的に判断すること
- 迷った場合は最も近いものを選ぶこと

## 出力形式
JSON配列を返してください。各要素は入力メモリに対応し、以下の形式です:

```json
[
  {{"id": "メモリID", "category": "選択したカテゴリ"}}
]
```

JSONのみを返してください。説明文は不要です。"""

    user_parts: List[str] = ["以下のメモリを適切なカテゴリに分類してください。\n"]

    for i, mem in enumerate(memories, 1):
        user_parts.append(
            f"### メモリ {i} (id: {mem.get('id', '')})\n"
            f"- **current_category**: {mem.get('category', '')}\n"
            f"- **event**: {mem.get('event', '')}\n"
            f"- **context**: {mem.get('context', '')}\n"
            f"- **core_lessons**: {mem.get('core_lessons', '')}\n"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def build_entity_extraction_prompt(
    memories: List[Dict],
) -> List[Dict[str, str]]:
    """メモリのバッチから entity/relation を再抽出するプロンプトを構築する。

    Args:
        memories: メモリ辞書のリスト（各要素に id, event, context, core_lessons, category, tags, related_files）

    Returns:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    system_content = """\
あなたは開発知見の記憶からエンティティ（固有名詞・技術名・ツール名・プロジェクト名・概念名）と
それらの間の関係性を抽出する専門家です。

## エンティティ抽出ルール
- 具体的な固有名詞を抽出する（例: "Next.js", "LanceDB", "Playwright", "GitHub Actions"）
- 一般的すぎる語（"ファイル", "エラー", "設定"）は除外する
- プロジェクト名、ライブラリ名、フレームワーク名、ツール名、サービス名を優先する
- コンポーネント名、クラス名、関数名も有意なものは含める
- 1つのメモリから3〜8個のエンティティを目安に抽出する

## リレーション抽出ルール
- source と target は抽出した entities に含まれるものを使う
- type は以下から選択: USES, DEPENDS_ON, EXTENDS, CONFIGURES, REPLACES, INTEGRATES, IMPLEMENTS, TESTS, DEPLOYS_TO, CONTAINS
- 明確な関係が読み取れるもののみ抽出する（無理に作らない）

## 出力形式
JSON配列を返してください。各要素は入力メモリに対応し、以下の形式です:

```json
[
  {
    "id": "メモリID",
    "entities": ["Entity1", "Entity2", "Entity3"],
    "relations": [
      {"source": "Entity1", "target": "Entity2", "type": "USES"}
    ]
  }
]
```

JSONのみを返してください。説明文は不要です。"""

    user_parts: List[str] = ["以下のメモリからエンティティとリレーションを抽出してください。\n"]

    for i, mem in enumerate(memories, 1):
        tags = ", ".join(mem.get("tags", []) or [])
        files = ", ".join(mem.get("related_files", []) or [])
        user_parts.append(
            f"### メモリ {i} (id: {mem.get('id', '')})\n"
            f"- **event**: {mem.get('event', '')}\n"
            f"- **context**: {mem.get('context', '')}\n"
            f"- **core_lessons**: {mem.get('core_lessons', '')}\n"
            f"- **category**: {mem.get('category', '')}\n"
            f"- **tags**: {tags}\n"
            f"- **related_files**: {files}\n"
        )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
