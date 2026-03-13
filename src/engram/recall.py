"""search_memories, format_output."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Optional

from engram.embedder import embed_text
from engram.db import get_table

_CATEGORY_PATTERN = re.compile(r"^[0-9A-Za-z_]+$")


def _validate_category(category: str) -> str:
    """Validate that category contains only alphanumerics and underscores."""
    if not isinstance(category, str) or not _CATEGORY_PATTERN.match(category):
        raise ValueError(
            f"Invalid category: {category!r}. "
            "Only alphanumerics and underscores are allowed."
        )
    return category


def search_memories(
    query: str,
    db_path: str,
    limit: int = 5,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search memories by semantic similarity.

    Returns list of dicts without vector field, with score field added.
    """
    if category is not None:
        _validate_category(category)

    try:
        table = get_table(db_path)
    except Exception:
        return []

    query_vector = embed_text(query)

    search_builder = table.search(query_vector).metric("cosine")
    if category is not None:
        search_builder = search_builder.where(f"category = '{category}'")
    search_builder = search_builder.limit(limit)
    raw_results = search_builder.to_pandas()

    if raw_results.empty:
        return []

    results = []
    for _, row in raw_results.iterrows():
        record = {}
        for col in row.index:
            if col == "vector":
                continue
            if col == "_distance":
                # cosine distance -> similarity score (1 - distance)
                # LanceDB cosine distance is in [0, 2], similarity = 1 - distance
                record["score"] = round(max(0.0, min(1.0, 1.0 - row[col])), 6)
                continue
            val = row[col]
            # Convert numpy/pandas types to native Python
            if hasattr(val, "tolist"):
                val = val.tolist()
            elif hasattr(val, "item"):
                val = val.item()
            record[col] = val
        results.append(record)

    # Sort by score descending
    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return results[:limit]


def format_output(results: List[Dict[str, Any]], fmt: str = "markdown") -> str:
    """Format search results as JSON or Markdown."""
    if fmt == "json":
        # Convert any non-serializable types
        serializable = []
        for r in results:
            item = {}
            for k, v in r.items():
                if hasattr(v, "isoformat"):
                    item[k] = v.isoformat()
                else:
                    item[k] = v
            serializable.append(item)
        return json.dumps(serializable, ensure_ascii=False, indent=2)

    # Markdown format
    lines = []
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        lines.append(f"## Memory {i} (score: {score:.4f})")
        lines.append("")
        lines.append(f"- **event**: {r.get('event', '')}")
        lines.append(f"- **context**: {r.get('context', '')}")
        lines.append(f"- **core_lessons**: {r.get('core_lessons', '')}")
        lines.append(f"- **category**: {r.get('category', '')}")
        tags = r.get("tags", [])
        if isinstance(tags, list):
            lines.append(f"- **tags**: {', '.join(str(t) for t in tags)}")
        else:
            lines.append(f"- **tags**: {tags}")
        files = r.get("related_files", [])
        if isinstance(files, list):
            lines.append(f"- **related_files**: {', '.join(str(f) for f in files)}")
        else:
            lines.append(f"- **related_files**: {files}")
        lines.append("")

    return "\n".join(lines)
