#!/usr/bin/env python3
"""ae-miner: 砂金掘りエンジン CLI エントリポイント.

cron等で定期実行し、short-term-memory/ の生ログから記憶を抽出する。
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Mine memories from session logs")
    parser.add_argument(
        "--log-dir",
        default=os.path.expanduser("~/.engram/short-term-memory"),
        help="Directory containing session log files",
    )
    parser.add_argument(
        "--db-path",
        default=os.path.expanduser("~/.engram/memory-db/vector_store"),
        help="Path to LanceDB database directory",
    )
    parser.add_argument(
        "--cursor-path",
        default=os.path.expanduser("~/.engram/config/cursor.json"),
        help="Path to cursor.json",
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help="Archive directory (default: <log-dir>/archive)",
    )
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=7,
        help="Days before a stale log is archived",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually calling LLM",
    )
    args = parser.parse_args()

    archive_dir = args.archive_dir or os.path.join(args.log_dir, "archive")

    try:
        from engram.cursor import CursorManager
        from engram.miner import scan_logs, process_log, archive_stale_logs

        cm = CursorManager(args.cursor_path)

        # 1. スキャン
        targets = scan_logs(args.log_dir, cm)

        if not targets:
            print("No log files to process.")
        else:
            print(f"Found {len(targets)} log file(s) to process.")

            if args.dry_run:
                for t in targets:
                    print(f"  [DRY RUN] {t['filename']}")
            else:
                # NOTE: 実運用では llm_fn に実際のLLM呼び出しを渡す
                # ここではプレースホルダーとしてエラーを出す
                def _llm_placeholder(messages):
                    raise NotImplementedError(
                        "LLM function not configured. "
                        "Set up an LLM provider (e.g., OpenAI, Anthropic) "
                        "and pass it as llm_fn."
                    )

                all_skipped = True
                for t in targets:
                    print(f"  Processing: {t['filename']} ...")
                    try:
                        process_log(
                            t["filepath"],
                            cm,
                            _llm_placeholder,
                            db_path=args.db_path,
                        )
                        all_skipped = False
                    except NotImplementedError as e:
                        print(f"    Skipped (no LLM configured): {e}", file=sys.stderr)

                # 全ターゲットがLLM未設定でスキップされた場合は異常終了
                if all_skipped:
                    sys.exit(1)

        # 2. アーカイブ
        archive_stale_logs(args.log_dir, archive_dir, cm, ttl_days=args.ttl_days)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
