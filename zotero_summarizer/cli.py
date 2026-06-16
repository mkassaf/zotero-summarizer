"""Command-line entry point.

    python -m zotero_summarizer "My Collection"
    python -m zotero_summarizer ABCD1234 --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .config import build_llm, load_settings
from .graph import build_graph
from .summarizer import Summarizer
from .zotero_client import ZoteroClient


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="zotero-summarizer",
        description="Summarize every paper's PDF in a Zotero collection and "
        "write the summary back as a note.",
    )
    parser.add_argument(
        "collection",
        help="Collection name (case-insensitive) or its 8-character key.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N papers."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-summarize papers that already have an AI summary note.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate summaries and print them, but don't write to Zotero.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        zclient = ZoteroClient(settings)
        llm = build_llm(settings)
    except Exception as exc:  # configuration / dependency errors
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    summarizer = Summarizer(llm, settings.llm_model)
    graph = build_graph(zclient, summarizer)

    try:
        final = graph.invoke(
            {
                "collection": args.collection,
                "force": args.force,
                "limit": args.limit,
                "dry_run": args.dry_run,
            },
            # The graph loops one set of nodes per paper; raise the step ceiling.
            config={"recursion_limit": 10_000},
        )
    except Exception as exc:
        # e.g. collection not found, or Zotero not reachable.
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 1

    results = final.get("results", [])
    print("\n=== Summary report ===")
    for r in results:
        print(f"  [{r['status']:<28}] {r['title']}")
    written = sum(1 for r in results if r["status"] == "summarized")
    print(f"\n{len(results)} paper(s) processed, {written} note(s) written.")

    fatal = final.get("fatal_error")
    if fatal:
        print(f"\nStopped early — LLM error: {fatal}", file=sys.stderr)
        print(
            "Check LLM_API_KEY / account balance in .env, or switch LLM_PROVIDER.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
