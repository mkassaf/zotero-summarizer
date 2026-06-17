"""Command-line entry point.

    zotery "My Collection"
    zotery ABCD1234 --limit 5 --dry-run

(equivalently, from a source checkout: python -m zotero_summarizer ...)
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
        prog="zotery",
        description="Summarize every paper's PDF in a Zotero collection and "
        "write the summary back as a note. With --rq, answer a research question "
        "from each paper (relevance, reasoning, findings, verbatim snippets) "
        "instead of writing a generic summary.",
    )
    parser.add_argument(
        "collection",
        help="Collection name (case-insensitive) or its 8-character key.",
    )
    parser.add_argument(
        "--rq",
        metavar="QUESTION",
        default=None,
        help="Research question. Instead of a generic summary, extract the "
        "passages that answer this question from each paper, with the model's "
        "relevance judgement, reasoning, findings, and verbatim snippets.",
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
    parser.add_argument(
        "--llm-api-key",
        metavar="KEY",
        default=None,
        help="API key for the LLM provider. Takes precedence over LLM_API_KEY "
        "and the provider's standard env var (DEEPSEEK_API_KEY, OPENAI_API_KEY, "
        "GOOGLE_API_KEY). Not needed for Ollama.",
    )
    parser.add_argument(
        "--zotero-api-key",
        metavar="KEY",
        default=None,
        help="Zotero Web API key. Takes precedence over ZOTERO_API_KEY. "
        "Not needed in local mode (ZOTERO_LOCAL=true).",
    )
    args = parser.parse_args(argv)

    settings = load_settings(
        llm_api_key=args.llm_api_key,
        zotero_api_key=args.zotero_api_key,
    )
    try:
        zclient = ZoteroClient(settings)
        llm = build_llm(settings)
    except Exception as exc:  # configuration / dependency errors
        print(f"Startup error: {exc}", file=sys.stderr)
        return 2

    summarizer = Summarizer(llm, settings.llm_model)
    graph = build_graph(zclient, summarizer)

    if args.rq:
        print(f'Research question: "{args.rq}"')

    try:
        final = graph.invoke(
            {
                "collection": args.collection,
                "force": args.force,
                "limit": args.limit,
                "dry_run": args.dry_run,
                "rq": args.rq,
            },
            # The graph loops one set of nodes per paper; raise the step ceiling.
            config={"recursion_limit": 10_000},
        )
    except Exception as exc:
        # e.g. collection not found, or Zotero not reachable.
        print(f"\nRun failed: {exc}", file=sys.stderr)
        return 1

    results = final.get("results", [])
    label = "RQ analysis" if args.rq else "Summary"
    print(f"\n=== {label} report ===")
    for r in results:
        print(f"  [{r['status']:<28}] {r['title']}")
    written = sum(1 for r in results if r["status"] in ("summarized", "answered"))
    print(f"\n{len(results)} paper(s) processed, {written} note(s) written.")

    fatal = final.get("fatal_error")
    if fatal:
        print(f"\nStopped early — LLM error: {fatal}", file=sys.stderr)
        print(
            "Check the API key (--llm-api-key, LLM_API_KEY, or the provider's "
            "standard env var) and account balance, or switch LLM_PROVIDER.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
