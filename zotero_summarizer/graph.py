"""LangGraph pipeline.

    START
      -> load_items        (scan the collection)
      -> process_paper     (find + download + extract the PDF)   <--+
      -> summarize         (LLM -> structured PaperSummary)          |
      -> write_note        (render HTML + push note to Zotero)  -----+ (loop)
      -> END               (when every paper is processed)
"""

from __future__ import annotations

import html
from typing import List, Optional, TypedDict, Union

from langgraph.graph import END, START, StateGraph

from .pdf_utils import extract_text
from .summarizer import (
    RQ_MARKER,
    SUMMARY_MARKER,
    PaperSummary,
    RQAnswer,
    Summarizer,
    preview,
    rq_answer_to_html,
    rq_preview,
    summary_to_html,
)
from .zotero_client import ZoteroClient


class PaperState(TypedDict, total=False):
    # inputs
    collection: str
    force: bool
    limit: Optional[int]
    dry_run: bool
    rq: Optional[str]  # research question; when set, run RQ analysis not summary
    note_title: Optional[str]  # custom <h1> title for the written note
    # queue
    items: List[dict]
    index: int
    results: List[dict]
    fatal_error: Optional[str]
    # scratch for the current paper
    current_title: str
    current_authors: str
    current_text: Optional[str]
    current_output: Optional[Union[PaperSummary, RQAnswer]]
    skip_reason: Optional[str]


def _llm_error_message(exc: Exception) -> str:
    msg = str(exc).strip()
    return msg if len(msg) <= 200 else msg[:200] + "…"


def _is_fatal_llm_error(exc: Exception) -> bool:
    """Errors that will recur on every paper (bad/empty key, no balance, no
    access) — stop the whole run instead of retrying 1 paper at a time."""
    status = getattr(exc, "status_code", None)
    if status in {401, 402, 403}:
        return True
    text = str(exc).lower()
    return any(
        s in text
        for s in ("insufficient balance", "invalid_api_key", "incorrect api key")
    )


def _authors(item: dict) -> str:
    names = []
    for creator in item["data"].get("creators", []):
        name = creator.get("name") or " ".join(
            p for p in (creator.get("firstName"), creator.get("lastName")) if p
        )
        if name:
            names.append(name)
    if not names:
        return "Unknown"
    if len(names) > 3:
        return ", ".join(names[:3]) + " et al."
    return ", ".join(names)


def build_graph(zclient: ZoteroClient, summarizer: Summarizer):
    max_chars = zclient.settings.max_pdf_chars

    def load_items(state: PaperState) -> dict:
        items = zclient.papers_in_collection(state["collection"])
        limit = state.get("limit")
        if limit:
            items = items[:limit]
        print(f"Found {len(items)} paper(s) in collection '{state['collection']}'.")
        return {"items": items, "index": 0, "results": []}

    def process_paper(state: PaperState) -> dict:
        item = state["items"][state["index"]]
        data = item["data"]
        title = data.get("title", "(untitled)")
        update = {
            "current_title": title,
            "current_authors": _authors(item),
            "current_text": None,
            "current_output": None,
            "skip_reason": None,
        }
        print(f"\n[{state['index'] + 1}/{len(state['items'])}] {title}")

        rq = state.get("rq")
        if not state.get("force"):
            if rq:
                # Same RQ already answered for this paper? (RQ text is embedded
                # in the note, so a different RQ still gets its own note.)
                already = zclient.note_matches(item["key"], RQ_MARKER, html.escape(rq))
                noun = "an analysis for this research question"
            else:
                already = zclient.note_matches(item["key"], SUMMARY_MARKER)
                noun = "an AI summary note"
            if already:
                update["skip_reason"] = "already done"
                print(f"  - skip: already has {noun} (use --force to redo)")
                return update

        attachment = zclient.find_pdf_attachment(item["key"])
        if not attachment:
            update["skip_reason"] = "no PDF attachment"
            print("  - skip: no PDF attachment found")
            return update

        pdf_bytes = zclient.download_pdf(attachment)
        if not pdf_bytes:
            update["skip_reason"] = "could not read PDF"
            print("  - skip: could not download/read the PDF file")
            return update

        text = extract_text(pdf_bytes, max_chars)
        if not text:
            update["skip_reason"] = "no extractable text"
            print("  - skip: no extractable text (scanned/image-only PDF?)")
            return update

        update["current_text"] = text
        print(f"  - extracted {len(text)} chars of text")
        return update

    def summarize(state: PaperState) -> dict:
        if not state.get("current_text"):
            return {}
        rq = state.get("rq")
        try:
            if rq:
                print("  - extracting evidence for the research question ...")
                output = summarizer.answer_rq(
                    state["current_title"],
                    state["current_authors"],
                    state["current_text"],
                    rq,
                )
            else:
                print("  - summarizing with the LLM ...")
                output = summarizer.summarize(
                    state["current_title"],
                    state["current_authors"],
                    state["current_text"],
                )
            return {"current_output": output}
        except Exception as exc:
            msg = _llm_error_message(exc)
            print(f"  - LLM error: {msg}")
            update = {"current_output": None, "skip_reason": f"LLM error: {msg}"}
            if _is_fatal_llm_error(exc):
                update["fatal_error"] = msg
                print("  - this looks fatal (auth/balance/access); stopping the run.")
            return update

    def write_note(state: PaperState) -> dict:
        item = state["items"][state["index"]]
        result = {"title": state["current_title"], "key": item["key"]}
        output = state.get("current_output")
        rq = state.get("rq")
        note_title = state.get("note_title")

        if output is None:
            result["status"] = state.get("skip_reason") or "skipped"
        elif state.get("dry_run"):
            result["status"] = "dry-run (not written)"
            print("  - dry-run: generated but NOT written to Zotero")
            print(rq_preview(rq, output) if rq else preview(output))
        else:
            if rq:
                html_note = rq_answer_to_html(
                    state["current_title"], rq, output,
                    summarizer.model_label, note_title,
                )
            else:
                html_note = summary_to_html(
                    state["current_title"], output,
                    summarizer.model_label, note_title,
                )
            zclient.add_note(item["key"], html_note)
            result["status"] = "answered" if rq else "summarized"
            print("  - note added to Zotero ✓")

        return {"results": state["results"] + [result], "index": state["index"] + 1}

    def has_items(state: PaperState) -> str:
        return "process_paper" if state["items"] else END

    def has_more(state: PaperState) -> str:
        if state.get("fatal_error"):
            return END
        return "process_paper" if state["index"] < len(state["items"]) else END

    graph = StateGraph(PaperState)
    graph.add_node("load_items", load_items)
    graph.add_node("process_paper", process_paper)
    graph.add_node("summarize", summarize)
    graph.add_node("write_note", write_note)

    graph.add_edge(START, "load_items")
    graph.add_conditional_edges("load_items", has_items, ["process_paper", END])
    graph.add_edge("process_paper", "summarize")
    graph.add_edge("summarize", "write_note")
    graph.add_conditional_edges("write_note", has_more, ["process_paper", END])

    return graph.compile()
