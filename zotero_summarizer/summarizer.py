"""LLM-backed summarizer that returns a structured 4-part summary and renders
it to Zotero-friendly note HTML."""

from __future__ import annotations

import html
from typing import List, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Stable text markers embedded in the note HTML so re-runs can detect what kind
# of note already exists. These live in the (always-present) <h1> heading, which
# Zotero preserves verbatim.
SUMMARY_HEADER = "🤖 AI Summary"
RQ_HEADER = "🤖 Research-Question Analysis"


class PaperSummary(BaseModel):
    """The structured summary the model is asked to produce."""

    motivation: str = Field(
        description="The motivation and the main problem the paper sets out to solve."
    )
    key_findings: List[str] = Field(
        description="The most important findings / contributions, as concise bullet points."
    )
    methodology: str = Field(
        description="The methodology, approach, data, or experimental setup used."
    )
    future_work: str = Field(
        description="Future work, limitations, and open directions the paper suggests."
    )


class RQSnippet(BaseModel):
    """One piece of verbatim evidence pulled from the paper."""

    quote: str = Field(
        description="A short excerpt copied VERBATIM from the paper text that "
        "bears on the research question. Do not paraphrase."
    )
    why: str = Field(
        description="One sentence on how this excerpt relates to the research question."
    )


class RQAnswer(BaseModel):
    """The model's grounded response to a research question for one paper."""

    relevance: Literal["high", "medium", "low", "none"] = Field(
        description="How directly this paper addresses the research question. "
        "Use 'none' if the paper does not address it at all."
    )
    answer: str = Field(
        description="A direct answer to the research question based ONLY on this "
        "paper. If the paper does not address it, say so plainly here."
    )
    reasoning: str = Field(
        description="Brief reasoning that explains how the evidence in the paper "
        "leads to the answer."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Concise bullet-point findings from the paper that are "
        "relevant to the research question.",
    )
    snippets: List[RQSnippet] = Field(
        default_factory=list,
        description="Verbatim supporting excerpts from the paper, each with a "
        "short note on its relevance. Empty if relevance is 'none'.",
    )


_SYSTEM = (
    "You are a meticulous research assistant. You read academic papers and "
    "produce faithful, concise summaries grounded strictly in the provided text. "
    "Do not invent results. If a section's information is missing from the text, "
    "say so briefly rather than guessing."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM),
        (
            "human",
            "Summarize the following paper.\n\n"
            "Title: {title}\n"
            "Authors: {authors}\n\n"
            "--- PAPER TEXT (may be truncated) ---\n{text}\n--- END ---",
        ),
    ]
)


_RQ_SYSTEM = (
    "You are a meticulous research assistant helping with a literature review. "
    "Given a research question and the full text of one paper, you judge how "
    "relevant the paper is, answer the question using only this paper, and pull "
    "out the specific passages that support your answer. Ground every statement "
    "strictly in the provided text and quote snippets verbatim. If the paper does "
    "not address the question, set relevance to 'none' and say so — never invent "
    "an answer or a quote."
)

_RQ_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RQ_SYSTEM),
        (
            "human",
            "Research question:\n{rq}\n\n"
            "Judge how relevant the following paper is to that question, answer it "
            "using only this paper, and extract verbatim supporting snippets.\n\n"
            "Title: {title}\n"
            "Authors: {authors}\n\n"
            "--- PAPER TEXT (may be truncated) ---\n{text}\n--- END ---",
        ),
    ]
)


class Summarizer:
    def __init__(self, llm, model_label: str):
        self.model_label = model_label
        self._summary_chain = _PROMPT | llm.with_structured_output(PaperSummary)
        self._rq_chain = _RQ_PROMPT | llm.with_structured_output(RQAnswer)

    def summarize(self, title: str, authors: str, text: str) -> PaperSummary:
        return self._summary_chain.invoke(
            {"title": title, "authors": authors, "text": text}
        )

    def answer_rq(self, title: str, authors: str, text: str, rq: str) -> RQAnswer:
        return self._rq_chain.invoke(
            {"title": title, "authors": authors, "text": text, "rq": rq}
        )


def _para(text: str) -> str:
    """Escape text and preserve line breaks for HTML notes."""
    return html.escape(text).replace("\n", "<br/>")


def summary_to_html(title: str, summary: PaperSummary, model_label: str) -> str:
    """Render a PaperSummary as a Zotero note (HTML). Contains SUMMARY_MARKER
    via the footer string so re-runs can detect it."""
    findings = "".join(f"<li>{_para(f)}</li>" for f in summary.key_findings)
    if not findings:
        findings = "<li>—</li>"

    return (
        '<div data-schema-version="9">'
        f"<h1>{SUMMARY_HEADER}</h1>"
        f"<p><strong>{html.escape(title)}</strong></p>"
        "<h2>Motivation &amp; Main Problem</h2>"
        f"<p>{_para(summary.motivation)}</p>"
        "<h2>Key Findings</h2>"
        f"<ul>{findings}</ul>"
        "<h2>Methodology</h2>"
        f"<p>{_para(summary.methodology)}</p>"
        "<h2>Future Work</h2>"
        f"<p>{_para(summary.future_work)}</p>"
        f"<p><em>Generated by zotery using {html.escape(model_label)}.</em></p>"
        "</div>"
    )


def rq_answer_to_html(
    title: str, rq: str, answer: RQAnswer, model_label: str
) -> str:
    """Render an RQAnswer as a Zotero note (HTML). The RQ text is embedded so a
    re-run with the *same* question can detect and skip it."""
    findings = "".join(f"<li>{_para(f)}</li>" for f in answer.findings) or "<li>—</li>"
    snippets = (
        "".join(
            f"<li><blockquote>{_para(s.quote)}</blockquote>"
            f"<p>{_para(s.why)}</p></li>"
            for s in answer.snippets
        )
        or "<li>—</li>"
    )
    return (
        '<div data-schema-version="9">'
        f"<h1>{RQ_HEADER}</h1>"
        f"<p><strong>{html.escape(title)}</strong></p>"
        f"<p><strong>Research question:</strong> {_para(rq)}</p>"
        f"<p><strong>Relevance:</strong> {html.escape(answer.relevance)}</p>"
        "<h2>Answer</h2>"
        f"<p>{_para(answer.answer)}</p>"
        "<h2>Reasoning</h2>"
        f"<p>{_para(answer.reasoning)}</p>"
        "<h2>Findings</h2>"
        f"<ul>{findings}</ul>"
        "<h2>Supporting Snippets</h2>"
        f"<ul>{snippets}</ul>"
        f"<p><em>Generated by zotery for research question "
        f"&ldquo;{html.escape(rq)}&rdquo; using {html.escape(model_label)}.</em></p>"
        "</div>"
    )


def preview(summary: PaperSummary) -> str:
    """Plain-text preview for --dry-run output."""
    findings = "\n".join(f"    - {f}" for f in summary.key_findings)
    return (
        "  Motivation & Problem:\n"
        f"    {summary.motivation}\n"
        "  Key Findings:\n"
        f"{findings}\n"
        "  Methodology:\n"
        f"    {summary.methodology}\n"
        "  Future Work:\n"
        f"    {summary.future_work}"
    )


def rq_preview(rq: str, answer: RQAnswer) -> str:
    """Plain-text preview of an RQAnswer for --dry-run output."""
    findings = "\n".join(f"    - {f}" for f in answer.findings) or "    - —"
    snippets = (
        "\n".join(f'    > {s.quote}\n      ({s.why})' for s in answer.snippets)
        or "    —"
    )
    return (
        f"  Research question: {rq}\n"
        f"  Relevance: {answer.relevance}\n"
        "  Answer:\n"
        f"    {answer.answer}\n"
        "  Reasoning:\n"
        f"    {answer.reasoning}\n"
        "  Findings:\n"
        f"{findings}\n"
        "  Supporting snippets:\n"
        f"{snippets}"
    )
