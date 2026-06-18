# zotery

[![PyPI](https://img.shields.io/pypi/v/zotery.svg)](https://pypi.org/project/zotery/)
[![Python](https://img.shields.io/pypi/pyversions/zotery.svg)](https://pypi.org/project/zotery/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Scan a **Zotero** collection, read each paper's attached **PDF**, run an **LLM**
(DeepSeek, Google Gemini, or a local Ollama model) over it, and write the result
back into Zotero as a **child note** on the paper.

## Contents

- [Quick start](#quick-start)
- [How it connects to Zotero](#how-it-connects-to-zotero)
- [Install](#install)
- [Configure the LLM](#configure-the-llm)
- [Usage](#usage)
  - [Answer a research question (`--rq`)](#answer-a-research-question---rq)
  - [Run locally & free with Ollama](#run-locally--free-with-ollama)
- [How it works](#how-it-works)
- [Notes & limits](#notes--limits)

## Quick start

```bash
# 1. Install
pip install zotery

# 2. Preview — no API key, no Zotero account, no writes (just needs Zotero 7 running)
zotery "Your Collection Name" --dry-run --limit 2

# 3. Write notes — needs a Zotero Web API key (see Configure below)
zotery "Your Collection Name"
```

> 💡 **Zero-config preview:** With the Zotero 7 desktop app running, you can
> immediately test summarization on any collection — no API keys, no `.env`:
> `zotery "My Collection" --dry-run --limit 3`. This reads PDFs straight
> from your local Zotero storage. Writing notes back requires the Web API
> (see [Configure Zotero](#configure-zotero-env)).

> Install it as **`zotery`**, run it as **`zotery`**. (The Python module is
> `zotero_summarizer`.)

Two modes:

1. **Summary** (default) — a structured 4-part summary of each paper:
   **Motivation & Main Problem · Key Findings · Methodology · Future Work**.
2. **Research question** (`--rq "..."`) — instead of a generic summary, the model
   reads each paper *against your question* and writes back: a **relevance**
   judgement (high/medium/low/none), a grounded **answer**, its **reasoning**,
   **findings**, and **verbatim supporting snippets** quoted from the paper. Great
   for screening a large collection during a literature review.

The pipeline is orchestrated with **LangGraph**:

```
START → load_items → process_paper → summarize → write_note → END
                          ↑__________________________|   (loops per paper)
```

`load_items` scans the collection · `process_paper` finds + downloads + extracts
the PDF · `summarize` calls the LLM for a structured `PaperSummary` (or an
`RQAnswer` in `--rq` mode) · `write_note` renders it to HTML and pushes it to
Zotero.

### What you get

Each paper gets a note in Zotero with four structured sections:

> **Motivation & Main Problem** — The authors address the challenge of ...
> **Key Findings** — Experiments show that ...
> **Methodology** — Using a dataset of ... with a transformer-based ...
> **Future Work** — The paper suggests extending the approach to ...

With `--rq`, notes instead include a **relevance** rating (high/medium/low/none),
a grounded **answer**, model **reasoning**, **findings**, and **verbatim
snippets** quoted from the paper. Papers that don't address your question come
back with relevance **`none`** and a one-line note — handy for screening a
large collection quickly.

## How it connects to Zotero

It uses [**pyzotero**](https://github.com/urschrei/pyzotero) as the connector,
which speaks to both Zotero APIs:

- **Web API** (`ZOTERO_LOCAL=false`) — the Zotero cloud library, via an API key.
  **Required to write notes back**, because Zotero's local API is read-only.
  Needs Zotero Sync turned on (so the library exists on zotero.org) and a
  write-enabled key. PDFs are still read locally from disk (see
  `ZOTERO_STORAGE_DIR`), so you do **not** need Zotero file sync.
- **Local API** (`ZOTERO_LOCAL=true`) — the running Zotero 7 desktop app. No API
  key, reads PDFs straight off disk. Good for read-only previews (`--dry-run`),
  but **cannot write notes** (the local API rejects writes).

> Prefer an MCP server? The summarization core (`summarizer.py` + `graph.py`) is
> independent of how items are fetched, so you can swap `zotero_client.py` for a
> Zotero MCP client. pyzotero is the default because it needs no extra service
> and reads local PDFs directly.

## Install

Requires **Python 3.10+** (the LangChain stack no longer supports 3.9).

From PyPI (current version **0.1.1**):

```bash
pip install zotery
# or, with uv:
uv tool install zotery      # installs the `zotery` command globally
```

This puts the `zotery` command on your PATH. Then create a config file
from the template and edit it (see below):

```bash
curl -O https://raw.githubusercontent.com/mkassaf/zotero-summarizer/main/.env.example
mv .env.example .env
# edit .env, or export the variables in your shell instead
```

> `.env` is optional — every setting can also come from real environment
> variables or CLI flags. The downloaded template defaults to
> `ZOTERO_LOCAL=true` (read-only preview); flip it to `false` and fill in the
> Web API fields when you're ready to write notes. See
> [Configuration](#configure-the-llm) below.

<details>
<summary>Install from source (for development)</summary>

```bash
git clone https://github.com/mkassaf/zotero-summarizer.git
cd zotero-summarizer

python3 -m venv .venv
source .venv/bin/activate
pip install -e .            # or: uv sync

cp .env.example .env
# then edit .env  (see below)
```
</details>

### Configure Zotero (`.env`)

To write notes you need the **Web API**:

1. **Turn on sync:** Zotero → *Settings → Sync* → log in. This puts your library
   metadata on zotero.org so the API can see it. (File sync is optional — PDFs
   are read locally.)
2. **Create a write-enabled key:** <https://www.zotero.org/settings/keys/new> —
   check **"Allow library access"** *and* **"Allow write access"**.

```ini
ZOTERO_LOCAL=false
ZOTERO_LIBRARY_TYPE=user
ZOTERO_LIBRARY_ID=your-username      # username OR numeric userID both work
ZOTERO_API_KEY=your-write-key

# Optional: where PDFs live on disk. Auto-detected to ~/Zotero/storage if unset.
# ZOTERO_STORAGE_DIR=/Users/you/Zotero/storage
```

`ZOTERO_LIBRARY_ID` accepts your **username** — it's resolved to the numeric id
the Web API requires, using your API key. The numeric id works too.

### Configure the LLM

Pick one provider:

| Provider | Settings | Standard key env var | Notes |
|----------|----------|----------------------|-------|
| **DeepSeek** (default) | `LLM_PROVIDER=deepseek`<br>`LLM_MODEL=deepseek-chat` | `DEEPSEEK_API_KEY` | Key from <https://platform.deepseek.com>. |
| **Google Gemini** | `LLM_PROVIDER=google`<br>`LLM_MODEL=gemini-2.5-flash` | `GOOGLE_API_KEY` | Fast, recommended for big runs. |
| **OpenAI-compatible** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=gpt-4o-mini`<br>`LLM_BASE_URL=...` | `OPENAI_API_KEY` | OpenAI, Together, vLLM, etc. |
| **Ollama (local, free)** | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=qwen3:8b` | *(none)* | Needs Ollama running + `ollama pull qwen3:8b`. Native JSON-schema output. Slower per paper. |

> The `LLM_PROVIDER` value accepts shorthand aliases: `gemini` or
> `google-genai` → `google`; `openai-compatible` or `compatible` → `openai`.

#### Where the API key comes from

The LLM key is resolved in this order — **first match wins**:

1. **CLI flag** — `--llm-api-key sk-...` (highest precedence; never written to disk).
2. **Generic override** — `LLM_API_KEY` (works for any provider).
3. **Provider's standard env var** — `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, or
   `GOOGLE_API_KEY` (see the table). Use these if you already export your keys
   globally in your shell — nothing extra to configure here.

The Zotero key works the same way: `--zotero-api-key` overrides `ZOTERO_API_KEY`.

```bash
# Example: provider + key entirely from the command line, no .env needed
zotery "Literature Review" \
  --llm-api-key "$MY_KEY" --zotero-api-key "$ZKEY"

# Example: rely on a globally-exported key (e.g. in ~/.zshrc)
export OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini zotery "Literature Review"
```

> Ollama tip: the default base URL is `http://127.0.0.1:11434`. Use `127.0.0.1`,
> not `localhost` — `localhost` can resolve to IPv6/Docker and miss your models.

## Usage

After `pip install zotery`, use the `zotery` command (or, from a source
checkout, `python -m zotero_summarizer`):

```bash
# Summarize every paper in a collection (by name or 8-char key)
zotery "Literature Review"

# Preview first: generate + print summaries, write nothing
zotery "Literature Review" --dry-run --limit 3

# Re-summarize papers that already have an AI note
zotery ABCD1234 --force
```

Override the provider per-run without editing `.env`:

```bash
LLM_PROVIDER=google LLM_MODEL=gemini-2.5-flash zotery "Literature Review"
```

Flags:

| flag                  | meaning                                                        |
|-----------------------|----------------------------------------------------------------|
| `--rq "QUESTION"`     | answer a research question per paper instead of summarizing    |
| `--note-title "..."`  | custom heading/title for the written note (see below)          |
| `--limit N`           | only process the first N papers                                |
| `--dry-run`           | generate and print results, but don't write notes to Zotero    |
| `--force`             | redo papers that already have a matching note                  |
| `--llm-api-key KEY`   | LLM API key; overrides `LLM_API_KEY` and the provider env var  |
| `--zotero-api-key KEY`| Zotero Web API key; overrides `ZOTERO_API_KEY`                 |

**Custom note title.** Zotero shows a note's first heading as its title. By
default that's `🤖 AI Summary` (or `🤖 Research-Question Analysis` with `--rq`);
override it with `--note-title` to make notes easy to spot in a project:

```bash
zotery "SW agentic arch" --note-title "📌 LitReview 2026 — multi-agent" \
  --rq "What architectural patterns are proposed?"
```

The title is purely cosmetic — re-run detection keys off a hidden marker in the
note footer, not the title, so changing it never breaks the idempotency check.

Re-runs are **idempotent**: papers that already have a matching note are skipped
unless you pass `--force` (summary notes and per-question RQ notes are tracked
separately, so a summary and several different `--rq` runs can coexist).

### Answer a research question (`--rq`)

Screen a collection against a specific question. For every paper, zotery writes a
note with a **relevance** rating, a grounded **answer**, the model's
**reasoning**, **findings**, and **verbatim snippets** quoted from the paper:

```bash
# Always preview first — see the answers without touching your library
zotery "SW agentic arch" --dry-run --limit 5 \
  --rq "What architectural patterns are proposed for multi-agent LLM systems?"

# Looks good? Run it for real (writes one RQ note per paper)
zotery "SW agentic arch" \
  --rq "What architectural patterns are proposed for multi-agent LLM systems?"

# Locally & free with Ollama
LLM_PROVIDER=ollama LLM_MODEL=qwen3:8b zotery "SW agentic arch" \
  --rq "How is agent reliability evaluated?"
```

Each note is tagged with its exact question, so you can ask **several different
questions** over the same collection and each produces its own note. Papers that
don't address the question come back with relevance **`none`** and a one-line
note saying so — handy for quickly excluding irrelevant papers.

> Tip: papers without an extractable PDF (scanned/image-only, or no attachment)
> are skipped in both modes. Use `--dry-run --limit N` to sanity-check output
> quality before a large run.

### Run locally & free with Ollama

[Ollama](https://ollama.com) runs an LLM on your own machine — **no API key, no
per-token cost, nothing leaves your computer**. Good for private libraries or
large runs you don't want to pay for. It's slower per paper and quality depends
on the model you pick.

**1. Install Ollama and pull a model** (a ~5 GB instruct model is a good start):

```bash
# Install: https://ollama.com/download  (or `brew install ollama` on macOS)
ollama serve            # start the server (skip if the desktop app is running)
ollama pull qwen3:8b    # download the model
```

**2. Point zotery at Ollama and run:**

```bash
# One-off, all on the command line (no .env edits, no key needed):
LLM_PROVIDER=ollama LLM_MODEL=qwen3:8b zotery "Literature Review"

# Safe first run: print summaries without writing notes to Zotero
LLM_PROVIDER=ollama LLM_MODEL=qwen3:8b zotery "Literature Review" --dry-run --limit 3
```

Or set it once in `.env` and just run `zotery "Literature Review"`:

```ini
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:8b
# LLM_BASE_URL=http://127.0.0.1:11434   # optional; this is the default
```

Notes:

- **No `--llm-api-key` / `LLM_API_KEY` needed** — Ollama is keyless. (You still
  need a Zotero Web API key to *write* notes; use `--dry-run` to skip that.)
- Use any model you've pulled — e.g. `LLM_MODEL=llama3.1:8b`,
  `LLM_MODEL=mistral`. Bigger models give better summaries but run slower.
- If Ollama runs on another host/port, set `LLM_BASE_URL` (e.g.
  `http://192.168.1.10:11434`). Use `127.0.0.1`, **not** `localhost` — `localhost`
  can resolve to IPv6/Docker and miss your local models.

## How it works

| file                | responsibility                                                  |
|---------------------|-----------------------------------------------------------------|
| `config.py`         | load `.env`; build the LLM (DeepSeek / Google / Ollama / OpenAI) |
| `zotero_client.py`  | list collection papers, find/download PDFs, write notes         |
| `pdf_utils.py`      | extract text from PDF bytes                                     |
| `summarizer.py`     | prompts + structured output (`PaperSummary` / `RQAnswer`) + note HTML |
| `graph.py`          | the LangGraph pipeline                                          |
| `cli.py`            | argument parsing and the run report                            |

## Notes & limits

- **Writing requires the Web API.** The local API is read-only; use it only for
  reading/`--dry-run`.
- **Scanned/image-only PDFs** yield no text and are skipped (no OCR).
- Long PDFs are truncated to `MAX_PDF_CHARS` (default 48k chars, configurable in
  `.env`) to stay within the model's context window. Raise or lower it to match
  your model's capacity.
- PDFs are fetched via the API, falling back to `ZOTERO_STORAGE_DIR` (the local
  `storage/` folder, auto-detected at `~/Zotero/storage`). This means Web API
  mode works **without** Zotero file sync.
- Never commit your `.env` — it holds your API keys (it's already in
  `.gitignore`).

## Contributing

Bug reports and pull requests are welcome —
[open an issue](https://github.com/mkassaf/zotero-summarizer/issues).
