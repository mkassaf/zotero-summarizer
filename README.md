# zotero-summarizer

Scan a **Zotero** collection, read each paper's attached **PDF**, generate a
structured summary with an **LLM** (DeepSeek, Google Gemini, or a local Ollama
model), and write that summary back into Zotero as a **child note** on the paper.

Every summary contains four sections:

- **Motivation & Main Problem**
- **Key Findings**
- **Methodology**
- **Future Work**

The pipeline is orchestrated with **LangGraph**:

```
START → load_items → process_paper → summarize → write_note → END
                          ↑__________________________|   (loops per paper)
```

`load_items` scans the collection · `process_paper` finds + downloads + extracts
the PDF · `summarize` calls the LLM for a structured `PaperSummary` · `write_note`
renders it to HTML and pushes it to Zotero.

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

## Setup

Requires **Python 3.10+** (the LangChain stack no longer supports 3.9).

```bash
git clone https://github.com/<you>/zotero-summarizer.git
cd zotero-summarizer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env  (see below)
```

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

### Configure the LLM (`.env`)

Pick one provider:

| Provider | `.env` | Notes |
|----------|--------|-------|
| **Google Gemini** | `LLM_PROVIDER=google`<br>`LLM_MODEL=gemini-2.5-flash` | Key read from the `GOOGLE_API_KEY` env var. Fast, recommended for big runs. |
| **DeepSeek** | `LLM_PROVIDER=deepseek`<br>`LLM_MODEL=deepseek-chat`<br>`LLM_API_KEY=sk-...` | Key from <https://platform.deepseek.com>. |
| **Ollama (local, free)** | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=qwen3:8b` | Needs Ollama running + `ollama pull qwen3:8b`. Uses native JSON-schema output. Slower per paper. |
| **OpenAI-compatible** | `LLM_PROVIDER=openai`<br>`LLM_MODEL=...`<br>`LLM_API_KEY=...`<br>`LLM_BASE_URL=...` | OpenAI, Together, vLLM, etc. |

> Ollama tip: the default base URL is `http://127.0.0.1:11434`. Use `127.0.0.1`,
> not `localhost` — `localhost` can resolve to IPv6/Docker and miss your models.

## Usage

```bash
# Summarize every paper in a collection (by name or 8-char key)
python -m zotero_summarizer "Literature Review"

# Preview first: generate + print summaries, write nothing
python -m zotero_summarizer "Literature Review" --dry-run --limit 3

# Re-summarize papers that already have an AI note
python -m zotero_summarizer ABCD1234 --force
```

Override the provider per-run without editing `.env`:

```bash
LLM_PROVIDER=google LLM_MODEL=gemini-2.5-flash python -m zotero_summarizer "Literature Review"
```

Flags:

| flag        | meaning                                                        |
|-------------|----------------------------------------------------------------|
| `--limit N` | only process the first N papers                                |
| `--dry-run` | generate and print summaries, but don't write notes to Zotero  |
| `--force`   | re-summarize even if an AI summary note already exists         |

Re-runs are **idempotent**: papers that already have an AI summary note are
skipped unless you pass `--force`.

## How it works

| file                | responsibility                                                  |
|---------------------|-----------------------------------------------------------------|
| `config.py`         | load `.env`; build the LLM (DeepSeek / Google / Ollama / OpenAI) |
| `zotero_client.py`  | list collection papers, find/download PDFs, write notes         |
| `pdf_utils.py`      | extract text from PDF bytes                                     |
| `summarizer.py`     | prompt + structured (`PaperSummary`) output + note HTML         |
| `graph.py`          | the LangGraph pipeline                                          |
| `cli.py`            | argument parsing and the run report                            |

## Notes & limits

- **Writing requires the Web API.** The local API is read-only; use it only for
  reading/`--dry-run`.
- **Scanned/image-only PDFs** yield no text and are skipped (no OCR).
- Long PDFs are truncated to `MAX_PDF_CHARS` (default 48k chars) to stay within
  the model's context window.
- PDFs are fetched via the API, falling back to `ZOTERO_STORAGE_DIR` (the local
  `storage/` folder, auto-detected at `~/Zotero/storage`). This means Web API
  mode works **without** Zotero file sync.
- Never commit your `.env` — it holds your API keys (it's already in
  `.gitignore`).
