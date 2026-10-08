# Document Q&A (RAG)

Ask questions about your own documents and get answers that **cite their sources**. It is a small, complete Retrieval-Augmented Generation (RAG) application: it splits documents into chunks, turns them into embeddings, stores them in a vector database, finds the passages closest in meaning to a question, and (optionally) has a large language model write an answer from those passages only.

**Stack:** Python, LangChain, ChromaDB, FastEmbed (local embeddings), Anthropic Claude (optional), Streamlit, pytest.

## Two modes

| | **Search-only mode** (default) | **Full AI mode** |
|---|---|---|
| Needs an API key | No | Yes, your own Anthropic API key |
| What you get | The most relevant passages from your documents, ranked, with file and page number | A written answer in plain language with citations like `[1]`, plus the passages |
| Cost | Free, runs fully on your machine | Billed by Anthropic per question (roughly one cent each on the default model; this is an estimate) |

Everything works without a key. Adding a key switches on written answers; nothing else changes.

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.10 or newer should work; tested on 3.14 only |
| Disk space | About 1 GB for the Python packages, plus about 65 MB for the embedding model |
| Internet | Needed once to install packages and to download the embedding model on first use. Search-only mode then works offline. Full AI mode needs internet for every question. |
| macOS / Linux / Windows | Developed on macOS |
| Anthropic API key | **Optional**, only for Full AI mode (see below) |

## Quick start (no API key)

```bash
git clone https://github.com/Ashishpawar6/genai-document-qa.git
cd genai-document-qa
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m docqa demo                 # downloads and indexes the "Attention Is All You Need" paper
python -m docqa ask "What is multi-head attention?"
streamlit run app/chat.py            # chat UI at http://localhost:8501
```

Add your own documents (PDF, `.txt` or `.md`):

```bash
python -m docqa ingest report.pdf notes.md
python -m docqa chat                 # interactive question loop
python -m docqa status               # shows the mode and what is stored
```

In the chat UI you can also upload files from the sidebar.

## Turn on Full AI mode (add your own API key)

The project ships with a placeholder for your key. The key is read from a local `.env` file that is **never committed** (it is in `.gitignore`).

1. **Get a key.** Create an account at [console.anthropic.com](https://console.anthropic.com/), add billing credit, then go to *Settings → API Keys* and create a key. It looks like `sk-ant-...`.
2. **Create your `.env` file** from the template:
   ```bash
   cp .env.example .env
   ```
3. **Paste the key** after `ANTHROPIC_API_KEY=` in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```
4. **Check it was picked up:**
   ```bash
   python -m docqa status
   ```
   You should see `Full AI mode: claude-opus-5-5 writes answers from the retrieved passages.`
5. **Ask a question.** `python -m docqa ask "..."` or restart the chat UI. The sidebar turns green and says *Full AI mode*.

You can also export the key as an environment variable instead of using a file: `export ANTHROPIC_API_KEY=sk-ant-...`.

**Keep your key private.** Never paste it into code, commit it, or share screenshots of `.env`. If it leaks, delete it in the Anthropic console and create a new one.

### Optional settings (in `.env`)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Model that writes answers. For lower cost try `claude-sonnet-5-5` or `claude-haiku-4-5`. |
| `ANTHROPIC_EFFORT` | `low` (on models that support it) | How much the model thinks before answering: `low`, `medium`, `high`, `xhigh`, `max`. Higher is slower and costs more. |
| `ANTHROPIC_MAX_TOKENS` | `8000` | Upper limit for one reply (thinking tokens count toward it). |
| `DOCQA_CHUNK_SIZE` / `DOCQA_CHUNK_OVERLAP` | `900` / `150` | Characters per chunk and how much neighbouring chunks overlap. |
| `DOCQA_TOP_K` | `4` | How many passages are retrieved per question. |
| `DOCQA_DATA_DIR` | `data` | Where the vector database and the embedding model are stored. |
| `DOCQA_EMBEDDINGS` | `fastembed` | `fastembed` (real model) or `hashing` (tiny offline stand-in used by the tests). |

### Troubleshooting

If a call to the language model fails, the app does not crash. It shows the retrieved passages and a note:

| Note shown | Cause and fix |
|---|---|
| "The Anthropic API key was rejected" | The key is wrong, revoked or has no credit. Check `.env`. |
| "The model was not found" | `ANTHROPIC_MODEL` is misspelled or not available to your account. |
| "The API rate limit was reached" | Wait a moment and ask again. |
| "Could not reach the Anthropic API" | No internet connection. |
| "The model declined to answer this request" | The model's safety system declined; rephrase the question. |

## How it works

```
 INGEST (once per file)                              ASK (every question)
 ┌──────────┐   ┌──────────┐   ┌────────────┐        ┌──────────┐   ┌─────────────┐
 │ PDF/TXT/ │──►│  split   │──►│  embed &   │        │ question │──►│  embed the  │
 │ Markdown │   │ into     │   │  store in  │        └──────────┘   │  question   │
 │ (pypdf)  │   │ chunks   │   │  ChromaDB  │◄───── similar? ───────┴─────────────┘
 └──────────┘   └──────────┘   └────────────┘                 │ top 4 passages
                                                              ▼
                       search-only mode  ◄── no key ──  prompt: "answer ONLY from these
                       shows the passages                numbered passages, cite them"
                                                              │ with a key
                                                              ▼
                                                    Claude writes the answer + [1][2] cites
```

- **Chunking** (`docqa/chunking.py`): `RecursiveCharacterTextSplitter` makes overlapping chunks of about 900 characters, keeping source and page. Chunk ids are a hash of source, page, position and text, so ingesting the same file twice adds nothing.
- **Embeddings** (`docqa/embeddings.py`): `BAAI/bge-small-en-v1.5` runs locally and turns text into a 384-number vector; texts with similar meaning get nearby vectors. No data leaves your machine in search-only mode.
- **Vector store** (`docqa/store.py`): ChromaDB with cosine similarity, saved on disk in `data/chroma`.
- **Answering** (`docqa/qa.py`): the question is embedded, the closest chunks are retrieved and placed in a LangChain prompt as numbered passages, and Claude answers from them only. The prompt tells the model to cite, to say so when the answer is not in the passages, and to treat passage text as data, not instructions (a defence against instructions hidden inside documents; it reduces that risk but cannot remove it).
- **Failure handling:** a refusal, an empty reply or any API error falls back to showing the passages with an explanatory note.

## Retrieval quality

`python -m docqa eval` checks whether the right passage comes back. It uses 12 questions about the demo paper (`eval/attention_questions.json`); a retrieved chunk counts as correct if it contains all the expected keywords. This measures **retrieval only**, not the written answers.

| Metric (12 questions, top 4 passages) | Result |
|---|---|
| Hit rate (right passage in the top 4) | **75%** (9 of 12) |
| Mean reciprocal rank | **0.75** (every hit was the very first result) |

Raising the number of passages did not change the hit rate (the same 9 questions hit at k = 1 through 8), so the 3 misses are not a cutoff problem:

- *"How does the model know the order of the words?"* The answer passage says "positional encodings"; the small embedding model does not connect that phrasing to "order of the words".
- *"Which regularization techniques were used?"* The passage is about dropout and label smoothing; closer-sounding training passages ranked higher.
- *"What is the size of the inner layer of the feed-forward network?"* The answer sits in a short passage and only ranked 9th.

These misses show the limits of a small embedding model with plain semantic search. Ideas to improve it are listed below. Results are for one paper and 12 questions, so treat them as a sanity check, not a benchmark.

## Project structure

```
docqa/
  config.py      settings from environment / .env
  loaders.py     PDF, text and Markdown readers
  chunking.py    chunk splitting and stable ids
  embeddings.py  local embedding model (+ a tiny offline stand-in for tests)
  store.py       ChromaDB vector store
  llm.py         Claude connection (only if a key is set)
  qa.py          retrieve, build prompt, generate, fall back
  evaluate.py    retrieval metrics
  pipeline.py    ingest and demo download helpers
  cli.py         command line (python -m docqa ...)
app/chat.py      Streamlit chat UI
eval/            retrieval test questions
tests/           75 automated tests
.env.example     template for your own API key and settings
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

75 tests cover loading, chunking, the vector store (including no duplicates and persistence), the prompt that is sent to the model, answer formatting, every API error path, the CLI and the Streamlit app. The tests use a small offline embedding and a **stand-in language model**, so they need no download and no API key and never call the paid API.

## Limitations

- **Full AI mode has not been run against the live API during development.** Its logic is tested with a stand-in model, and the request that would be sent to Claude is checked (no unsupported sampling parameters, explicit token limit, effort setting). Try it with your own key and report any problem.
- Each question is answered independently: there is no conversation memory, so follow-ups like "and the second one?" do not work.
- Scanned PDFs (images of text) are rejected because there is no OCR. Tables and figures are read as plain text.
- Plain semantic search can miss passages that use different wording (see the retrieval results).
- Single user, no authentication; documents are stored unencrypted on disk.
- If a request is declined by the model's safety system the app shows the passages instead of retrying with another model.

## Ideas for improvement

- Hybrid search (keyword BM25 plus embeddings) and a reranking model.
- A larger embedding model, and query rewriting before searching.
- Conversation memory and streaming answers.
- Retrieval evaluation on more documents, plus a check of answer faithfulness.
- Server-side refusal fallbacks and caching of repeated questions.

## Credits

The demo uses *Attention Is All You Need* (Vaswani et al., 2017), downloaded from arXiv when you run `python -m docqa demo`. It is not included in this repository.
