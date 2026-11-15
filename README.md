# Playlist Search

Search any YouTube playlist (built around The Cherno's C++ series) and jump to the exact moment in the video.
Optionally get a short answer with citations. Off-topic questions are refused instead of answered with junk.

- **Backend:** FastAPI, hybrid retrieval (embeddings + BM25), optional LLM answers (Groq or Gemini)
- **Frontend:** React + Vite + Tailwind + shadcn/ui

## Quick start

You need **Python 3.10+** and **Node 20+**.

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
```

Open `.env` and set an LLM key. It is only needed for the **Ask** tab; search works without one.

```
LLM_PROVIDER=gemini                  # or groq
GEMINI_API_KEY=your_key              # free: https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-flash-latest
```
(Groq: `LLM_PROVIDER=groq`, `GROQ_API_KEY=...`, free key at https://console.groq.com/keys.)

### 2. Get the data (first playlist)

```bash
python -m scripts.ingest             # downloads captions (resumable; try --limit 5 first)
python -m scripts.build_index --strategy semantic
```
The first run downloads the embedding model (about 130 MB). Use `--playlist "<url>"` on `ingest` for a different
playlist, or skip this step and add playlists from the UI (see below).

### 3. Run

```bash
# terminal 1, in backend/
uvicorn app.main:app --port 8000

# terminal 2, in frontend/
npm install
npm run dev                          # open http://localhost:5173
```

## Using the app

- **Find moments** shows transcript passages. Click a timestamp and the player jumps there.
- **Ask a question** writes a short answer from the top passages, with clickable `[n]` citations.
- **Playlist menu (top right)** switches playlists or adds one: paste a playlist link and wait for the progress bar.
  Each playlist is stored in `backend/data/playlists/<id>/`; your first playlist stays in `backend/data/`.
- **All videos** limits the search to some videos. **Options** holds search mode, result count and extras.
- `/` focuses the search box.

## Settings (`backend/.env`)

| Setting | Meaning |
|---|---|
| `LLM_PROVIDER`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `*_MODEL` | LLM used by Ask |
| `DEFAULT_STRATEGY` | how transcripts are split: `semantic` (default), `fixed` or `sliding` |
| `MIN_RELEVANCE` | off-topic cutoff, 0 to 1 (default 0.55). Higher refuses more; 0 turns it off |
| `MAX_PLAYLIST_VIDEOS` | cap on playlists added from the UI (default 60) |
| `PLAYLIST_URL` | playlist used by `scripts.ingest` |
| `DATA_DIR` | where transcripts and indexes are stored (default `./data`) |
| `CORS_ORIGINS` | allowed frontend origins |

Changes to `.env` need a backend restart.

## Troubleshooting

- **Ask tab is greyed out:** no LLM key is set in `backend/.env`.
- **Ask shows `401`:** the API key is wrong. **`404`/model error:** the model name was retired; list your models in
  the provider's dashboard and update `GEMINI_MODEL` / `GROQ_MODEL`. **`429`/`503`:** rate limit or overload; retry shortly.
- **Real questions get "doesn't look related":** lower `MIN_RELEVANCE` (try 0.45), or run
  `python -m scripts.calibrate --questions eval/questions.json` to get a suggested value.
- **503 "Index ... has not been built":** run `python -m scripts.build_index --strategy semantic`.
- **Captions fail to download (`RequestBlocked` / `IpBlocked`):** YouTube blocks some networks and all cloud servers.
  Wait, switch network, or run locally. For videos without captions: `pip install -r requirements-whisper.txt` then
  `python -m scripts.ingest --whisper-fallback`.
- **Adding a playlist from the UI fails:** it needs English captions; videos without them are skipped.
- **Port 8000 already in use:** start on another port (`--port 8001`) and run the frontend with
  `VITE_PROXY_TARGET=http://localhost:8001 npm run dev`.
- **Slow first request:** the models load on first use.

## Optional: command line tools and evaluation

```bash
python -m scripts.query "why use references instead of pointers"     # search from the terminal (add --ask for an answer)
pytest -q                                                             # offline tests, no model downloads
python -m scripts.eval                                                # accuracy tables (needs eval/questions.json, see questions.example.json)
```
`scripts.audit` re-checks your eval labels. API docs are at http://localhost:8000/docs while the backend runs.