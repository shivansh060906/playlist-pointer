# Cherno C++ Lecture Search

Ask a question, get the exact moment in The Cherno's C++ series where it's explained, with a
jump-to-timestamp link, and optionally a short cited answer.

Hybrid RAG (dense + BM25 -> RRF -> cross-encoder rerank), three chunking strategies, and an eval
harness that measures **timestamp accuracy** so every design choice has a number behind it.

```
YouTube playlist
   | yt-dlp (list) + youtube-transcript-api (captions)  /  faster-whisper (fallback, C++-biased prompt)
   v
transcripts/*.json  (text + start + duration per caption line)
   | chunking: fixed | sliding | semantic (embedding-similarity topic shifts)
   v
chunks (video_id, start, end, text)
   |-- bge-small embeddings --> Chroma (cosine)
   '-- C++-aware tokenizer ---> BM25
                    |
query --(glossary expansion)--> BM25 top30 \
query ------------------------> dense top30 --> RRF fusion --> cross-encoder rerank --> dedupe overlaps
                                                                         |
                                       hits with seek_seconds + youtu.be link --> (optional) LLM answer with [n] citations
```

## Setup

Requires **Python 3.10+**. Work from `backend/`.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Optional but recommended: CPU-only torch is ~10x smaller than the default wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
```

Edit `.env`: put a free **Groq** (console.groq.com/keys) or **Gemini** (aistudio.google.com/apikey) key in,
and set `LLM_PROVIDER` to match. The key is only needed for `/api/ask`; search works without it.

### 1. Check the tests pass (offline, ~5 s)
```bash
pytest -q
```

### 2. Download transcripts
```bash
python -m scripts.ingest --limit 5      # smoke test; first title should be "Welcome to C++"
python -m scripts.ingest                # whole playlist, resumable (re-run to continue)
```
- If the first title isn't right, find the playlist URL yourself and pass `--playlist "<url>"`.
- Ingestion is polite (1-2.5 s between videos) because YouTube blocks scrapers. If you get an IP-block
  error, wait, change network, or use Whisper:
  ```bash
  pip install -r requirements-whisper.txt
  python -m scripts.ingest --whisper-fallback     # only for videos without captions
  python -m scripts.ingest --force-whisper        # everything (slow on CPU; best on GPU)
  ```

### 3. Build the indexes (first run downloads the embedding model)
```bash
python -m scripts.build_index --strategy all
```
Prints chunk counts, average words and average seconds per chunk for each strategy.

### 4. Try it from the terminal
```bash
python -m scripts.query "why should I use references instead of pointers"
python -m scripts.query "what is nullptr" --mode bm25 --expand        # opt in to glossary expansion
python -m scripts.query "explain move semantics" --rerank             # opt in to the cross-encoder
python -m scripts.query "explain move semantics" --ask                # needs the LLM key
```

### 5. Run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive docs.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | which indexes are built, whether an LLM key is configured |
| `GET /api/videos` | indexed videos in playlist order |
| `GET /api/search?q=...` | ranked moments. Params: `k`, `strategy` (fixed/sliding/semantic), `mode` (hybrid/dense/bm25), `rerank` (default off), `expand` (default off), `video_id` (repeatable), `group_by=video` |
| `POST /api/ask` | `{question, k?, strategy?, mode?, rerank?, expand?, video_ids?}` -> answer + citations |

The API defaults are **rerank off, expand off, hybrid RRF weighted 2:1** (`RRF_WEIGHTS=dense,bm25` in `.env`): in
our eval the reranker lowered hit@1 and added seconds, and glossary expansion showed no significant gain. Re-check
on your own questions.

Each hit includes `video_id, title, start, end, seek_seconds, timestamp, url, text, score` and per-stage
ranks (`dense_rank`, `bm25_rank`, `rerank_score`). `seek_seconds` is what the React player should
`seekTo()`. It's the chunk start minus a 2 s lead-in (captions lag speech). Responses also carry
`timings_ms` per stage.

`group_by=video` answers "which episodes cover X?": it retrieves a deeper pool and returns one entry per
episode with its best moment.

CORS is open to `localhost:5173` (Vite) and `localhost:3000` by default (`CORS_ORIGINS`).

## Building the eval set (the part that makes this CV-worthy)

You need ~50 questions with known answer timestamps.

```bash
python -m scripts.peek --videos                       # list video ids
python -m scripts.peek "null pointer"                 # every occurrence, timestamped
python -m scripts.peek --video VIDEO_ID --at 12:30    # read the transcript around a moment
cp eval/questions.example.json eval/questions.json    # then fill in your questions
```
Write questions the way a *student* would ask them ("why does my linker say unresolved external symbol"),
not copied from the transcript. Otherwise BM25 wins trivially. Include some that use different words than the
lecturer did.

```bash
python -m scripts.eval                       # every built strategy x 10 retrieval configurations
python -m scripts.eval --skip-rerank         # much faster: drops the slow cross-encoder rows
python -m scripts.eval --tol 15              # stricter timestamp tolerance (sensitivity check)
python -m scripts.eval --misses-strategy sliding --misses-config "dense only"   # analyse another config
```
Outputs in `eval/results/`:

| file | what it is |
|---|---|
| `results.md` / `results.json` | hit@1/3/5, MRR, median seek error, chunk length and latency per (chunking x configuration), with **95% bootstrap CIs** on hit@1 and MRR. This is your headline table. |
| `results.md` (lower sections) | **paired comparisons**: every configuration vs a baseline (default "dense only"), and every chunking strategy vs `fixed`, as question-by-question differences with CIs. `*` means the CI excludes 0. |
| `misses.md` / `misses.json` | every question that missed at hit@1 for one configuration (default semantic / "hybrid + glossary"), classified and shown next to what was retrieved and what the index holds at the labelled moment. |
| `results.md` (effects) | **isolated effects**: glossary on vs off and weighted vs equal-weight RRF, each as a paired comparison against its own control (not against dense). The seek-error column shows a bootstrap CI on the median. |
| `per_question.json`, `paired.json` | raw per-question ranks and paired stats, for your own analysis. |

Configurations include weighted RRF (`hybrid RRF 2:1`, `3:1` = dense weighted 2x/3x over BM25), because equal-weight
fusion lets a weak BM25 list drag a strong dense list down.

**Reading the numbers honestly**
- With ~50 questions each question moves hit@1 by 2 points, so most gaps between configurations are within noise.
  Trust a difference only if the paired CI excludes 0, and expect about 1 in 20 starred results to be chance
  (there are many comparisons). More questions is the real fix; ~100+ makes the intervals much tighter.
- A chunk counts as a hit if it's in the right video and its span +/- tolerance contains the labelled moment.
  Bigger chunks contain more moments, so hit@k flatters them. Compare it against `avg chunk (s)` and
  `median seek err (s)` (how far from the true moment a click lands). The best strategy is accurate *and* lands close.

**Working through `misses.md`:** misses are sorted so the cheapest fixes come first.
1. `label: video not indexed` / `label: timestamp past end of video` are mistakes in `questions.json`. Fix them and re-run
   before drawing any conclusions.
2. `hit below rank 1`: the right moment was retrieved but ranked lower. Look at what outranked it.
3. `right video, wrong moment`: topic found, wrong spot; usually chunking or a lecturer revisiting a topic. Compare the
   "Expected" snippet with the retrieved ones.
4. `wrong video`: usually vocabulary mismatch (the lecturer never says the word your question uses) or a concept covered
   in several videos (add the other occurrence to that question's `answers`).

## Design notes (interview talking points)

- **Chunks keep real timestamps** and snap to caption boundaries, so a citation is always a playable moment.
- **Title is prepended to every chunk** before embedding/BM25/rerank. Titles like "Templates in C++" carry
  topic signal that mid-lecture captions lack.
- **Semantic chunking** is TextTiling-style: embed ~30-word blocks, compare the mean of the 2 blocks before vs
  after each gap, cut at local similarity minima below the 25th percentile, bounded to 120-320 words.
- **BM25 tokenizer is C++-aware**: keeps `std::vector` whole (and its parts), maps `c++`/`c plus plus` to `cpp`,
  stems plurals. A generic tokenizer destroys exactly the tokens keyword search exists for.
- **Glossary query expansion** repairs ASR mangling on the keyword side (`cout` <-> "see out", `lvalue` <-> "l value",
  `nullptr` <-> "null pointer") without touching transcripts. It's a toggle so the eval measures what it buys.
- **RRF** fuses by rank only, so there's no need to normalise BM25 scores against cosine similarities.
- **Cross-encoder rerank runs on 30 candidates**, not the corpus: precision where it matters, bounded latency.
- **Overlap dedupe** stops sliding windows from returning five near-identical neighbours.
- **Citations are validated**: `[n]` numbers outside the retrieved set are dropped rather than trusted.
- **Whisper is biased with a C++ prompt** (`initial_prompt`), and transcript `source` is stored so you can ablate
  YouTube auto-captions vs Whisper by pointing `DATA_DIR` at two different folders.

## Auditing your labels (pooled relevance judgments)

One labelled timestamp per question under-counts: lecturers answer the same thing in several places, so valid
retrievals show up as "misses". The audit finds them without favouring any system (TREC-style pooling):

```bash
python -m scripts.audit build     # pool the top 3 of EVERY configuration, minus what already counts as correct
python -m scripts.audit judge     # y/n per excerpt; shuffled, config and rank hidden; resumable (q to stop)
python -m scripts.audit apply     # writes eval/questions.audited.json
python -m scripts.eval --questions eval/questions.audited.json --out eval/results_audited
```
Rules that keep it defensible: judge **blind** (do not peek at `found_by` in `candidates.json` first), pool **all**
configurations rather than only the best one, judge every candidate (unjudged ones count as not relevant), and
**report both** the original and audited numbers plus how many alternatives you added. Judge "would this excerpt on
its own be a good answer?", not "is it the best one".

## Layout

```
backend/
  app/
    ingest/youtube.py      playlist listing, captions, whisper fallback
    chunking.py            fixed / sliding / semantic
    index/                 Chroma + BM25 + jsonl store, index builder
    retrieval/             tokenizer, glossary, fusion, reranker, embedder, hybrid retriever
    llm.py                 Groq/Gemini (OpenAI-compatible) client, prompt, citation parsing
    engine.py              owns models + retrievers
    main.py, schemas.py    FastAPI app
    evaluation.py          timestamp-aware metrics, bootstrap CIs, miss classification
    eval_runner.py         ablation grid, paired comparisons, reports (used by scripts/eval.py)
    audit.py               pooled relevance judgments (build / judge / apply)
  scripts/                 ingest, build_index, query, peek, eval, audit
  eval/                    questions.example.json -> questions.json, results/
  tests/                   71 offline tests (fake embedder/reranker; no model downloads)
```

## Troubleshooting

- **`RequestBlocked` / `IpBlocked` during ingest:** YouTube is blocking your IP. Wait, switch network/VPN, or use Whisper.
- **503 "Index ... has not been built":** run `python -m scripts.build_index --strategy <name>`.
- **503 "No API key":** set `GROQ_API_KEY` or `GEMINI_API_KEY` and `LLM_PROVIDER` in `.env`.
- **LLM 400/404 about the model:** free-tier model names change. Update `GROQ_MODEL` / `GEMINI_MODEL`.
- **Slow first request:** models load lazily; the server warms them in the background at startup (`WARMUP=0` to disable).
- **Semantic build is slow:** it embeds every ~30-word block of every video. A few minutes on CPU is normal.

## Next

1. React page: search box, result cards, embedded YouTube player that `seekTo(seek_seconds)` on click, "Ask" tab.
2. Eval additions: RAGAS-style faithfulness for `/api/ask`, and a Whisper-vs-captions ablation.
3. Multi-video "do they agree?" queries on top of `group_by=video`.
