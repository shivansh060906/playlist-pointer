# Design and evaluation notes

Everything that isn't needed to *run* the project but explains how it works and why. Setup lives in the main README.

## Architecture

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
query ------------------------> dense top30 --> weighted RRF --> (cross-encoder rerank) --> dedupe overlaps
                                                      |                                         |
                                       top dense cosine --> out-of-scope check        hits with seek_seconds + youtu.be link
                                                                                                |
                                                                          (optional) LLM answer with [n] citations
```

Defaults: hybrid, RRF weighted 2:1 (dense:BM25), rerank off, glossary expansion off. In the evaluation the reranker
lowered hit@1 and added seconds, and glossary expansion showed no significant gain. Re-check on your own questions.

## Design decisions

- **Chunks keep real timestamps** and snap to caption boundaries, so a citation is always a playable moment.
- **Title is prepended to every chunk** before embedding/BM25/rerank. Titles like "Templates in C++" carry topic
  signal that mid-lecture captions lack.
- **Semantic chunking** is TextTiling-style: embed ~30-word blocks, compare the mean of the 2 blocks before vs after
  each gap, cut at local similarity minima below the 25th percentile, bounded to 120-320 words.
- **BM25 tokenizer is C++-aware**: keeps `std::vector` whole (and its parts), maps `c++`/`c plus plus` to `cpp`,
  stems plurals. A generic tokenizer destroys exactly the tokens keyword search exists for.
- **Glossary query expansion** repairs ASR mangling on the keyword side (`cout` <-> "see out", `lvalue` <-> "l value",
  `nullptr` <-> "null pointer") without touching transcripts. It's a toggle so the eval measures what it buys.
- **RRF** fuses by rank only, so BM25 scores never need normalising against cosine similarities. Weighted RRF
  (default 2:1) stops a weak BM25 list dragging a strong dense list down.
- **Cross-encoder rerank runs on 30 candidates**, not the corpus: precision where it matters, bounded latency.
- **Overlap dedupe** stops sliding windows from returning five near-identical neighbours.
- **Citations are validated**: `[n]` numbers outside the retrieved set are dropped rather than trusted.
- **Whisper is biased with a C++ prompt** (`initial_prompt`), and the transcript `source` is stored so YouTube
  auto-captions vs Whisper can be ablated by pointing `DATA_DIR` at two folders.

## Results

50 hand-written student-style questions, tolerance +/-30 s, 95% bootstrap CIs over questions. Full tables:
`eval/results/` (original labels) and `eval/results_audited/` (after the relevance audit below).

hit@1 (95% CI) / MRR. "Original" counts only the one labelled timestamp per question; "audited" also counts other
excerpts judged to answer the question (315 pooled candidates judged, 52 relevant).

| chunking | configuration | original hit@1 | audited hit@1 | audited hit@5 | audited MRR | p50 latency |
|---|---|---|---|---|---|---|
| fixed | dense only | 0.68 (0.56-0.80) | 0.88 (0.78-0.96) | 1.00 | 0.926 | 18 ms |
| fixed | bm25 only | 0.48 (0.36-0.62) | 0.66 (0.54-0.78) | 0.92 | 0.760 | 0 ms |
| fixed | hybrid RRF 2:1 | 0.62 (0.48-0.76) | 0.82 (0.70-0.92) | 1.00 | 0.900 | 19 ms |
| fixed | hybrid + rerank | n/a | 0.86 (0.76-0.94) | 0.96 | 0.895 | 613 ms |
| sliding | dense only | 0.74 (0.62-0.86) | 0.94 (0.88-1.00) | 1.00 | 0.967 | 16 ms |
| sliding | bm25 only | 0.56 (0.42-0.70) | 0.68 (0.54-0.80) | 0.92 | 0.793 | 1 ms |
| sliding | hybrid RRF 2:1 | 0.72 (0.60-0.84) | 0.90 (0.82-0.98) | 0.98 | 0.940 | 18 ms |
| sliding | hybrid + rerank | n/a | 0.82 (0.72-0.92) | 0.98 | 0.881 | 591 ms |
| semantic | dense only | 0.72 (0.60-0.84) | 0.86 (0.76-0.96) | 1.00 | 0.917 | 17 ms |
| semantic | bm25 only | 0.56 (0.42-0.70) | 0.68 (0.56-0.80) | 0.90 | 0.782 | 0 ms |
| semantic | hybrid RRF 2:1 | 0.74 (0.62-0.86) | 0.88 (0.80-0.96) | 1.00 | 0.928 | 17 ms |
| semantic | hybrid + rerank | n/a | 0.80 (0.68-0.90) | 0.98 | 0.877 | 806 ms |

(The original run used `--skip-rerank`, hence n/a.) Conclusions, from the audited paired comparisons:
- **Dense retrieval is the strongest single component.** Audited hit@1 is 0.86-0.94 and hit@5 is 0.98-1.00.
- **BM25 alone is clearly worse**: 0.18-0.26 lower hit@1 than dense in every chunking, CI excluding 0 in all three.
  Glossary expansion does not rescue it.
- **Hybrid fusion did not significantly beat dense-only.** All hybrid-vs-dense differences for plain and weighted RRF
  have CIs including 0 (from -0.06 to +0.02 hit@1). Weighting dense 2:1 recovers most of what equal-weight RRF loses,
  but the weighted-vs-equal differences are small (at most +0.04; the sliding gain's interval just touches 0).
  Hybrid is the default for keyword-style queries (error codes, identifiers) that this question set barely tests,
  not because this table shows it winning.
- **Glossary expansion: no significant effect** in the isolated on/off comparisons. Combined with plain equal-weight
  hybrid on sliding chunks it was worse than dense (-0.14, CI excludes 0).
- **Cross-encoder reranking did not help and cost 35-50x the latency** (about 0.6-0.8 s vs 18 ms). Relative to dense
  it was never better and on sliding chunks significantly worse (-0.12 hit@1). So it ships off by default.
- **Chunking is suggestive, not conclusive.** Almost every sliding/semantic-vs-fixed interval includes 0; two
  configurations (of 20) are starred, about what chance gives across this many comparisons. Sliding and semantic have a
  lower median seek error than fixed (dense-only: 14 s and 12 s vs 23 s), but those intervals overlap.

Caveats you should state when quoting these numbers:
- Audited numbers are higher for every system, so quote them next to the original ones, never instead of them.
- The audit judgments were made by an LLM, not a human, on candidates pooled from the same systems being evaluated.
  Spot-check a sample (about 20 "y" and 20 "n") before treating them as ground truth.
- 50 questions from one author: each question is 2 points of hit@1, so differences under about 0.10 are mostly noise.

## Refusing off-topic queries

Search and ask responses carry `out_of_scope` (bool) and `relevance` (best dense cosine). A query is flagged when its
best cosine is below `MIN_RELEVANCE` **and** it contains no rare in-corpus term (a token in at most ~1% of chunks, so
`LNK2005` or `std::move` still work). There is no extra LLM call and no added latency. The closest hits are still
returned so the UI can offer "show closest matches anyway". `/api/ask` skips the LLM entirely when flagged, and also
honours a `NOT_COVERED` reply from the model as a backstop. BM25-only searches aren't gated (no dense score).

Why not an LLM call per query? It adds a network round-trip to every search, needs an API key even for plain search,
and is itself a guess. The retrieval score is free and already measures "is anything close to this?".

The shipped default (0.55) is a starting guess: bge-small cosines are compressed and depend on your index.
`python -m scripts.calibrate --questions eval/questions.json` compares your real questions with ~30 built-in
off-topic queries and prints a threshold that keeps false refusals within budget (2% by default).
Known limit: an off-topic word that happens to occur in a few transcripts (say, an analogy) bypasses the gate.

## Multiple playlists

Each playlist is isolated: `data/playlists/<youtube list id>/{meta.json, transcripts/, index/, chroma/}`. The original
CLI playlist stays in `data/` and appears as `default`. Adding one from the UI runs a background job: list videos,
fetch English captions, chunk + embed + index (default strategy only, no eval). One job at a time, capped by
`MAX_PLAYLIST_VIDEOS`. The engine caches one retriever per (playlist, strategy); finishing or deleting a playlist
drops its cache. The answer prompt receives the playlist title as its topic.

Limits: YouTube blocks caption requests from most cloud IPs, so this works best run locally. The glossary is
C++-specific, and `MIN_RELEVANCE` is a single global value.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | which indexes are built, whether an LLM key is configured |
| `GET /api/videos?playlist=` | indexed videos in playlist order |
| `GET /api/search?q=` | ranked moments. Params: `k`, `strategy`, `mode` (hybrid/dense/bm25), `rerank`, `expand`, `video_id` (repeatable), `group_by=video`, `playlist` |
| `POST /api/ask` | `{question, k?, strategy?, mode?, rerank?, expand?, video_ids?, playlist?}` -> answer + citations |
| `GET/POST /api/playlists`, `GET/DELETE /api/playlists/{id}` | list; `POST {url}` starts indexing and returns 202; poll the id for `status` (`queued/listing/fetching/indexing/ready/failed`) and `done/total` |

Each hit includes `video_id, title, start, end, seek_seconds, timestamp, url, text, score` plus per-stage ranks
(`dense_rank`, `bm25_rank`, `rerank_score`). `seek_seconds` is the chunk start minus a 2 s lead-in (captions lag
speech), which is what the player seeks to. `group_by=video` answers "which episodes cover X?" with one entry per
episode. Search and ask also return `out_of_scope` and `relevance`. Responses carry `timings_ms` per stage.

## Building the eval set

You need ~50 questions with known answer timestamps.

```bash
python -m scripts.peek --videos                       # list video ids
python -m scripts.peek "null pointer"                 # every occurrence, timestamped
python -m scripts.peek --video VIDEO_ID --at 12:30    # read the transcript around a moment
cp eval/questions.example.json eval/questions.json    # then fill in your questions
```
Write questions the way a *student* would ask them ("why does my linker say unresolved external symbol"), not copied
from the transcript, or BM25 wins trivially. Include some that use different words than the lecturer did.

```bash
python -m scripts.eval                       # every built strategy x 10 retrieval configurations
python -m scripts.eval --skip-rerank         # much faster: drops the slow cross-encoder rows
python -m scripts.eval --tol 15              # stricter timestamp tolerance (sensitivity check)
python -m scripts.eval --misses-strategy sliding --misses-config "dense only"
```
Outputs in `eval/results/`:

| file | what it is |
|---|---|
| `results.md` / `results.json` | hit@1/3/5, MRR, median seek error, chunk length and latency per (chunking x configuration), with **95% bootstrap CIs** on hit@1 and MRR |
| `results.md` (lower sections) | **paired comparisons**: every configuration vs a baseline (default "dense only"), and every chunking strategy vs `fixed`, as question-by-question differences with CIs. `*` means the CI excludes 0 |
| `results.md` (effects) | **isolated effects**: glossary on/off and weighted vs equal-weight RRF, each against its own control |
| `misses.md` / `misses.json` | every hit@1 miss for one configuration, classified and shown next to what was retrieved and what the index holds at the labelled moment |
| `per_question.json`, `paired.json` | raw per-question ranks and paired stats |

**Reading the numbers honestly**
- With ~50 questions each question moves hit@1 by 2 points, so most gaps between configurations are within noise.
  Trust a difference only if the paired CI excludes 0, and expect about 1 in 20 starred results to be chance (there are
  many comparisons). More questions is the real fix; ~100+ tightens the intervals a lot.
- A chunk counts as a hit if it's in the right video and its span +/- tolerance contains the labelled moment. Bigger
  chunks contain more moments, so hit@k flatters them. Compare against `avg chunk (s)` and `median seek err (s)`
  (how far from the true moment a click lands). The best strategy is accurate *and* lands close.

**Working through `misses.md`** (cheapest fixes first):
1. `label: video not indexed` / `label: timestamp past end of video` are mistakes in `questions.json`. Fix and re-run first.
2. `hit below rank 1`: the right moment was retrieved but ranked lower. Look at what outranked it.
3. `right video, wrong moment`: topic found, wrong spot; usually chunking or a lecturer revisiting a topic.
4. `wrong video`: vocabulary mismatch (the lecturer never says your word) or a concept covered in several videos
   (add the other occurrence to that question's `answers`).

## Auditing the labels (pooled relevance judgments)

One labelled timestamp per question under-counts: lecturers answer the same thing in several places, so valid
retrievals show up as "misses". The audit finds them without favouring any system (TREC-style pooling):

```bash
python -m scripts.audit build     # pool the top 3 of EVERY configuration, minus what already counts as correct
python -m scripts.audit judge     # y/n per excerpt; shuffled, config and rank hidden; resumable (q to stop)
python -m scripts.audit apply     # writes eval/questions.audited.json
python -m scripts.eval --questions eval/questions.audited.json --out eval/results_audited
```
Rules that keep it defensible: judge **blind** (don't peek at `found_by` in `candidates.json`), pool **all**
configurations rather than only the best one, judge every candidate (unjudged ones count as not relevant), and
**report both** the original and audited numbers plus how many alternatives were added. Judge "would this excerpt
on its own be a good answer?", not "is it the best one".

## Code layout

```
backend/
  app/
    ingest/youtube.py      playlist listing, captions, whisper fallback
    chunking.py            fixed / sliding / semantic
    index/                 Chroma + BM25 + jsonl store, index builder
    retrieval/             tokenizer, glossary, fusion, reranker, embedder, hybrid retriever
    library.py             playlists: add/list/delete, background ingest + index job
    scope.py               off-topic query list and threshold calibration
    llm.py                 Groq/Gemini (OpenAI-compatible) client, prompt, citation parsing
    engine.py              owns models + per-playlist retrievers
    main.py, schemas.py    FastAPI app
    evaluation.py          timestamp-aware metrics, bootstrap CIs, miss classification
    eval_runner.py         ablation grid, paired comparisons, reports (used by scripts/eval.py)
    audit.py               pooled relevance judgments
  scripts/                 ingest, build_index, query, peek, eval, audit, calibrate
  eval/                    questions.example.json -> questions.json, results/
  tests/                   96 offline tests (fake embedder/reranker; no model downloads)
frontend/src/              App.tsx, lib/api.ts, components/ (hit-card, player, answer-panel, playlist-switcher, ...)
```

## Ideas for later

- RAGAS-style faithfulness checks for `/api/ask`, and a Whisper-vs-captions ablation.
- Multi-video "do the episodes agree?" queries on top of `group_by=video`.
- Per-playlist relevance thresholds, calibrated automatically at ingest time.
- Rate limiting and auth before exposing the API publicly (see the hosting notes in the project history).
