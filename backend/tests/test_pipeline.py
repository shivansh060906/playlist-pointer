"""End-to-end offline test: transcripts -> chunk -> Chroma + BM25 -> hybrid search -> eval metrics."""
import dataclasses

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("rank_bm25")

from app.config import Settings
from app.engine import RAGEngine
from app.evaluation import Answer, first_hit_rank
from app.index.builder import build_index
from app.index.store import IndexNotBuilt
from app.retrieval.glossary import Glossary
from app.models import Segment, Transcript

from .fakes import HashingEmbedder, OverlapReranker


def _seg(text, start):
    return Segment(text=text, start=start, duration=4.0)


def _video(video_id, title, index, sections):
    """sections: list of (text_repeated, n_segments)."""
    segs, t = [], 0.0
    for text, n in sections:
        for _ in range(n):
            segs.append(_seg(text, t))
            t += 4.0
    return Transcript(video_id, title, index, "test", segs)


@pytest.fixture()
def settings(tmp_path):
    base = Settings.from_env()
    s = dataclasses.replace(base, data_dir=tmp_path, groq_api_key=None, gemini_api_key=None)
    t1 = _video(
        "ptrvid", "Pointers in C++", 1,
        [
            ("welcome today we talk about pointers and memory addresses in general", 30),
            ("a null pointer holds no address so you must check before you dereference it", 30),
            ("the stack and the heap are two places where variables can live in memory", 30),
        ],
    )
    t2 = _video(
        "tmplvid", "Templates in C++", 2,
        [
            ("templates let the compiler generate code for any type you pass in", 40),
            ("template specialization lets you override behaviour for one particular type", 40),
        ],
    )
    for t in (t1, t2):
        t.save(s.transcripts_dir / f"{t.video_id}.json")
    return s


@pytest.fixture()
def engine(settings):
    emb = HashingEmbedder()
    for strategy in ("fixed", "sliding", "semantic"):
        stats = build_index(settings, strategy, emb, log=lambda *_: None)
        assert stats["chunks"] > 0
    return RAGEngine(settings, embedder=emb, reranker=OverlapReranker(), glossary=Glossary.load())


def test_all_strategies_built_and_listed(engine):
    assert engine.available_strategies() == ["fixed", "sliding", "semantic"]
    vids = engine.videos()
    assert [v["video_id"] for v in vids] == ["ptrvid", "tmplvid"]
    assert vids[0]["chunk_count"] > 0 and vids[0]["duration"] > 0


@pytest.mark.parametrize("strategy", ["fixed", "sliding", "semantic"])
@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_search_finds_the_right_moment(engine, strategy, mode):
    res = engine.search("what happens if I dereference a null pointer", 3, strategy=strategy, mode=mode)
    assert res.hits, "expected hits"
    top = res.hits[0].chunk
    assert top.video_id == "ptrvid"
    # the null-pointer section spans 120s-240s of the video
    spans = [(h.chunk.video_id, h.chunk.start, h.chunk.end) for h in res.hits]
    assert first_hit_rank(spans, [Answer("ptrvid", 150.0)], tol=30) is not None
    assert {"total"} <= set(res.timings_ms)


def test_rerank_flag_and_timings(engine):
    on = engine.search("template specialization", 3, rerank=True)
    off = engine.search("template specialization", 3, rerank=False)
    assert on.rerank_applied and "rerank" in on.timings_ms
    assert not off.rerank_applied and "rerank" not in off.timings_ms
    assert on.hits[0].chunk.video_id == "tmplvid"


def test_glossary_expansion_reported_for_bm25(engine):
    res = engine.search("what is nullptr", 3, mode="bm25", expand=True)
    assert res.expanded_query and "null pointer" in res.expanded_query
    # The transcript says "null pointer", never "nullptr": only the expansion can find it by keyword.
    assert res.hits and res.hits[0].chunk.video_id == "ptrvid"
    plain = engine.search("what is nullptr", 3, mode="bm25", expand=False)
    assert plain.expanded_query is None
    assert not plain.hits


def test_video_filter(engine):
    res = engine.search("type", 5, video_ids=["ptrvid"], mode="hybrid")
    assert res.hits and all(h.chunk.video_id == "ptrvid" for h in res.hits)


def test_bad_inputs(engine):
    with pytest.raises(ValueError):
        engine.search("x y", 3, mode="telepathy")
    with pytest.raises(ValueError):
        engine.retriever("nonexistent")


def test_missing_index_raises_helpful_error(settings, tmp_path):
    empty = dataclasses.replace(settings, data_dir=tmp_path / "elsewhere")
    with pytest.raises(IndexNotBuilt) as exc:
        RAGEngine(empty, embedder=HashingEmbedder()).retriever("semantic")
    assert "build_index" in str(exc.value)


def test_ask_requires_llm_key(engine):
    from app.llm import LLMNotConfigured

    with pytest.raises(LLMNotConfigured):
        engine.ask("what is a pointer")


def test_ask_with_stub_llm_parses_citations(settings):
    from app.llm import cited_indices

    class StubLLM:
        model = "stub-model"

        def chat(self, messages, **_):
            assert "Excerpts" in messages[1]["content"]
            return "A null pointer holds no address [1]. See also [2] and a fake one [99]."

    emb = HashingEmbedder()
    build_index(settings, "fixed", emb, log=lambda *_: None)
    eng = RAGEngine(settings, embedder=emb, reranker=OverlapReranker(), llm=StubLLM(), glossary=Glossary.load())
    result = eng.ask("what is a null pointer", 4, strategy="fixed")
    assert result.answer.startswith("A null pointer")
    assert result.cited == [1, 2][: len(result.hits)] or result.cited == [1]
    assert 99 not in result.cited
    assert "llm" in result.timings_ms
    assert cited_indices("x [3] [3] [1] [0]", 3) == [3, 1]
