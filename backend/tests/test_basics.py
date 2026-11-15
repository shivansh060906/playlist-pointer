from app.evaluation import Answer, first_hit_rank, load_questions, seek_error, span_hits, summarize
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.glossary import Glossary
from app.retrieval.hybrid import Hit, dedupe_overlaps, group_hits_by_video
from app.retrieval.tokenizer import tokenize
from app.utils import format_timestamp, parse_timestamp, youtube_url
from app.models import Chunk


# ---- utils
def test_timestamp_roundtrip():
    assert format_timestamp(92) == "1:32"
    assert format_timestamp(3725) == "1:02:05"
    assert format_timestamp(-5) == "0:00"
    assert parse_timestamp("12:34") == 754
    assert parse_timestamp("1:02:05") == 3725
    assert parse_timestamp(754) == 754.0
    assert youtube_url("abc", 61.9) == "https://www.youtube.com/watch?v=abc&t=61s"


# ---- tokenizer
def test_tokenizer_keeps_cpp_terms():
    toks = tokenize("Why does std::vector beat C++ arrays?")
    assert "std::vector" in toks and "std" in toks and "vector" in toks
    assert "cpp" in toks
    assert "array" in toks  # plural stemmed


def test_tokenizer_variants_and_stopwords():
    assert "cpp" in tokenize("C plus plus is fun")
    assert tokenize("the a of") == []
    assert tokenize("pointers") == tokenize("pointer")
    assert tokenize("classes") == tokenize("class")


# ---- glossary
def test_glossary_expands_both_directions():
    g = Glossary.load()
    assert "null pointer" in g.expand("what is nullptr")
    assert "nullptr" in g.expand("what is a null pointer")
    assert "see out" in g.expand("how does cout work")
    assert g.expand("completely unrelated words here") == "completely unrelated words here"


def test_glossary_matches_whole_words_only():
    g = Glossary({"cin": ["c in"]})
    assert g.expand("cinema") == "cinema"


# ---- fusion
def test_rrf_prefers_items_ranked_by_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "b", "e"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "b"  # rank 2 in both lists beats rank 1 in only one
    assert set(ids) == {"a", "b", "c", "d", "e"}
    assert fused == sorted(fused, key=lambda kv: (-kv[1], kv[0]))


def test_rrf_weights():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 3.0])
    assert fused[0][0] == "b"


# ---- dedupe / grouping
def _chunk(cid, vid, start, end):
    return Chunk(cid, vid, f"T-{vid}", 1, start, end, "text", "sliding")


def test_dedupe_drops_overlapping_neighbours_only_within_same_video():
    hits = [
        Hit(_chunk("a", "v1", 0, 100), 3.0),
        Hit(_chunk("b", "v1", 50, 150), 2.0),   # 50% overlap with a -> dropped
        Hit(_chunk("c", "v1", 120, 220), 1.0),  # 0 overlap with a -> kept
        Hit(_chunk("d", "v2", 0, 100), 0.5),    # different video -> kept
    ]
    assert [h.chunk.chunk_id for h in dedupe_overlaps(hits)] == ["a", "c", "d"]


def test_group_by_video_keeps_best_first():
    hits = [Hit(_chunk("a", "v1", 0, 10), 3.0), Hit(_chunk("b", "v2", 0, 10), 2.0), Hit(_chunk("c", "v1", 20, 30), 1.0)]
    groups = group_hits_by_video(hits)
    assert [g.video_id for g in groups] == ["v1", "v2"]
    assert groups[0].hit_count == 2 and groups[0].best_chunk.chunk_id == "a"


# ---- eval metrics
def test_span_hit_uses_tolerance():
    answers = [Answer("v1", 100.0)]
    assert span_hits("v1", 60, 90, answers, tol=30)      # 100 <= 90+30
    assert not span_hits("v1", 60, 90, answers, tol=5)
    assert not span_hits("v2", 60, 140, answers, tol=30)  # wrong video


def test_first_hit_rank_and_seek_error():
    answers = [Answer("v1", 100.0), Answer("v2", 500.0)]
    spans = [("v3", 0, 50), ("v1", 90, 130), ("v2", 480, 520)]
    assert first_hit_rank(spans, answers, tol=0) == 2
    assert first_hit_rank([("v3", 0, 5)], answers, tol=30) is None
    assert seek_error(("v1", 90, 130), answers) == 10.0
    assert seek_error(("v9", 0, 1), answers) is None


def test_summarize():
    rows = [
        {"rank": 1, "seek_error": 4.0, "latency_ms": 10},
        {"rank": 3, "seek_error": 20.0, "latency_ms": 30},
        {"rank": None, "seek_error": None, "latency_ms": 20},
        {"rank": 1, "seek_error": 8.0, "latency_ms": 40},
    ]
    s = summarize(rows)
    assert s["n"] == 4
    assert s["hit@1"] == 0.5 and s["hit@3"] == 0.75 and s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 1 / 3 + 0 + 1) / 4) < 1e-9
    assert s["median_seek_error_s"] == 6.0


def test_load_questions_skips_placeholders(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        '{"questions": [{"question": "q1", "video_id": "REPLACE_ME", "timestamp": 1},'
        ' {"id": "real", "question": "q2", "answers": [{"video_id": "abc", "timestamp": "1:30"}]}]}'
    )
    qs = load_questions(path)
    assert len(qs) == 1 and qs[0].id == "real" and qs[0].answers[0].timestamp == 90.0
