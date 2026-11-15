"""Tests for bootstrap stats, miss classification, weighted RRF and the full eval runner."""
import dataclasses
import json

import numpy as np
import pytest

pytest.importorskip("chromadb")

from app.config import Settings
from app.engine import RAGEngine
from app.eval_runner import CONFIGS, evaluate_and_report, render_misses_md
from app.evaluation import (
    MISS_CATEGORIES,
    Answer,
    Question,
    bootstrap_ci,
    classify_miss,
    excludes_zero,
    paired_bootstrap,
    per_question_arrays,
)
from app.index.builder import build_index
from app.models import Segment, Transcript
from app.retrieval.glossary import Glossary

from .fakes import HashingEmbedder, OverlapReranker


# --------------------------------------------------------------------- statistics
def test_per_question_arrays():
    a = per_question_arrays([1, 2, None, 5, 7])
    assert a["hit@1"].tolist() == [1, 0, 0, 0, 0]
    assert a["hit@3"].tolist() == [1, 1, 0, 0, 0]
    assert a["hit@5"].tolist() == [1, 1, 0, 1, 0]
    assert a["rr"].tolist() == pytest.approx([1, 0.5, 0, 0.2, 1 / 7])


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [1, 0, 1, 1, 0] * 10  # mean 0.6, n=50
    lo, hi = bootstrap_ci(values, seed=3)
    assert lo < 0.6 < hi
    assert 0.1 < hi - lo < 0.35  # roughly +/-13 points at n=50
    assert bootstrap_ci(values, seed=3) == (lo, hi)
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)
    assert all(np.isnan(bootstrap_ci([])))


def test_paired_bootstrap_detects_real_and_ignores_null_differences():
    base = np.array([0, 1] * 25, dtype=float)
    better = np.ones(50)  # strictly >= baseline on every question
    diff, lo, hi = paired_bootstrap(better, base, seed=1)
    assert diff == pytest.approx(0.5) and excludes_zero(lo, hi)

    same_diff, lo, hi = paired_bootstrap(base, base)
    assert (same_diff, lo, hi) == (0.0, 0.0, 0.0) and not excludes_zero(lo, hi)

    rng = np.random.default_rng(0)
    noisy_a = rng.integers(0, 2, 50).astype(float)
    noisy_b = rng.integers(0, 2, 50).astype(float)
    _, lo, hi = paired_bootstrap(noisy_a, noisy_b, seed=1)
    assert lo < 0 < hi  # coin-flip differences are not significant


# -------------------------------------------------------------- miss classification
EXTENT = {"v1": 600.0, "v2": 900.0}


def test_classify_label_problems_come_first():
    cat, detail = classify_miss([("v1", 0, 10)], [Answer("typo", 5)], 30, EXTENT)
    assert cat == "label: video not indexed"
    assert "index" in detail
    cat, detail = classify_miss([("v1", 0, 10)], [Answer("v1", 5000)], 30, EXTENT)
    assert cat == "label: timestamp past end of video"


def test_classify_retrieval_failures():
    answers = [Answer("v1", 300)]
    assert classify_miss([("v2", 0, 60), ("v1", 280, 340)], answers, 30, EXTENT)[0] == "hit below rank 1"
    cat, detail = classify_miss([("v2", 0, 60), ("v1", 0, 60)], answers, 30, EXTENT)
    assert cat == "right video, wrong moment" and "240s" in detail
    assert classify_miss([("v2", 0, 60), ("v2", 100, 160)], answers, 30, EXTENT)[0] == "wrong video"
    assert classify_miss([("v1", 280, 340)], answers, 30, EXTENT)[0] == "hit at rank 1"
    assert set(MISS_CATEGORIES) >= {"wrong video", "right video, wrong moment", "hit below rank 1"}


def test_classify_multi_answer_uses_any_known_video():
    answers = [Answer("gone", 10), Answer("v2", 100)]
    assert classify_miss([("v1", 0, 60)], answers, 30, EXTENT)[0] == "wrong video"


# ------------------------------------------------------------------ pipeline fixtures
def _video(video_id, title, index, sections):
    segs, t = [], 0.0
    for text, n in sections:
        for _ in range(n):
            segs.append(Segment(text=text, start=t, duration=4.0))
            t += 4.0
    return Transcript(video_id, title, index, "test", segs)


@pytest.fixture()
def engine(tmp_path):
    s = dataclasses.replace(Settings.from_env(), data_dir=tmp_path, groq_api_key=None, gemini_api_key=None)
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
    emb = HashingEmbedder()
    for strategy in ("fixed", "semantic"):
        build_index(s, strategy, emb, log=lambda *_: None)
    return RAGEngine(s, embedder=emb, reranker=OverlapReranker(), glossary=Glossary.load())


# --------------------------------------------------------------------- weighted RRF
def test_weighted_rrf_extremes_match_single_retrievers(engine):
    r = engine.retriever("fixed")
    q = "what happens if I dereference a null pointer"
    dense = r.search(q, 10, mode="dense", rerank=False, dedupe=False)
    only_dense = r.search(q, 10, mode="hybrid", rerank=False, dedupe=False, rrf_weights=(1.0, 0.0))
    assert [h.chunk.chunk_id for h in dense.hits] == [h.chunk.chunk_id for h in only_dense.hits][: len(dense.hits)]

    bm25 = r.search(q, 10, mode="bm25", rerank=False, expand=False, dedupe=False)
    only_bm25 = r.search(q, 10, mode="hybrid", rerank=False, expand=False, dedupe=False, rrf_weights=(0.0, 1.0))
    assert [h.chunk.chunk_id for h in bm25.hits] == [h.chunk.chunk_id for h in only_bm25.hits][: len(bm25.hits)]


def test_weighted_rrf_validates_length(engine):
    with pytest.raises(ValueError):
        engine.retriever("fixed").search("pointers", 3, rrf_weights=(1.0, 2.0, 3.0))


def test_default_configs_include_weighted_rrf_and_unique_labels():
    labels = [label for label, _ in CONFIGS]
    assert len(labels) == len(set(labels))
    assert "dense only" in labels and "hybrid RRF 2:1" in labels and "hybrid 2:1 + glossary" in labels
    assert dict(CONFIGS)["hybrid RRF 2:1"]["rrf_weights"] == (2.0, 1.0)


# ---------------------------------------------------------------- end-to-end runner
QUESTIONS = [
    Question("ptr", "what happens if I dereference a null pointer", (Answer("ptrvid", 150.0),)),
    Question("tmpl", "how does template specialization work", (Answer("tmplvid", 200.0),)),
    Question("bad-label", "what is the stack", (Answer("typo-id", 10.0),)),          # label mistake
    Question("late-label", "what are templates", (Answer("tmplvid", 99999.0),)),    # label mistake
    Question("wrong-moment", "what is the stack and heap", (Answer("ptrvid", 10.0),)),  # real answer is at ~240s+
]


def test_evaluate_and_report_end_to_end(engine, tmp_path):
    out = tmp_path / "results"
    logs = []
    outcome = evaluate_and_report(
        {s: engine.retriever(s) for s in ("fixed", "semantic")},
        QUESTIONS,
        out,
        baseline="dense only",
        n_boot=200,
        misses_strategy="semantic",
        misses_config="hybrid + glossary",
        log=logs.append,
    )

    for name in ("results.md", "results.json", "paired.json", "per_question.json", "misses.md", "misses.json"):
        assert (out / name).exists(), name

    summaries = json.loads((out / "results.json").read_text())
    assert len(summaries) == 2 * len(CONFIGS)
    first = summaries[0]
    assert "median_seek_error_ci" in first
    assert first["n"] == 5 and len(first["hit@1_ci"]) == 2 and first["hit@1_ci"][0] <= first["hit@1"] <= first["hit@1_ci"][1]

    md = (out / "results.md").read_text()
    assert "hit@1 (95% CI)" in md and "vs \"dense only\"" in md and "Chunking strategies vs fixed" in md

    paired = json.loads((out / "paired.json").read_text())
    assert paired["vs_baseline"] and paired["vs_fixed"]
    effect_names = {e["effect"] for e in paired["effects"]}
    assert effect_names == {"Glossary expansion on vs off", "Weighted RRF vs equal-weight RRF"}
    assert "Glossary expansion on vs off" in md and "median seek err (s, 95% CI)" in md
    assert all(e["strategy"] == "semantic" for e in paired["vs_fixed"])
    assert all(e["reference"] == "dense only" for e in paired["vs_baseline"])

    per_q = json.loads((out / "per_question.json").read_text())
    assert len(per_q["fixed"]["dense only"]) == len(QUESTIONS)

    # Misses: the two deliberately broken labels must be flagged as label problems, not retrieval failures.
    misses = outcome["misses"]
    by_id = {m["id"]: m for m in misses["misses"]}
    assert by_id["bad-label"]["category"] == "label: video not indexed"
    assert by_id["late-label"]["category"] == "label: timestamp past end of video"
    assert "ptr" not in by_id  # answered correctly at rank 1
    # Sorted so the cheapest fixes (label mistakes) come first, in MISS_CATEGORIES order.
    cats = [m["category"] for m in misses["misses"]]
    assert cats[:2] == ["label: video not indexed", "label: timestamp past end of video"]
    md_misses = (out / "misses.md").read_text()
    assert "bad-label" in md_misses and "Retrieved:" in md_misses
    assert any("Misses at hit@1" in line for line in logs)


def test_misses_report_skipped_when_target_missing(engine, tmp_path):
    out = tmp_path / "r2"
    logs = []
    outcome = evaluate_and_report(
        {"fixed": engine.retriever("fixed")},
        QUESTIONS[:2],
        out,
        configs=[c for c in CONFIGS if not c[1].get("rerank")],
        n_boot=50,
        misses_strategy="semantic",  # not part of this run
        log=logs.append,
    )
    assert outcome["misses"] is None
    assert not (out / "misses.md").exists()
    assert (out / "results.md").exists()
    assert any("skipping misses report" in line for line in logs)


def test_bad_baseline_is_rejected(engine, tmp_path):
    with pytest.raises(ValueError):
        evaluate_and_report({"fixed": engine.retriever("fixed")}, QUESTIONS[:1], tmp_path, baseline="nope", n_boot=10)


def test_render_misses_md_handles_empty():
    md = render_misses_md({"total_questions": 3, "counts": {}, "misses": []}, "semantic", "dense only")
    assert "0 of 3" in md
