import json

import pytest

from app.audit import (
    apply_judgments, file_sha, load_judgments, pool_candidates, run_interactive,
    save_candidates, load_candidates, status,
)
from app.eval_runner import CONFIGS
from app.evaluation import Answer, Question, load_questions, span_hits

from .test_eval_tools import engine  # noqa: F401  (fixture)

QS = [
    Question("stack", "what is the stack and heap", (Answer("ptrvid", 10.0),)),  # label is wrong on purpose
    Question("ptr", "what happens if I dereference a null pointer", (Answer("ptrvid", 150.0),)),
]
CFGS = [c for c in CONFIGS if not c[1].get("rerank")]


def _pool(engine, depth=3):
    return pool_candidates({"semantic": engine.retriever("semantic")}, QS, CFGS, depth=depth, tol=30.0)


def test_pool_excludes_already_relevant_and_dedupes(engine):
    cands = _pool(engine)
    assert cands
    retr = engine.retriever("semantic")
    for c in cands:
        # nothing already counted as correct under the original labels
        assert not span_hits(c["video_id"], c["start"], c["end"], QS[c["q_index"]].answers, 30.0)
        assert c["found_by"]
    keys = [c["key"] for c in cands]
    assert len(keys) == len(set(keys))
    # merged: several configs found the same passage
    assert any(len(c["found_by"]) > 1 for c in cands)
    assert retr  # sanity


def test_interactive_ynsq_and_resume(engine, tmp_path):
    cands = _pool(engine)[:4]
    path = tmp_path / "j.json"
    answers = iter(["y", "x", "n", "s", "q"])
    out = run_interactive(cands, path, input_fn=lambda _: next(answers), print_fn=lambda *_: None)
    assert sorted(out.values()) == ["n", "y"] and load_judgments(path) == out

    # resume: only unjudged ones (including the skipped) are offered; EOF acts as quit
    seen = []
    def inp(_):
        seen.append(1)
        raise EOFError
    run_interactive(cands, path, input_fn=inp, print_fn=lambda *_: None)
    assert len(seen) == 1 and load_judgments(path) == out


def test_prompt_does_not_leak_source(engine, tmp_path):
    cands = _pool(engine)[:2]
    shown = []
    run_interactive(cands, tmp_path / "j.json", input_fn=lambda _: "q", print_fn=shown.append)
    text = "\n".join(shown)
    assert "semantic/" not in text and "found_by" not in text and "hybrid" not in text


def test_apply_adds_midpoints_that_hit(engine, tmp_path):
    cands = _pool(engine)
    judgments = {c["key"]: "y" for c in cands[:3]}
    judgments.update({c["key"]: "n" for c in cands[3:]})
    out = tmp_path / "audited.json"
    stats = apply_judgments(QS, cands, judgments, out)
    assert stats["moments_added"] == 3 and stats["unjudged"] == 0
    audited = load_questions(out)
    assert [q.id for q in audited] == ["stack", "ptr"]
    for c in cands[:3]:
        assert span_hits(c["video_id"], c["start"], c["end"], audited[c["q_index"]].answers, 0.0)
    raw = json.loads(out.read_text())
    assert any(a.get("added_by") == "audit" for q in raw["questions"] for a in q["answers"])
    # originals preserved
    assert Answer("ptrvid", 10.0) in audited[0].answers


def test_status_and_sha(engine, tmp_path):
    cands = _pool(engine)
    assert status(cands, {})["unjudged"] == len(cands)
    p = tmp_path / "q.json"
    p.write_text("[]")
    a = file_sha(p)
    p.write_text("[1]")
    assert file_sha(p) != a
    save_candidates(tmp_path / "c.json", cands, questions_sha=a, depth=3, tol=30.0)
    assert load_candidates(tmp_path / "c.json")["questions_sha"] == a
