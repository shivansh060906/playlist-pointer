import dataclasses

import pytest

from app.config import Settings
from app.evaluation import bootstrap_median_ci

from .test_eval_tools import engine  # noqa: F401


def test_rrf_weights_env(monkeypatch):
    monkeypatch.delenv("RRF_WEIGHTS", raising=False)
    assert Settings.from_env().rrf_weights == (2.0, 1.0)
    monkeypatch.setenv("RRF_WEIGHTS", "3, 1")
    assert Settings.from_env().rrf_weights == (3.0, 1.0)


@pytest.mark.parametrize("bad", ["1", "1,2,3", "a,b", "-1,2", "0,0"])
def test_rrf_weights_validation(monkeypatch, bad):
    monkeypatch.setenv("RRF_WEIGHTS", bad)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_engine_uses_settings_weights_and_defaults_off(engine):
    seen = {}
    retr = engine.retriever("semantic")
    orig = retr.search
    def spy(*a, **kw):
        seen.update(kw)
        return orig(*a, **kw)
    retr.search = spy
    engine.search("null pointer", 3, strategy="semantic")
    assert seen["rrf_weights"] == engine.settings.rrf_weights
    assert seen["rerank"] is False and seen["expand"] is False
    engine.search("null pointer", 3, strategy="semantic", rrf_weights=(1, 1))
    assert seen["rrf_weights"] == (1, 1)


def test_bootstrap_median_ci():
    lo, hi = bootstrap_median_ci([5, 6, 7, 8, 9], n_boot=300)
    assert 5 <= lo <= 7 <= hi <= 9
    assert bootstrap_median_ci([5, 6, 7, 8, 9], n_boot=300) == (lo, hi)
    assert all(x != x for x in bootstrap_median_ci([]))
