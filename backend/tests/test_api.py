from fastapi.testclient import TestClient

from app.index.store import IndexNotBuilt
from app.llm import LLMNotConfigured
from app.main import app, get_engine
from app.models import Chunk
from app.retrieval.hybrid import Hit, SearchResult


def _hit(cid, vid, start, end, score, text="some text"):
    return Hit(Chunk(cid, vid, f"Title {vid}", 3, start, end, text, "semantic"), score, dense_rank=1, bm25_rank=2)


class FakeSettings:
    default_strategy = "semantic"
    lead_in_seconds = 2
    llm_provider = "groq"


class StubEngine:
    settings = FakeSettings()

    def __init__(self):
        self.last = {}

    def available_strategies(self):
        return ["semantic"]

    def llm_configured(self):
        return True

    def videos(self, playlist=None):
        return [{"video_id": "v1", "title": "Pointers", "playlist_index": 1, "duration": 600.0, "chunk_count": 4}]

    def search(self, query, k, **kw):
        self.last = {"query": query, "k": k, **kw}
        if kw.get("strategy") == "bogus":
            raise ValueError("strategy must be one of ...")
        hits = [_hit("c1", "v1", 100.4, 160, 2.5), _hit("c2", "v2", 5, 40, 1.5), _hit("c3", "v1", 300, 360, 1.0)]
        off = "pony" in query
        return SearchResult(
            hits[:k], {"total": 12.0, "dense": 5.0}, kw.get("mode", "hybrid"), True, "q expanded",
            0.31 if off else 0.82, off,
        )

    def ask(self, question, k, **kw):
        from app.engine import AskResult

        if question == "no index":
            raise IndexNotBuilt("semantic")
        if question == "no key":
            raise LLMNotConfigured("No API key for provider 'groq'")
        if question == "boom":
            raise RuntimeError("upstream exploded")
        self.last = {"question": question, "k": k, **kw}
        hits = [_hit("c1", "v1", 100.4, 160, 2.5)]
        if "pony" in question:
            return AskResult("Not related.", hits, [], {"total": 3.0}, "stub", False, True, 0.31)
        return AskResult("A pointer holds an address [1].", hits, [1], {"total": 9.0, "llm": 4.0}, "stub", True)


client = TestClient(app)
stub = StubEngine()
app.dependency_overrides[get_engine] = lambda: stub


def test_health_and_videos():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["strategies_built"] == ["semantic"]
    assert client.get("/api/videos").json()[0]["video_id"] == "v1"


def test_search_shape_and_seek_math():
    r = client.get("/api/search", params={"q": "what is a pointer", "k": 2, "mode": "bm25", "rerank": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["rerank_applied"] is True and body["expanded_query"] == "q expanded"
    first = body["hits"][0]
    assert first["seek_seconds"] == 98  # int(100.4) - 2s lead-in
    assert first["timestamp"] == "1:38"
    assert first["url"] == "https://www.youtube.com/watch?v=v1&t=98s"
    assert stub.last["mode"] == "bm25" and stub.last["rerank"] is False and stub.last["k"] == 2
    assert body["videos"] is None


def test_search_group_by_video_fetches_deeper_pool():
    r = client.get("/api/search", params={"q": "pointers", "k": 2, "group_by": "video"})
    body = r.json()
    assert stub.last["k"] == 20
    assert [v["video_id"] for v in body["videos"]] == ["v1", "v2"]
    assert body["videos"][0]["hit_count"] == 2


def test_search_validation_errors():
    assert client.get("/api/search", params={"q": "x"}).status_code == 422          # too short
    assert client.get("/api/search", params={"q": "ok ok", "mode": "magic"}).status_code == 422
    assert client.get("/api/search", params={"q": "ok ok", "strategy": "bogus"}).status_code == 400


def test_ask_ok_with_citations():
    r = client.post("/api/ask", json={"question": "what is a pointer"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].endswith("[1].") and body["model"] == "stub"
    assert body["citations"][0]["index"] == 1
    assert body["citations"][0]["hit"]["url"].endswith("t=98s")


def test_ask_error_mapping():
    assert client.post("/api/ask", json={"question": "no index"}).status_code == 503
    r = client.post("/api/ask", json={"question": "no key"})
    assert r.status_code == 503 and "API key" in r.json()["detail"]
    assert client.post("/api/ask", json={"question": "boom"}).status_code == 502
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422  # too short


def test_cors_headers_for_vite_dev_server():
    r = client.options(
        "/api/search",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_default_search_and_ask_leave_rerank_and_expand_off():
    client.get("/api/search", params={"q": "what is a pointer"})
    assert stub.last["rerank"] is False and stub.last["expand"] is False
    client.post("/api/ask", json={"question": "what is a pointer"})
    assert stub.last["rerank"] is False and stub.last["expand"] is False


def test_search_reports_out_of_scope_and_relevance():
    ok = client.get("/api/search", params={"q": "what is a pointer"}).json()
    assert ok["out_of_scope"] is False and ok["relevance"] == 0.82
    off = client.get("/api/search", params={"q": "my little pony"}).json()
    assert off["out_of_scope"] is True and off["relevance"] == 0.31 and off["hits"]  # closest hits kept


def test_ask_reports_out_of_scope_without_citations():
    body = client.post("/api/ask", json={"question": "tell me about my little pony"}).json()
    assert body["out_of_scope"] is True and body["citations"] == [] and body["hits"]
    ok = client.post("/api/ask", json={"question": "what is a pointer"}).json()
    assert ok["out_of_scope"] is False
