from collections import Counter

import pytest

from app.chunking import chunk_transcript, clean_text, fixed_chunks, semantic_chunks, sliding_chunks
from app.models import Segment

from .fakes import HashingEmbedder, make_segments, make_transcript


def _transcript(plan=(("pointers", 480), ("templates", 480), ("linker", 480)), seed=1):
    segments, labels = make_segments(plan, seed=seed)
    return make_transcript("vid1", "Test Video", segments), labels


def test_clean_text_strips_caption_noise_but_keeps_code_brackets():
    assert clean_text("[Music] hello   >> world ♪") == "hello world"
    assert clean_text("array[0] is first") == "array[0] is first"


def test_fixed_preserves_all_text_and_orders_times():
    t, _ = _transcript()
    chunks = fixed_chunks(t, target_words=150)
    assert " ".join(c.text for c in chunks).split() == " ".join(s.text for s in t.segments).split()
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert all(a.end <= b.start + 1e-6 for a, b in zip(chunks, chunks[1:]))
    assert all(c.word_count >= 150 or c is chunks[-1] for c in chunks)


def test_sliding_chunks_overlap_and_cover_everything():
    t, _ = _transcript()
    chunks = sliding_chunks(t, window_words=160, stride_words=80)
    assert len(chunks) > len(fixed_chunks(t, target_words=160))
    assert any(a.end > b.start for a, b in zip(chunks, chunks[1:]))  # genuine overlap
    assert chunks[0].start == t.segments[0].start
    assert chunks[-1].end == t.segments[-1].end
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_semantic_splits_near_topic_changes():
    t, labels = _transcript()
    embed = HashingEmbedder().embed_documents
    chunks = semantic_chunks(t, embed, block_words=32, min_words=100, max_words=320, percentile=30)

    assert len(chunks) >= 3
    assert all(c.word_count <= 320 + 64 for c in chunks)  # max + one block of slack

    # Each chunk should be dominated by a single topic (that is the whole point of semantic chunking).
    seg_topic = {s.start: lab for s, lab in zip(t.segments, labels)}
    purities = []
    for c in chunks:
        topics = Counter(lab for start, lab in seg_topic.items() if c.start <= start < c.end)
        purities.append(max(topics.values()) / sum(topics.values()))
    assert sum(purities) / len(purities) >= 0.9

    # No text lost, times monotonic.
    assert " ".join(c.text for c in chunks).split() == " ".join(s.text for s in t.segments).split()
    assert [c.start for c in chunks] == sorted(c.start for c in chunks)


def test_semantic_on_short_video_returns_single_chunk():
    segs = [Segment("one two three four five six seven eight", i * 4.0, 4.0) for i in range(3)]
    chunks = semantic_chunks(make_transcript("v", "Short", segs), HashingEmbedder().embed_documents)
    assert len(chunks) == 1 and chunks[0].start == 0.0


def test_empty_transcript_gives_no_chunks():
    t = make_transcript("v", "Empty", [Segment("[Music]", 0, 5)])
    assert fixed_chunks(t) == [] and sliding_chunks(t) == []
    assert semantic_chunks(t, HashingEmbedder().embed_documents) == []


def test_dispatcher_validates_inputs():
    t, _ = _transcript()
    with pytest.raises(ValueError):
        chunk_transcript(t, "nope")
    with pytest.raises(ValueError):
        chunk_transcript(t, "semantic")  # missing embed_fn
    assert chunk_transcript(t, "fixed")
