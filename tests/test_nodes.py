from src.graph.state import State
from src.graph.nodes import score_confidence

def test_score_confidence_sufficient_evidence():
    state = State(
        query="test",
        retrieved_chunks=[{"chunk_id": "c1", "text": "x", "rerank_score": 0.9, "source": "s"}],
    )
    result = score_confidence(state)
    assert result["evidence_sufficient"] is True

def test_score_confidence_empty_chunks():
    state = State(query="test", retrieved_chunks=[])
    result = score_confidence(state)
    assert result["confidence_score"] == 0.0
    assert result["evidence_sufficient"] is False