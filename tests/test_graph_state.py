from types import SimpleNamespace

from app.core import graph as graph_module


def test_generate_answer_sets_rag_state_for_api(monkeypatch):
    metadata = SimpleNamespace(
        model_dump=lambda: {
            "route": "rag",
            "retrieved_chunks": [{"text": "chunk-1", "source": "doc-1", "score": 0.9}],
            "cache_hit": True,
            "reflection_iterations": 1,
            "refined_question": "refined",
        },
        retrieved_chunks=[SimpleNamespace(model_dump=lambda: {"text": "chunk-1", "source": "doc-1", "score": 0.9})],
        reflection_iterations=1,
        refined_question="refined",
    )
    response = SimpleNamespace(
        answer="The answer",
        sources=["doc-1"],
        confidence=0.88,
        cache_hit=True,
        metadata=metadata,
    )

    monkeypatch.setattr(graph_module, "run_rag", lambda *args, **kwargs: response)

    state = {"question": "what is this?", "intent": "rag", "flags": {}}
    result = graph_module.generate_answer(state)

    assert result["final_answer"] == "The answer"
    assert result["sources"] == ["doc-1"]
    assert result["cache_hit"] is True
    assert result["retrieved_chunks"] == ["doc-1"]
    assert result["metadata"]["cache_hit"] is True
