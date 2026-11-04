"""Step through the RAG pipeline one feature at a time.

Flip the FLAGS below phase by phase:
  Phase 2  basic/dense : search_mode="dense",  everything else False
  Phase 3  sparse/hybrid: search_mode="sparse" / "hybrid"  (optional: enable_hyde=True)
  Phase 4  reranking   : enable_rerank=True
  Phase 5  CRAG        : enable_crag=True
  Phase 6  Self-RAG    : enable_self_reflective=True
"""

from app.services.rag_service import _generate, _retrieve

Q = "How does a Kubernetes Deployment handle rolling updates?"

FLAGS = dict(
    search_mode="dense",  # "dense" | "sparse" | "hybrid"
    top_k=5,
    enable_rerank=False,
    enable_hyde=False,
    enable_crag=False,
    enable_self_reflective=False,
)

chunks = _retrieve(Q, FLAGS)
print("\n=== Retrieved chunks ===")
for c in chunks:
    print(f"{c.score:.3f}  {c.source}  {c.text[:80]!r}")

resp = _generate(Q, chunks, FLAGS)
print("\n=== Answer ===")
print(resp.answer)
print("\n=== Metadata ===")
print(
    f"reflection_iterations={resp.metadata.reflection_iterations} "
    f"reflection_score={resp.metadata.reflection_score} "
    f"refined_question={resp.metadata.refined_question}"
)
