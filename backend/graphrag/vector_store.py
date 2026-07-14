"""VectorStore — 청크 임베딩 저장/검색 (Chroma).

schemas/graph_vector_db.md: "청크 임베딩 + 메타데이터. 메타데이터 필터: project_id, meeting_id, source_type."
임베딩은 주입식(embed_fn). 기본은 의존성 없는 결정론적 해싱 임베더(오프라인/테스트용) —
실서비스는 OpenAI 등 embed_fn을 주입한다. chunk_id로 Graph DB의 Chunk.vector_ref와 교차 참조.
"""

import hashlib

import chromadb


def hashing_embed(text: str, dim: int = 256) -> list[float]:
    """의존성 없는 결정론적 임베더 (문자 n-gram 해싱). 실서비스는 embed_fn 주입으로 대체."""
    v = [0.0] * dim
    t = " ".join((text or "").split())
    for n in (2, 3):
        for i in range(len(t) - n + 1):
            h = int.from_bytes(hashlib.md5(t[i:i + n].encode()).digest()[:4], "big")
            v[h % dim] += 1.0
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v] if norm else v


class VectorStore:
    def __init__(self, embed_fn=None, path: str | None = None, collection: str = "chunks"):
        self.embed_fn = embed_fn or hashing_embed
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        # 임베딩을 직접 넣으므로 embedding_function 없이 생성 (무거운 기본 임베더 회피)
        self.col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, project_id: str, meeting_id: str, chunks: list[dict]):
        """chunk = {chunk_id, text, source_type}. 임베딩 후 메타데이터와 함께 저장."""
        if not chunks:
            return
        self.col.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=[self.embed_fn(c.get("text", "")) for c in chunks],
            documents=[c.get("text", "") for c in chunks],
            metadatas=[{"project_id": project_id, "meeting_id": meeting_id,
                        "source_type": c.get("source_type") or "unknown"} for c in chunks])

    def query(self, project_id: str, text: str, k: int = 8, source_type: str | None = None):
        where = {"project_id": project_id} if not source_type else \
            {"$and": [{"project_id": project_id}, {"source_type": source_type}]}
        res = self.col.query(query_embeddings=[self.embed_fn(text)], n_results=k, where=where)
        out = []
        for cid, doc, dist, meta in zip(res["ids"][0], res["documents"][0],
                                        res["distances"][0], res["metadatas"][0]):
            out.append({"chunk_id": cid, "text": doc, "score": round(1.0 - dist, 4),
                        "meeting_id": meta.get("meeting_id"), "source_type": meta.get("source_type")})
        return out

    def reset(self, project_id: str):
        self.col.delete(where={"project_id": project_id})
