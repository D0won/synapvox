"""HybridSearch — Vector top-k → 그래프 이웃 확장 → 재정렬 (graphrag Baseline).

벡터로 관련 청크를 찾고, Neo4j에서 그 청크의 주제·회의·산출물을 확장해 맥락을 붙인 뒤,
'여러 회의에 걸친 공유 주제'(세션 간 연결)를 신호로 재정렬한다.
"""

import re

from . import schema as S


def select_focus_topics(query_and_answer: str, hits: list[dict]) -> list[str]:
    """Choose a small concept set for graph focus without expanding every hit topic."""
    compact_context = re.sub(r"\s+", "", query_and_answer).casefold()
    topic_scores: dict[str, float] = {}
    direct_matches: set[str] = set()
    for rank, hit in enumerate(hits):
        relevance = max(float(hit.get("score") or 0), 0.0) + 1.0 / (rank + 2)
        for topic in hit.get("topics") or []:
            label = str(topic).strip()
            if not label:
                continue
            compact_label = re.sub(r"\s+", "", label).casefold()
            if len(compact_label) >= 2 and compact_label in compact_context:
                direct_matches.add(label)
            topic_scores[label] = topic_scores.get(label, 0.0) + relevance

    if direct_matches:
        return sorted(direct_matches, key=lambda topic: topic_scores.get(topic, 0.0), reverse=True)
    if not topic_scores:
        return []
    strongest_score = max(topic_scores.values())
    threshold = strongest_score * 0.65
    return [
        topic for topic in sorted(topic_scores, key=topic_scores.get, reverse=True)
        if topic_scores[topic] >= threshold
    ]


class HybridSearch:
    def __init__(self, driver, vector_store, database: str | None = None):
        self.driver = driver
        self.vec = vector_store
        self.database = database

    def _expand_many(self, project_id: str, chunk_ids: list[str]) -> dict[str, dict]:
        """Batch all chunk → topic/meeting expansion into one Neo4j round trip."""
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        with self.driver.session(database=self.database) as s:
            rows = s.run(
                "UNWIND $chunk_ids AS requested_id "
                f"OPTIONAL MATCH (c:{S.CHUNK} {{project_id:$pid, chunk_id:requested_id}}) "
                f"OPTIONAL MATCH (c)-[:{S.DISCUSSES}]->(t:{S.TOPIC}) "
                f"OPTIONAL MATCH (m:{S.MEETING})-[:{S.HAS_CHUNK}]->(c) "
                "RETURN requested_id AS chunk_id, "
                "collect(DISTINCT t{.topic_id, .name}) AS topic_nodes, "
                "head(collect(DISTINCT m{.meeting_id, .title})) AS meeting",
                pid=project_id, chunk_ids=unique_ids,
            ).data()
        return {
            row["chunk_id"]: {
                "topic_nodes": [
                    {"id": topic["topic_id"], "label": topic["name"]}
                    for topic in (row.get("topic_nodes") or [])
                    if topic and topic.get("topic_id") and topic.get("name")
                ],
                "meeting_id": (row.get("meeting") or {}).get("meeting_id"),
                "meeting_title": (row.get("meeting") or {}).get("title"),
            }
            for row in rows
        }

    def search(self, project_id: str, query: str, k: int = 8) -> list[dict]:
        hits = self.vec.query(project_id, query, k=k)
        expansions = self._expand_many(project_id, [hit["chunk_id"] for hit in hits])
        enriched = []
        for h in hits:
            expansion = expansions.get(h["chunk_id"], {})
            topic_nodes = expansion.get("topic_nodes") or []
            enriched.append({
                **h,
                "topics": [topic["label"] for topic in topic_nodes],
                "topic_nodes": topic_nodes,
                "meeting_id": expansion.get("meeting_id") or h.get("meeting_id"),
                "meeting_title": expansion.get("meeting_title"),
            })
        # 재정렬: 공유 주제는 신호로 쓰되 반복 수만큼 점수가 폭증하지 않게 상한을 둔다.
        topic_freq: dict[str, int] = {}
        for e in enriched:
            for t in e["topics"]:
                topic_freq[t] = topic_freq.get(t, 0) + 1
        for e in enriched:
            shared_topics = sum(1 for topic in e["topics"] if topic_freq[topic] > 1)
            boost = min(0.08, 0.02 * shared_topics)
            e["rerank_score"] = round(e["score"] + boost, 4)
        enriched.sort(key=lambda e: e["rerank_score"], reverse=True)
        return enriched


def expansion_from_hits(hits: list[dict], concept_labels: list[str] | None = None) -> dict:
    """Build the answer focus subgraph from HybridSearch's batched enrichment."""
    allowed = set(concept_labels) if concept_labels is not None else None
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    for hit in hits:
        session_id = hit.get("meeting_id")
        if not session_id:
            continue
        session_title = hit.get("meeting_title") or session_id
        for topic in hit.get("topic_nodes") or []:
            concept_id = topic.get("id")
            concept_label = topic.get("label")
            if not concept_id or not concept_label or (allowed is not None and concept_label not in allowed):
                continue
            nodes[session_id] = {
                "id": session_id, "type": "session", "label": session_title, "meta": {},
            }
            nodes[concept_id] = {
                "id": concept_id, "type": "concept", "label": concept_label, "meta": {},
            }
            edges[(session_id, concept_id)] = {
                "src": session_id, "dst": concept_id, "rel_type": "SESSION_MENTIONS_CONCEPT",
            }
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}
