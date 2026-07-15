from backend.graphrag.search import HybridSearch, expansion_from_hits, select_focus_topics


def test_select_focus_topics_prefers_concept_named_in_question():
    hits = [
        {"score": 0.31, "topics": ["미분", "함수", "계산", "람다"]},
        {"score": 0.29, "topics": ["미분", "제약", "조건"]},
        {"score": 0.28, "topics": ["거리", "마진", "하이퍼플레인"]},
    ]

    assert select_focus_topics("미분이 뭐야", hits) == ["미분"]


def test_select_focus_topics_keeps_every_concept_mentioned_in_answer():
    hits = [
        {"score": 0.3, "topics": ["미분", "함수", "기울기", "제약", "람다", "최댓값"]},
        {"score": 0.29, "topics": ["미분", "경계", "조건", "거리"]},
    ]

    selected = select_focus_topics(
        "미분이 뭐야\n미분은 함수의 기울기를 구하며 제약 조건이 있는 최댓값 문제에도 사용됩니다.",
        hits,
    )

    assert set(selected) == {"미분", "함수", "기울기", "제약", "조건", "최댓값"}
    assert "람다" not in selected
    assert "거리" not in selected


def test_hybrid_search_batches_graph_expansion_and_preserves_hit_order():
    class _Vector:
        def query(self, project, query, k):
            return [
                {"chunk_id": "c-high", "text": "미분", "score": 0.9, "meeting_id": "m1"},
                {"chunk_id": "c-low", "text": "함수", "score": 0.6, "meeting_id": "m1"},
            ]

    class _Rows:
        def __init__(self, calls):
            self.calls = calls

        def data(self):
            return [
                {
                    "chunk_id": "c-high",
                    "topic_nodes": [{"topic_id": "t1", "name": "미분"}],
                    "meeting": {"meeting_id": "m1", "title": "최적화 개론"},
                },
                {
                    "chunk_id": "c-low",
                    "topic_nodes": [{"topic_id": "t2", "name": "함수"}],
                    "meeting": {"meeting_id": "m1", "title": "최적화 개론"},
                },
            ]

    class _Session:
        def __init__(self, calls):
            self.calls = calls

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def run(self, query, **params):
            self.calls.append((query, params))
            return _Rows(self.calls)

    class _Driver:
        def __init__(self):
            self.calls = []

        def session(self, database=None):
            return _Session(self.calls)

    driver = _Driver()
    hits = HybridSearch(driver, _Vector(), "neo4j").search("project", "미분", 2)

    assert len(driver.calls) == 1
    assert driver.calls[0][1]["chunk_ids"] == ["c-high", "c-low"]
    assert [hit["chunk_id"] for hit in hits] == ["c-high", "c-low"]
    assert hits[0]["topic_nodes"] == [{"id": "t1", "label": "미분"}]


def test_expansion_from_hits_reuses_batched_topic_nodes_without_duplicates():
    hits = [
        {
            "meeting_id": "m1",
            "meeting_title": "최적화 개론",
            "topic_nodes": [{"id": "t1", "label": "미분"}, {"id": "t2", "label": "함수"}],
        },
        {
            "meeting_id": "m1",
            "meeting_title": "최적화 개론",
            "topic_nodes": [{"id": "t1", "label": "미분"}],
        },
    ]

    expansion = expansion_from_hits(hits, ["미분"])

    assert {node["id"] for node in expansion["nodes"]} == {"m1", "t1"}
    assert expansion["edges"] == [
        {"src": "m1", "dst": "t1", "rel_type": "SESSION_MENTIONS_CONCEPT"},
    ]
