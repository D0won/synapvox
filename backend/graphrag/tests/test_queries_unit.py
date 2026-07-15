from backend.graphrag.queries import concept_detail


class _Record:
    def data(self):
        return {
            "concept_id": "topic-kkt",
            "label": "KKT 조건",
            "summary": "제약 최적화의 필요조건",
            "sessions": [{"session_id": "lecture-05", "title": "최적화 개론"}],
            "related_concepts": [
                {"concept_id": "topic-lagrange", "label": "라그랑지안"},
                {"concept_id": "topic-duality", "label": "쌍대성"},
            ],
        }


class _Result:
    def single(self):
        return _Record()


class _Session:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def run(self, query, **params):
        self.calls.append((query, params))
        return _Result()


class _Driver:
    def __init__(self):
        self.calls = []

    def session(self, database=None):
        return _Session(self.calls)


def test_concept_detail_includes_bidirectional_related_concepts():
    driver = _Driver()

    result = concept_detail(driver, "project-1", "topic-kkt", "neo4j")

    assert [concept["label"] for concept in result["related_concepts"]] == [
        "라그랑지안",
        "쌍대성",
    ]
    query, params = driver.calls[0]
    assert "-[:RELATES_TO]-" in query
    assert params == {"pid": "project-1", "tid": "topic-kkt"}
