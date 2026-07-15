import json

from fastapi.testclient import TestClient

from backend.integration.api import main as api_main


app = api_main.app
app.dependency_overrides[api_main.require_user] = lambda: {"sub": "test-user"}


def test_graph_ingest_routes_use_api_prefix():
    documented_post_routes = {
        route.path
        for route in app.routes
        if "POST" in getattr(route, "methods", set()) and getattr(route, "include_in_schema", False)
    }

    assert "/api/ingest-doc" in documented_post_routes
    assert "/api/ingest-stt" in documented_post_routes


def test_legacy_graph_ingest_routes_remain_as_hidden_aliases():
    hidden_post_routes = {
        route.path
        for route in app.routes
        if "POST" in getattr(route, "methods", set()) and not getattr(route, "include_in_schema", True)
    }

    assert "/ingest-doc" in hidden_post_routes
    assert "/ingest-stt" in hidden_post_routes


def test_api_ingest_stt_uses_active_project(monkeypatch):
    import backend.graphrag as graphrag
    from backend.integration import pipeline
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (None, object(), None))
    monkeypatch.setattr(api_main, "_optional_vector_store", lambda: None)
    monkeypatch.setattr(api_main, "_owned_transcript_exists", lambda user, project, meeting: True)
    monkeypatch.setattr(graphrag, "graph_data", lambda driver, project, database: {
        "nodes": [{"id": "t1", "type": "concept", "label": "그래프", "meta": {}}], "edges": [],
    })
    monkeypatch.setattr(
        pipeline, "ingest_intermediate",
        lambda transcript, store, vector, project: {
            "project_id": project, "meeting_id": transcript["meeting_id"],
            "chunks": [{"chunk_id": "c1"}], "topic_ids": ["t1"], "relations": 0,
        },
    )
    transcript = {
        "source": "lecture.wav",
        "meeting_id": "lecture-01",
        "project_id": "display-name",
        "date": "2026-07-15",
        "mode": "lecture",
        "segments": [
            {"id": 0, "speaker": "A", "start": 0.0, "end": 1.0, "text": "그래프 이론"},
        ],
    }

    response = TestClient(app).post(
        "/api/ingest-stt",
        headers={"X-Project-Id": "project-uuid", "X-API-Key": "test-key"},
        json=transcript,
    )

    assert response.status_code == 200
    assert response.json()["project"] == "project-uuid"


def test_api_ingest_stt_rejects_unsaved_transcript(monkeypatch):
    monkeypatch.setattr(api_main, "_owned_transcript_exists", lambda user, project, meeting: False)
    transcript = {
        "source": "lecture.wav",
        "meeting_id": "lecture-cancelled",
        "project_id": "project-uuid",
        "date": "2026-07-15",
        "mode": "lecture",
        "segments": [
            {"id": 0, "speaker": "A", "start": 0.0, "end": 1.0, "text": "취소된 전사"},
        ],
    }

    response = TestClient(app).post(
        "/api/ingest-stt",
        headers={"X-Project-Id": "project-uuid"},
        json=transcript,
    )

    assert response.status_code == 409


def test_api_ingest_doc_stores_text_file(monkeypatch):
    import backend.graphrag as graphrag
    from backend.integration import pipeline
    captured = {}
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (None, object(), None))
    monkeypatch.setattr(api_main, "_optional_vector_store", lambda: None)
    monkeypatch.setattr(graphrag, "graph_data", lambda driver, project, database: {
        "nodes": [{"id": "t1", "type": "concept", "label": "graph", "meta": {}}], "edges": [],
    })
    def fake_ingest(text, title, project, store, vector, meeting, document_id=None):
        captured["document_id"] = document_id
        return {
            "project_id": project, "meeting_id": meeting, "chunks": [{"chunk_id": "c1"}],
            "topic_ids": ["t1"], "relations": 0,
        }

    monkeypatch.setattr(pipeline, "ingest_document_text", fake_ingest)

    response = TestClient(app).post(
        "/api/ingest-doc",
        headers={
            "X-Project-Id": "project-uuid",
            "X-Meeting-Id": "lecture-01",
            "X-Source-Id": "material-123",
            "X-API-Key": "test-key",
        },
        files={"file": ("notes.txt", b"graph theory notes", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "project": "project-uuid",
        "meeting": "lecture-01",
        "title": "notes",
        "chunks_ingested": 1,
        "concepts_new": 1,
        "concepts_total": 1,
        "relations_new": 0,
    }
    assert captured["document_id"] == "material-123"


def test_api_graph_and_ask_use_current_project(monkeypatch):
    import backend.graphrag as graphrag

    class _FakeSearch:
        def __init__(self, driver, vector, database):
            pass

        def search(self, project, question, k):
            return [{"chunk_id": "c1", "text": "근거", "meeting_id": "m1", "topics": []}]

    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), object(), "neo4j"))
    monkeypatch.setattr(api_main, "_optional_vector_store", lambda: object())
    monkeypatch.setattr(api_main, "_answer_from_hits", lambda question, hits: question)
    monkeypatch.setattr(
        graphrag, "graph_data",
        lambda driver, project, database: {
            "nodes": [{"id": project, "type": "session", "label": "강의", "meta": {}}], "edges": [],
        },
    )
    monkeypatch.setattr(graphrag, "HybridSearch", _FakeSearch)
    monkeypatch.setattr(
        graphrag, "expansion_from_hits",
        lambda hits, concept_labels: {"nodes": [], "edges": []},
    )
    client = TestClient(app)

    graph = client.get("/api/graph", params={"project": "project-uuid"})
    answer = client.get("/api/ask", params={"project": "project-uuid", "q": "질문", "k": 4})

    assert graph.status_code == 200
    assert graph.json()["nodes"][0]["id"] == "project-uuid"
    assert answer.status_code == 200
    assert answer.json()["answer"] == "질문"


def test_api_ask_stream_emits_deltas_then_focus_graph(monkeypatch):
    import backend.graphrag as graphrag

    class _FakeSearch:
        def __init__(self, driver, vector, database):
            pass

        def search(self, project, question, k):
            return [{
                "chunk_id": "c1",
                "text": "미분 근거",
                "score": 0.8,
                "meeting_id": "m1",
                "meeting_title": "최적화 개론",
                "topics": ["미분"],
                "topic_nodes": [{"id": "t1", "label": "미분"}],
            }]

    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), object(), "neo4j"))
    monkeypatch.setattr(api_main, "_optional_vector_store", lambda: object())
    monkeypatch.setattr(api_main, "_stream_answer_text", lambda question, hits: iter(["미분은 ", "변화율입니다."]))
    monkeypatch.setattr(graphrag, "HybridSearch", _FakeSearch)
    monkeypatch.setattr(
        graphrag,
        "expansion_from_hits",
        lambda hits, labels: {
            "nodes": [{"id": "t1", "type": "concept", "label": "미분", "meta": {}}],
            "edges": [],
        },
    )

    response = TestClient(app).get(
        "/api/ask-stream",
        params={"project": "project-uuid", "q": "미분이 뭐야", "k": 4},
    )
    events = [json.loads(line) for line in response.text.splitlines() if line]

    assert response.status_code == 200
    assert "".join(event.get("text", "") for event in events) == "미분은 변화율입니다."
    assert events[-1]["type"] == "complete"
    assert events[-1]["answer"] == "미분은 변화율입니다."
    assert events[-1]["expansion"]["nodes"][0]["id"] == "t1"


def test_delete_recording_source_removes_graph_meeting_and_vectors(monkeypatch):
    calls = []

    class _GraphStore:
        def delete_meeting(self, project, meeting):
            calls.append(("graph", project, meeting))
            return 3

    class _VectorStore:
        def delete_meeting(self, project, meeting):
            calls.append(("vector", project, meeting))
            return 3

    monkeypatch.setattr(api_main, "_owned_source_record", lambda user, source: {
        "id": source,
        "project_id": "project-uuid",
        "recording_id": "recording-123",
        "kind": "audio",
        "original_name": "lecture.webm",
        "source_payload": {"graphMeetingId": "meeting-123"},
    })
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), _GraphStore(), "neo4j"))
    monkeypatch.setattr(api_main, "_vector_store", lambda: _VectorStore())

    response = TestClient(app).delete("/api/source-graph", params={"source_id": "recording-123"})

    assert response.status_code == 200
    assert response.json()["graph_chunks_deleted"] == 3
    assert set(calls) == {
        ("graph", "project-uuid", "meeting-123"),
        ("vector", "project-uuid", "meeting-123"),
    }


def test_delete_document_source_removes_only_its_chunk_prefix(monkeypatch):
    calls = []

    class _GraphStore:
        def delete_chunks_by_prefix(self, project, prefix):
            calls.append(("graph", project, prefix))
            return 2

    class _VectorStore:
        def delete_chunks_by_prefix(self, project, prefix):
            calls.append(("vector", project, prefix))
            return 2

    monkeypatch.setattr(api_main, "_owned_source_record", lambda user, source: {
        "id": source,
        "project_id": "project-uuid",
        "recording_id": "recording-123",
        "kind": "document",
        "original_name": "lecture-notes.pdf",
        "source_payload": {},
    })
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), _GraphStore(), "neo4j"))
    monkeypatch.setattr(api_main, "_vector_store", lambda: _VectorStore())

    response = TestClient(app).delete("/api/source-graph", params={"source_id": "material-123"})

    assert response.status_code == 200
    prefix = response.json()["chunk_prefix"]
    assert prefix.startswith("doc-") and prefix.endswith("-d")
    assert set(calls) == {
        ("graph", "project-uuid", prefix),
        ("vector", "project-uuid", prefix),
    }


def test_delete_project_data_resets_owned_trashed_project(monkeypatch):
    calls = []

    class _GraphStore:
        def reset_many(self, projects):
            calls.append(("graph", tuple(projects)))

    class _VectorStore:
        def reset_many(self, projects):
            calls.append(("vector", tuple(projects)))

    monkeypatch.setattr(api_main, "_owned_project_record", lambda user, project: {
        "id": project,
        "name": "삭제할 프로젝트",
        "trashed_at": "2026-07-15T00:00:00+00:00",
    })
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), _GraphStore(), "neo4j"))
    monkeypatch.setattr(api_main, "_vector_store", lambda: _VectorStore())

    response = TestClient(app).delete("/api/project-data", params={"project_id": "project-uuid"})

    assert response.status_code == 200
    assert response.json() == {"project_id": "project-uuid", "deleted": True}
    assert set(calls) == {
        ("graph", ("project-uuid",)),
        ("vector", ("project-uuid",)),
    }


def test_bulk_delete_project_data_uses_one_batch_per_store(monkeypatch):
    calls = []

    class _GraphStore:
        def reset_many(self, projects):
            calls.append(("graph", tuple(projects)))

    class _VectorStore:
        def reset_many(self, projects):
            calls.append(("vector", tuple(projects)))

    monkeypatch.setattr(
        api_main,
        "_owned_trashed_project_ids",
        lambda user, projects: list(projects),
    )
    monkeypatch.setattr(api_main, "_graph_runtime", lambda: (object(), _GraphStore(), "neo4j"))
    monkeypatch.setattr(api_main, "_vector_store", lambda: _VectorStore())

    response = TestClient(app).post(
        "/api/projects-data/delete",
        json={"project_ids": ["project-a", "project-b"]},
    )

    assert response.status_code == 200
    assert response.json() == {
        "project_ids": ["project-a", "project-b"],
        "deleted": 2,
    }
    assert set(calls) == {
        ("graph", ("project-a", "project-b")),
        ("vector", ("project-a", "project-b")),
    }


def test_delete_project_data_requires_trash_first(monkeypatch):
    monkeypatch.setattr(api_main, "_owned_project_record", lambda user, project: {
        "id": project,
        "name": "활성 프로젝트",
        "trashed_at": None,
    })

    response = TestClient(app).delete("/api/project-data", params={"project_id": "project-uuid"})

    assert response.status_code == 409
