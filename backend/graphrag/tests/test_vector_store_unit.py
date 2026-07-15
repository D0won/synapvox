from types import SimpleNamespace

from backend.graphrag import vector_store as vector_store_module


class _FakeCursor:
    def __init__(self, executed=None):
        self.executed = executed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        if self.executed is not None:
            self.executed.append((sql, params))


class _FakeConnection:
    def __init__(self):
        self.commits = 0
        self.executed = []

    def cursor(self):
        return _FakeCursor(self.executed)

    def commit(self):
        self.commits += 1


def test_add_chunks_batches_default_openai_embeddings(monkeypatch):
    calls = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[
                SimpleNamespace(index=index, embedding=[float(index)])
                for index, _ in enumerate(kwargs["input"])
            ])

    inserted = []
    statements = []
    monkeypatch.setattr(
        vector_store_module,
        "_openai_client",
        SimpleNamespace(embeddings=_Embeddings()),
    )
    monkeypatch.setattr(
        vector_store_module,
        "execute_values",
        lambda _cursor, sql, rows: (statements.append(sql), inserted.extend(rows)),
    )

    store = vector_store_module.VectorStore.__new__(vector_store_module.VectorStore)
    store.embed_fn = vector_store_module.openai_embed
    store.table = "chunks"
    store.conn = _FakeConnection()
    store.add_chunks("project", "meeting", [
        {"chunk_id": "one", "text": "첫 번째"},
        {"chunk_id": "two", "text": "두 번째"},
        {"chunk_id": "three", "text": "세 번째"},
    ])

    assert len(calls) == 1
    assert calls[0]["input"] == ["첫 번째", "두 번째", "세 번째"]
    assert "ON CONFLICT (project_id, chunk_id)" in statements[0]
    assert [row[-1] for row in inserted] == [[0.0], [1.0], [2.0]]
    assert store.conn.commits == 1


def test_reset_many_uses_one_delete_query():
    store = vector_store_module.VectorStore.__new__(vector_store_module.VectorStore)
    store.table = "chunks"
    store.conn = _FakeConnection()

    store.reset_many(["project-a", "project-b"])

    assert len(store.conn.executed) == 1
    assert "project_id = ANY(%s)" in store.conn.executed[0][0]
    assert store.conn.executed[0][1] == (["project-a", "project-b"],)
    assert store.conn.commits == 1
