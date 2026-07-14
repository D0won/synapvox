# graphrag — ③④ Vector/Graph DB · 검색

담당: 용하

## Baseline
- Graph DB 스키마 구현·적재 로직 (Neo4j)
- Vector DB 구축 (pgvector 또는 Chroma)
- 하이브리드 검색: Vector top-k → 그래프 이웃 확장 → 재정렬
- 시간별·맥락별 조회 쿼리 (Cypher)

## 고도화
- 검색 재정렬 튜닝
- Entity Store SSOT · 교차 링킹
- 크로스 프로젝트 검색
- 관리형 DB로 이전

## 소유 스키마
Graph/Vector DB 스키마 — 노드/엣지 정의. → [`schemas/graph_vector_db.md`](../../schemas/graph_vector_db.md)

---

## 구현 (v0.1 — feat/graphrag-store-and-search)

`schemas/graph_vector_db.md` draft를 그대로 구현. **plain Neo4j + Chroma**. integration(D)이 함수로 소비.

| 파일 | 역할 |
| --- | --- |
| `schema.py` | Neo4j 노드/엣지 상수 + 제약(모든 노드 `(project_id, key)` 유니크) |
| `graph_store.py` | 적재: `load_intermediate`(→Project·Meeting) · `load_chunks`(→Chunk) · `load_extraction`(→Topic·Decision·ActionItem + DISCUSSES/DECIDED_IN/RAISED_IN/SUPERSEDES). FOLLOWS는 date 순 자동 |
| `vector_store.py` | Chroma 청크 임베딩. `embed_fn` 주입식(기본=의존성 없는 해싱, 실서비스는 OpenAI 등). 메타 필터 `project_id/meeting_id/source_type` |
| `search.py` | `HybridSearch.search` — 벡터 top-k → 그래프 이웃(주제·회의) 확장 → 공유주제 가중 재정렬 |
| `queries.py` | `timeline`(FOLLOWS) · `meetings_by_topic`(DISCUSSES) · `decision_history`(SUPERSEDES) |

사용 예는 `__init__.py` docstring. 테스트: `pytest backend/graphrag/tests/`(Neo4j 없으면 자동 skip).

**입력 계약 메모(도윤·B와 확인 필요)**: `load_chunks`/`add_chunks`가 받는 청크는
`{chunk_id, source_type, raw_span, timestamps, text}` 형태로 가정했습니다. chunking 산출 청크 포맷을
`schemas/`에 명시하면 맞추겠습니다. `schemas/graph_vector_db.md`는 손대지 않았습니다(draft 그대로 구현).

**레퍼런스**: 팀 스키마와 별개로, 동일 아이디어(세션 간 개념 연결 + GraphRAG)를 end-to-end로 돌려본
단독 프로토타입 — https://github.com/click6067-ship-it/synapVOX (커스텀판 + Graphiti판). 하이브리드 검색·Neo4j
적재·시각화·RAG 답변을 실제로 확인해볼 수 있습니다. 가져다 참고용.
