import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

import pytest

from backend.stt.refine_transcript import (
    _chunk_text,
    _cosine_similarity,
    _parse_llm_output,
    _rank_chunks,
    build_refinement_prompt,
    retrieve_relevant_context,
)

SEGMENTS = [
    {"id": 0, "speaker": "A", "start": 0.0, "end": 2.0, "text": "추가경영예산안을 논의합니다"},
    {"id": 1, "speaker": "B", "start": 2.0, "end": 4.0, "text": "네 알겠습니다"},
]


def test_build_refinement_prompt_includes_transcript_and_material():
    prompt = build_refinement_prompt(SEGMENTS, material_text="이번 회의 안건: 추가경정예산안")

    assert "추가경영예산안" in prompt
    assert "추가경정예산안" in prompt
    assert '"id": 0' in prompt


def test_build_refinement_prompt_includes_past_meetings():
    prompt = build_refinement_prompt(SEGMENTS, past_meeting_texts=["1차 회의: 결제 모듈 논의", "2차 회의: 일정 조정"])

    assert "결제 모듈" in prompt
    assert "일정 조정" in prompt


def test_build_refinement_prompt_notes_absence_of_materials():
    prompt = build_refinement_prompt(SEGMENTS)

    assert "사전 자료 없음" in prompt


def test_parse_llm_output_maps_id_to_corrected_text():
    raw = json.dumps({"segments": [{"id": 0, "text": "추가경정예산안을 논의합니다"}, {"id": 1, "text": "네 알겠습니다"}]})

    corrections = _parse_llm_output(raw, expected_ids={0, 1})

    assert corrections[0] == "추가경정예산안을 논의합니다"
    assert corrections[1] == "네 알겠습니다"


def test_parse_llm_output_rejects_id_mismatch():
    raw = json.dumps({"segments": [{"id": 0, "text": "x"}]})

    with pytest.raises(ValueError, match="mismatch"):
        _parse_llm_output(raw, expected_ids={0, 1})


def test_parse_llm_output_rejects_missing_segments_key():
    with pytest.raises(ValueError, match="segments"):
        _parse_llm_output(json.dumps({"foo": "bar"}), expected_ids={0})


def test_parse_llm_output_rejects_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_output("not json", expected_ids={0})


def test_chunk_text_splits_by_paragraph():
    text = "첫 번째 문단입니다.\n\n두 번째 문단입니다."

    assert _chunk_text(text) == ["첫 번째 문단입니다.", "두 번째 문단입니다."]


def test_chunk_text_splits_long_paragraph_into_fixed_windows():
    text = "가" * 700

    chunks = _chunk_text(text, chunk_size=300)

    assert len(chunks) == 3
    assert chunks[0] == "가" * 300


def test_chunk_text_empty_returns_empty_list():
    assert _chunk_text(None) == []
    assert _chunk_text("") == []
    assert _chunk_text("   ") == []


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_rank_chunks_returns_most_similar_first():
    candidates = [("material", "A"), ("material", "B"), ("material", "C")]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    query = [1.0, 0.0]

    ranked = _rank_chunks(candidates, embeddings, query, top_k=2)

    assert ranked == [("material", "A"), ("material", "C")]


class _FakeEmbeddingsAPI:
    def __init__(self, embedding_map):
        self._embedding_map = embedding_map

    def create(self, model, input):
        data = [type("EmbeddingObj", (), {"embedding": self._embedding_map[text]})() for text in input]
        return type("Response", (), {"data": data})()


class _FakeClient:
    def __init__(self, embedding_map):
        self.embeddings = _FakeEmbeddingsAPI(embedding_map)


def test_retrieve_relevant_context_keeps_only_top_k_similar_chunks():
    material_text = "결제 모듈 관련 내용입니다.\n\n날씨가 좋은 하루였습니다."
    query_text = "결제 모듈 오류를 논의합니다"
    embedding_map = {
        "결제 모듈 관련 내용입니다.": [1.0, 0.0],
        "날씨가 좋은 하루였습니다.": [0.0, 1.0],
        query_text: [1.0, 0.0],
    }

    material, past = retrieve_relevant_context(
        query_text, material_text=material_text, top_k=1, client=_FakeClient(embedding_map),
    )

    assert material == "결제 모듈 관련 내용입니다."
    assert past is None


def test_retrieve_relevant_context_returns_input_unchanged_when_nothing_to_chunk():
    material, past = retrieve_relevant_context("query", material_text=None, past_meeting_texts=None)

    assert material is None
    assert past is None
