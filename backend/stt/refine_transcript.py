import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

_MAX_RETRIES = 2


def _chunk_text(text: str, chunk_size: int = 300) -> list:
    """Paragraph-based chunking (split on blank lines), falling back to a fixed-size
    char window for paragraphs longer than chunk_size. No sentence-boundary awareness —
    good enough for retrieval scoring, not meant for display."""
    if not text or not text.strip():
        return []
    chunks = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            for i in range(0, len(paragraph), chunk_size):
                chunks.append(paragraph[i:i + chunk_size])
    return chunks


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _rank_chunks(candidates: list, chunk_embeddings: list, query_embedding: list, top_k: int) -> list:
    """candidates: list of (source_label, chunk_text), parallel to chunk_embeddings.
    Returns the top_k candidates ranked by cosine similarity to query_embedding, most
    similar first."""
    scored = sorted(
        zip(candidates, chunk_embeddings),
        key=lambda pair: _cosine_similarity(query_embedding, pair[1]),
        reverse=True,
    )
    return [candidate for candidate, _ in scored[:top_k]]


def retrieve_relevant_context(
    query_text: str,
    material_text: str = None,
    past_meeting_texts: list = None,
    top_k: int = 5,
    embedding_model: str = "text-embedding-3-small",
    client=None,
):
    """TEMPORARY self-contained stand-in to verify the retrieval concept end-to-end —
    not the intended long-term integration. Chunks material_text/past_meeting_texts,
    embeds them + query_text (the raw 1차 전사 text) via OpenAI, keeps only the top_k
    chunks most similar to the transcript.

    The real integration point is 용하's `backend.graphrag.VectorStore`
    (`add_chunks(project_id, meeting_id, chunks)` to store, `query(project_id, text, k,
    source_type)` to retrieve) once he wires in a real embed_fn (currently a hashing
    stub, see progress.md 2026-07-14). When that's ready, replace this function's body
    with calls to that store instead of chunking/embedding locally — keep the same
    (material_text, past_meeting_texts) return shape so refine_transcript()/
    build_refinement_prompt() don't need to change either way."""
    candidates = [("material", c) for c in _chunk_text(material_text)]
    for i, text in enumerate(past_meeting_texts or []):
        candidates += [(f"past_meeting_{i}", c) for c in _chunk_text(text)]

    if not candidates:
        return material_text, past_meeting_texts

    client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    texts_to_embed = [c[1] for c in candidates] + [query_text]
    response = client.embeddings.create(model=embedding_model, input=texts_to_embed)
    embeddings = [d.embedding for d in response.data]
    chunk_embeddings, query_embedding = embeddings[:-1], embeddings[-1]

    top = _rank_chunks(candidates, chunk_embeddings, query_embedding, top_k)

    retrieved_material = "\n\n".join(text for label, text in top if label == "material") or None
    retrieved_past = [text for label, text in top if label.startswith("past_meeting")] or None

    return retrieved_material, retrieved_past


def build_refinement_prompt(segments: list, material_text: str = None, past_meeting_texts: list = None) -> str:
    """Stage-2 STT refinement prompt: RAG over (1차 전사 + 회의자료 + 과거 회의록).
    Assumes material_text/past_meeting_texts have already been narrowed to the relevant
    chunks (see retrieve_relevant_context()) — this function itself just dumps whatever
    it's given, it doesn't retrieve."""
    transcript_json = json.dumps(
        [{"id": s["id"], "speaker": s["speaker"], "text": s["text"]} for s in segments],
        ensure_ascii=False,
    )

    context_parts = []
    if material_text:
        context_parts.append(f"회의 자료:\n{material_text}")
    if past_meeting_texts:
        context_parts.append("과거 회의록:\n" + "\n---\n".join(past_meeting_texts))
    context_block = "\n\n".join(context_parts) if context_parts else "(사전 자료 없음)"

    return f"""다음은 회의 전사문 세그먼트 목록과 참고 자료입니다. 전문용어·고유명사 오인식을 자료를 참고해 교정하고,
말이 안 되는 부분은 맥락에 맞게 자연스럽게 다듬어 주세요. 화자나 세그먼트 순서·개수는 바꾸지 마세요.

# 참고 자료
{context_block}

# 전사문 세그먼트 (JSON)
{transcript_json}

# 출력 형식
각 세그먼트의 id와 교정된 text만 담은 JSON으로 출력하세요. 다른 설명은 넣지 마세요.
{{"segments": [{{"id": 0, "text": "교정된 텍스트"}}, ...]}}
"""


def _parse_llm_output(raw_content: str, expected_ids: set) -> dict:
    data = json.loads(raw_content)
    if "segments" not in data:
        raise ValueError("missing 'segments' key in LLM output")

    got_ids = {s["id"] for s in data["segments"]}
    if got_ids != expected_ids:
        raise ValueError(f"segment id mismatch: expected {expected_ids}, got {got_ids}")

    return {s["id"]: s["text"] for s in data["segments"]}


def refine_transcript(
    data: dict,
    material_text: str = None,
    past_meeting_texts: list = None,
    model: str = "gpt-4o",
) -> dict:
    """Apply stage-2 refinement to a 중간 포맷 JSON object (source/mode/segments shape).
    Returns the same shape with segments[].text replaced by the corrected version."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = build_refinement_prompt(data["segments"], material_text, past_meeting_texts)
    expected_ids = {s["id"] for s in data["segments"]}

    messages = [{"role": "user", "content": prompt}]
    corrections = None
    last_error = None

    for _ in range(_MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        try:
            corrections = _parse_llm_output(content, expected_ids)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"오류: {last_error}. 형식을 정확히 지켜 다시 출력하세요."})

    if corrections is None:
        raise RuntimeError(f"LLM refinement failed after {_MAX_RETRIES + 1} attempts: {last_error}")

    refined_segments = [{**seg, "text": corrections[seg["id"]]} for seg in data["segments"]]
    return {**data, "segments": refined_segments}


def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 STT refinement: RAG-correct a 중간 포맷 JSON transcript using materials + past meetings"
    )
    parser.add_argument("intermediate_json", help="Output of backend.stt.stt_normalizer (stage-1 raw transcript)")
    parser.add_argument("--material", help="Path to extracted pre-meeting materials text")
    parser.add_argument(
        "--past-meeting",
        action="append",
        default=[],
        help="Path to a past meeting's transcript text (same project). Repeatable.",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to keep (default: 5)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Skip embedding-based retrieval, dump all material/past-meeting text directly (old behavior)",
    )
    parser.add_argument("-o", "--output", help="Write JSON to this path instead of stdout")
    args = parser.parse_args()

    data = json.loads(Path(args.intermediate_json).read_text(encoding="utf-8"))
    material_text = Path(args.material).read_text(encoding="utf-8") if args.material else None
    past_meeting_texts = [Path(p).read_text(encoding="utf-8") for p in args.past_meeting] or None

    if not args.no_retrieval and (material_text or past_meeting_texts):
        query_text = " ".join(s["text"] for s in data["segments"])
        material_text, past_meeting_texts = retrieve_relevant_context(
            query_text, material_text, past_meeting_texts, args.top_k, args.embedding_model
        )

    result = refine_transcript(data, material_text, past_meeting_texts, args.model)
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
