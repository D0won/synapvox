import base64
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

from backend.stt.image_description import _build_vision_messages, describe_image


def test_build_vision_messages_encodes_image_and_includes_prompt():
    messages = _build_vision_messages(b"fake-png-bytes", "image/png", "설명해주세요")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "설명해주세요"}

    expected_b64 = base64.b64encode(b"fake-png-bytes").decode("ascii")
    assert content[1] == {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{expected_b64}"}}


class _FakeChatAPI:
    def __init__(self, reply):
        self._reply = reply

    def create(self, model, messages):
        message = type("Message", (), {"content": self._reply})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, reply):
        self.chat = type("Chat", (), {"completions": _FakeChatAPI(reply)})()


def test_describe_image_returns_stripped_model_reply():
    fake_client = _FakeClient("  차트에는 2026년 예산 항목이 나와 있습니다.  ")

    description = describe_image(b"fake-png-bytes", client=fake_client)

    assert description == "차트에는 2026년 예산 항목이 나와 있습니다."
