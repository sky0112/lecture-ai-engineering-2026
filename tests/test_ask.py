"""会話履歴を保つ質問スクリプトのテスト（Gemini APIは呼び出さない）。"""

import importlib
import sys
from types import SimpleNamespace


def test_main_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    ask_module = importlib.import_module("app.ask")
    sent_messages = []

    class FakeChat:
        def send_message(self, question):
            sent_messages.append(question)
            return SimpleNamespace(
                text=f"回答: {question}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=2,
                    candidates_token_count=3,
                ),
            )

    chat = FakeChat()
    created_models = []
    fake_client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda *, model: (created_models.append(model), chat)[1]
        )
    )
    questions = iter(["続けて質問", "exit"])

    def create_client(*, api_key):
        assert api_key == "test-key"
        return fake_client

    monkeypatch.setattr(ask_module.genai, "Client", create_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask_module.main()

    output = capsys.readouterr().out
    assert sent_messages == ["最初の質問", "続けて質問"]
    assert created_models == [ask_module.MODEL]
    assert "回答: 最初の質問" in output
    assert "回答: 続けて質問" in output
