"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, request_tool_approvals
from app import tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz", "save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_waits_for_approval():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("学習記録を保存して")

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_name for call in result.output.approvals] == ["save_learning_log"]


@pytest.mark.parametrize("answer, should_save", [("y", True), ("n", False)])
def test_save_learning_log_respects_approval(answer, should_save, tmp_path, monkeypatch):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    monkeypatch.setattr("builtins.input", lambda _: answer)

    pending = agent.run_sync("学習記録を保存して", model=TestModel())
    assert isinstance(pending.output, DeferredToolRequests)
    assert not log_dir.exists()

    deferred_results = request_tool_approvals(pending.output)
    result = agent.run_sync(
        message_history=pending.all_messages(),
        deferred_tool_results=deferred_results,
        model=TestModel(call_tools=[]),
    )

    assert isinstance(result.output, str)
    saved_files = list(log_dir.glob("*.md")) if log_dir.exists() else []
    assert bool(saved_files) is should_save
    if should_save:
        assert saved_files[0].read_text(encoding="utf-8").strip()
