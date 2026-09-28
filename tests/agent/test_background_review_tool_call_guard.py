"""The background-review fork must not spawn when it could only no-op.

The review fork's whole job is to emit ``memory`` / ``skill_manage`` tool calls,
and by default it inherits the parent's live runtime. When the parent provider IS
an autonomous agent reached through a client shim that cannot carry Hermes tool
calls back, that fork is a guaranteed no-op — one that still pays for a full
agent spawn (a whole CLI process, sometimes a JVM) on every review cadence.

So: a client declaring ``SUPPORTS_HERMES_TOOL_CALLS = False`` skips the fork with
a log line pointing at the ``auxiliary.background_review`` override. A client
that can emit tool calls is unaffected, as are ordinary providers whose clients
say nothing at all.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent.background_review as bg  # noqa: E402


def _fake_parent(client, *, runtime=None) -> SimpleNamespace:
    """The minimal parent-agent surface _run_review_in_thread touches pre-fork."""
    return SimpleNamespace(
        provider="acp-agent",
        model="acp-agent",
        client=client,
        session_id="s1",
        platform="cli",
        request_overrides={},
        max_tokens=None,
        acp_command="acp-agent",
        acp_args=["--acp"],
        enabled_toolsets=None,
        disabled_toolsets=None,
        reasoning_config=None,
        _credential_pool=None,
        _current_main_runtime=lambda: runtime or {
            "api_key": "k",
            "base_url": "acp://agent",
            "api_mode": "chat_completions",
        },
        _emit_auxiliary_failure=lambda *_a, **_k: None,
        _safe_print=lambda *_a, **_k: None,
        background_review_callback=None,
        _memory_store=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        _cached_system_prompt="",
        session_start=None,
        memory_notifications="on",
    )


def _run(agent, task_cfg=None):
    """Run the worker with AIAgent patched; return the AIAgent mock."""
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.AIAgent") as mock_aiagent,
        patch("tools.terminal_tool.set_approval_callback"),
    ):
        bg._run_review_in_thread(
            agent, [{"role": "user", "content": "hi"}], "review please", task_cfg
        )
    return mock_aiagent


def test_fork_is_skipped_when_the_provider_cannot_emit_tool_calls(caplog):
    client = MagicMock()
    client.SUPPORTS_HERMES_TOOL_CALLS = False
    with caplog.at_level(logging.WARNING, logger=bg.logger.name):
        mock_aiagent = _run(_fake_parent(client))
    mock_aiagent.assert_not_called()
    # The user needs to know which knob makes the review work again.
    assert "auxiliary.background_review" in caplog.text


def test_fork_is_spawned_when_the_provider_can_emit_tool_calls():
    client = MagicMock()
    client.SUPPORTS_HERMES_TOOL_CALLS = True
    assert _run(_fake_parent(client)).called


def test_ordinary_providers_are_unaffected():
    # A plain OpenAI-style client says nothing about the capability.
    class _PlainClient:
        pass

    assert _run(_fake_parent(_PlainClient())).called


def test_the_capability_is_read_off_the_class_too():
    """Clients declare it as a class attribute; an instance need not set it."""

    class _IncapableClient:
        SUPPORTS_HERMES_TOOL_CALLS = False

    assert bg._parent_can_emit_tool_calls(_fake_parent(_IncapableClient())) is False
    assert bg._parent_can_emit_tool_calls(_fake_parent(None)) is True


def test_an_incapable_provider_still_reviews_when_the_review_is_routed_away():
    """``auxiliary.background_review.{provider,model}`` sends the fork to a normal
    model, so the parent's shim no longer matters."""

    class _IncapableClient:
        SUPPORTS_HERMES_TOOL_CALLS = False

    routed = {
        "provider": "openai",
        "model": "gpt-5",
        "api_key": "k",
        "base_url": None,
        "api_mode": "chat_completions",
        "credential_pool": None,
        "request_overrides": {},
        "max_tokens": None,
        "command": None,
        "args": [],
        "routed": True,
    }
    with patch.object(bg, "_resolve_review_runtime", return_value=routed):
        assert _run(_fake_parent(_IncapableClient())).called


def test_busy_fence_defers_review_before_fork_writes():
    from agent import post_write_validation as gate

    parent = _fake_parent(MagicMock())
    with (
        gate.acquire_operation_fence(),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.AIAgent") as mock_aiagent,
        patch("tools.terminal_tool.set_approval_callback"),
    ):
        bg._run_review_in_thread(parent, [], "review")
        mock_aiagent.return_value.run_conversation.assert_not_called()
        assert not gate.gate_state_path().exists()


def test_background_review_gate_controls_completion_message():
    from agent import post_write_validation as gate

    for state in (gate.GateState.PASS, gate.GateState.FINDINGS,
                  gate.GateState.OPERATIONAL_ERROR, gate.GateState.INTERRUPTED):
        parent = _fake_parent(MagicMock())
        messages = []
        parent.background_review_callback = messages.append
        parent._safe_print = messages.append
        result = gate.GateResult("check", state, "operation", reason="test-result")
        with (
            patch("hermes_cli.config.load_config", return_value={}),
            patch("run_agent.AIAgent") as mock_aiagent,
            patch("tools.terminal_tool.set_approval_callback"),
            patch.object(bg, "summarize_background_review_actions", return_value=["Skill updated"]),
            patch.object(gate, "changed_paths_since", return_value={"skills/sample/SKILL.md"}),
            patch.object(gate, "run_applicable_validators", return_value=[result]),
        ):
            mock_aiagent.return_value.session_id = "fork-session"
            def review_write(**_kwargs):
                assert gate.gate_state_path().exists()
                assert "PENDING" in gate.gate_state_path().read_text()
                with pytest.raises(gate.FenceBusy):
                    with gate.acquire_operation_fence(timeout=0):
                        pass
            mock_aiagent.return_value.run_conversation.side_effect = review_write
            bg._run_review_in_thread(parent, [], "review")
        assert gate.gate_state_path().exists()
        assert state.value in gate.gate_state_path().read_text()
        if state == gate.GateState.PASS:
            assert any("Skill updated" in message for message in messages)
        else:
            assert not any("Self-improvement review: Skill updated" in message for message in messages)
            assert any(f"validation {state.value}" in message for message in messages)


def test_background_terminal_write_failure_keeps_pending_and_suppresses_success(tmp_path, monkeypatch):
    import json
    from agent import post_write_validation as gate

    home = tmp_path / "profile"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    parent = _fake_parent(MagicMock())
    messages = []
    parent._safe_print = messages.append
    parent.background_review_callback = messages.append
    original = gate.record_gate_state
    def fail_terminal(*args, **kwargs):
        if args[2] != gate.GateState.PENDING:
            raise OSError("terminal write failed")
        return original(*args, **kwargs)
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.AIAgent") as mock_aiagent,
        patch("tools.terminal_tool.set_approval_callback"),
        patch.object(bg, "summarize_background_review_actions", return_value=["Skill updated"]),
        patch.object(gate, "record_gate_state", side_effect=fail_terminal),
    ):
        mock_aiagent.return_value.session_id = "fork-session"
        bg._run_review_in_thread(parent, [], "review")
    assert json.loads(gate.gate_state_path().read_text())["state"] == "PENDING"
    assert not (home / "skills/.curator_state").exists()
    assert any("validation OPERATIONAL_ERROR" in message for message in messages)
    assert not any("Self-improvement review: Skill updated" in message for message in messages)
    with gate.acquire_operation_fence(timeout=0):
        pass
