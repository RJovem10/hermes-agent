"""Behavioral tests for the generic autonomous post-write gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from agent import post_write_validation as gate


@pytest.fixture
def home(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    (profile / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    return profile


def _validator(code: str, *, stability=(), paths=("skills/sample",)) -> gate.Validator:
    return gate.Validator("test-validator", paths, (sys.executable, "-c", code), 2, stability)


def _code(exit_code: int, status: str | None = None) -> str:
    payload = {"status": status} if status is not None else {"errors": [], "warnings": []}
    return f"import json, sys; print(json.dumps({payload!r})); sys.exit({exit_code})"


def test_fence_is_shared_across_threads_and_processes(home):
    assert gate.fence_path() == home / "skills/.locks/self-improvement.fence"
    outcomes = []
    with gate.acquire_operation_fence():
        def attempt():
            try:
                with gate.acquire_operation_fence(timeout=0):
                    outcomes.append("acquired")
            except gate.FenceBusy:
                outcomes.append("busy")

        worker = threading.Thread(target=attempt)
        worker.start()
        worker.join(timeout=2)
        assert outcomes == ["busy"]
        code = (
            "from agent.post_write_validation import acquire_operation_fence, FenceBusy\n"
            "try:\n"
            "    with acquire_operation_fence(timeout=0): print('acquired')\n"
            "except FenceBusy: print('busy')\n"
        )
        child = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parents[2],
                               env=os.environ.copy(), capture_output=True, text=True, timeout=10)
        assert child.returncode == 0
        assert child.stdout.strip() == "busy"
    with gate.acquire_operation_fence(timeout=0):
        pass


def test_config_requires_explicit_opt_in_and_secure_argv(home):
    from hermes_cli.config import _validate_config_key

    assert _validate_config_key("post_write_validators.skill-check.command")[0] is True
    valid = {"post_write_validators": {"lint": {
        "enabled": True, "paths": ["skills/sample"],
        "command": [sys.executable, "-c", "print('{}')"],
        "timeout": 9999, "stability_paths": ["skills/sample"],
    }}}
    assert gate.parse_validators(valid)[0].timeout == gate.MAX_TIMEOUT_S
    assert gate.validators_for_paths(valid, {"skills/sample/SKILL.md"})
    assert gate.validators_for_paths(valid, {"reports/output.json"}) == []
    assert gate.validators_for_paths(valid, {"cache/output.json"}) == []
    for field, value in (("enabled", "true"), ("paths", []), ("command", "echo ok"),
                         ("command", ["echo"]), ("stability_paths", ["../private"]),
                         ("paths", ["C:/outside"])):
        broken = json.loads(json.dumps(valid))
        broken["post_write_validators"]["lint"][field] = value
        with pytest.raises(gate.ValidatorConfigError):
            gate.parse_validators(broken)
    valid["post_write_validators"]["lint"] = {"enabled": False}
    assert gate.parse_validators(valid) == []
    valid["post_write_validators"] = {"..\\outside": {"enabled": False}}
    with pytest.raises(gate.ValidatorConfigError):
        gate.parse_validators(valid)


def test_subprocess_is_unattended_argv_without_shell(home, monkeypatch):
    original = gate.subprocess.Popen
    observed = []

    def inspect_launch(argv, **kwargs):
        assert isinstance(argv, list)
        assert kwargs["shell"] is False
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.PIPE
        assert kwargs["stderr"] is subprocess.PIPE
        observed.append(argv)
        return original(argv, **kwargs)

    monkeypatch.setattr(gate.subprocess, "Popen", inspect_launch)
    code = "import json,sys; assert sys.stdin.read() == ''; print(json.dumps({'status':'pass'}))"
    result = gate.run_validator(_validator(code), "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.PASS
    assert len(observed) == 1


def test_validator_child_environment_tracks_profile_a_b_a(home, tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    other = tmp_path / "other-profile"
    (other / "skills").mkdir(parents=True)
    monkeypatch.setenv("OPENAI_API_KEY", "launch-profile-placeholder")
    code = (
        "import json,os; print(json.dumps({'status':'pass', "
        "'home':os.environ['HERMES_HOME'], "
        "'launch_key':os.environ.get('OPENAI_API_KEY') == 'launch-profile-placeholder'}))"
    )
    validator = _validator(code)
    for expected in (home, other, home):
        token = set_hermes_home_override(expected)
        try:
            result = gate.run_validator(validator, "operation", {"skills/sample/SKILL.md"})
            assert result.state == gate.GateState.PASS
            payload = json.loads(Path(result.artifact_path).read_text())["stdout"]
            assert payload["home"] == str(expected)
            if expected == other:
                assert payload["launch_key"] is False
        finally:
            reset_hermes_home_override(token)


def test_malformed_config_is_operational_error_without_launch(home, monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "post_write_validators": {"broken": {"enabled": True, "paths": ["skills/sample"],
                                            "command": "echo unsafe"}}
    })
    monkeypatch.setattr(gate, "run_validator", lambda *_: pytest.fail("validator launched"))
    result = gate.run_applicable_validators("operation", {"skills/sample/SKILL.md"})[0]
    assert result.state == gate.GateState.OPERATIONAL_ERROR
    assert "validator_config_error" in (result.reason or "")


@pytest.mark.parametrize(("code", "state", "exit_code"), [
    (_code(0, "pass"), gate.GateState.PASS, 0),
    (_code(1, "findings"), gate.GateState.FINDINGS, 1),
    (_code(2, "error"), gate.GateState.OPERATIONAL_ERROR, 2),
    (_code(3, "error"), gate.GateState.OPERATIONAL_ERROR, 3),
    (_code(126, "error"), gate.GateState.OPERATIONAL_ERROR, 126),
    (_code(127, "error"), gate.GateState.OPERATIONAL_ERROR, 127),
    (_code(0, "findings"), gate.GateState.OPERATIONAL_ERROR, 0),
    ("print('not json')", gate.GateState.OPERATIONAL_ERROR, 0),
    ("pass", gate.GateState.OPERATIONAL_ERROR, 0),
])
def test_validator_exit_json_agreement_and_artifacts(home, code, state, exit_code):
    result = gate.run_validator(_validator(code), "operation", {"skills/sample/SKILL.md"})
    assert result.state == state
    assert result.exit_code == exit_code
    assert result.artifact_path and Path(result.artifact_path).exists()
    assert not Path(result.artifact_path).is_relative_to(home / "skills")
    artifact = json.loads(Path(result.artifact_path).read_text())
    assert artifact["operation_id"] == "operation"
    assert artifact["gate"] == state.value


def test_timeout_is_operational_error(home):
    validator = gate.Validator("slow", ("skills/sample",),
                               (sys.executable, "-c", "import time; time.sleep(10)"), 1)
    result = gate.run_validator(validator, "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.OPERATIONAL_ERROR
    assert result.timed_out is True
    assert result.reason == "timeout"


def test_timeout_terminates_grandchildren_holding_pipes(home):
    marker = home / "grandchild-survived"
    child = ("import time; from pathlib import Path; time.sleep(2); "
             f"Path({str(marker)!r}).write_text('survived')")
    code = ("import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(10)")
    validator = gate.Validator("tree", ("skills/sample",),
                               (sys.executable, "-c", code), 1)
    result = gate.run_validator(validator, "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.OPERATIONAL_ERROR
    time.sleep(2.2)
    assert not marker.exists()


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_stability_manifest_invalidates_verdict(home, change):
    root = home / "skills/sample"
    root.mkdir()
    existing = root / "SKILL.md"
    existing.write_text("before")
    if change == "modify":
        script = "from pathlib import Path; Path('skills/sample/SKILL.md').write_text('after')"
    elif change == "add":
        script = "from pathlib import Path; Path('skills/sample/new.txt').write_text('new')"
    else:
        script = "from pathlib import Path; Path('skills/sample/SKILL.md').unlink()"
    code = f"import os, json; os.chdir(os.environ['HERMES_HOME']); {script}; print(json.dumps({{'status':'pass'}}))"
    result = gate.run_validator(_validator(code, stability=("skills/sample",)),
                                "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.INTERRUPTED
    assert result.reason == "corpus_changed_during_validation"
    artifact = json.loads(Path(result.artifact_path).read_text())
    assert artifact["validator_state"] == "PASS"
    assert artifact["validator_exit_code"] == 0
    assert artifact["gate"] == "INTERRUPTED"


def test_findings_verdict_survives_stability_interruption(home):
    root = home / "skills/sample"
    root.mkdir()
    (root / "SKILL.md").write_text("before")
    code = (
        "import json, os; from pathlib import Path; "
        "Path(os.environ['HERMES_HOME'], 'skills/sample/SKILL.md').write_text('after'); "
        "print(json.dumps({'status':'findings'})); raise SystemExit(1)"
    )
    result = gate.run_validator(_validator(code, stability=("skills/sample",)),
                                "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.INTERRUPTED
    artifact = json.loads(Path(result.artifact_path).read_text())
    assert artifact["validator_state"] == "FINDINGS"
    assert artifact["validator_exit_code"] == 1
    assert artifact["gate"] == "INTERRUPTED"


def test_findings_do_not_roll_back_validator_write(home):
    target = home / "skills/sample/SKILL.md"
    target.parent.mkdir()
    target.write_text("before")
    code = (
        "import json, os; from pathlib import Path; "
        "Path(os.environ['HERMES_HOME'], 'skills/sample/SKILL.md').write_text('after'); "
        "print(json.dumps({'status':'findings'})); raise SystemExit(1)"
    )
    result = gate.run_validator(_validator(code), "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.FINDINGS
    assert target.read_text() == "after"


def test_stability_includes_other_configured_root_and_skips_symlinks(home):
    root = home / "other-reference/data"
    root.mkdir(parents=True)
    (root / "reference.txt").write_text("before")
    outside = home / "outside.txt"
    outside.write_text("private")
    (root / "linked.txt").symlink_to(outside)
    manifest = gate.stability_manifest(("other-reference/data",))
    assert "other-reference/data/linked.txt" not in manifest
    code = (
        "import os,json; from pathlib import Path; "
        "Path(os.environ['HERMES_HOME'], 'other-reference/data/reference.txt').write_text('after'); "
        "print(json.dumps({'status':'pass'}))"
    )
    result = gate.run_validator(_validator(code, stability=("other-reference/data",)),
                                "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.INTERRUPTED


def test_unchanged_stability_root_preserves_pass(home):
    root = home / "skills/sample"
    root.mkdir()
    (root / "SKILL.md").write_text("unchanged")
    result = gate.run_validator(_validator(_code(0, "pass"), stability=("skills/sample",)),
                                "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.PASS


def test_symlink_stability_root_is_operational_error(home, monkeypatch):
    outside = home / "elsewhere"
    outside.mkdir()
    (home / "skills/sample").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(gate.subprocess, "Popen", lambda *_a, **_k: pytest.fail("launched"))
    result = gate.run_validator(_validator(_code(0, "pass"), stability=("skills/sample",)),
                                "operation", {"skills/sample/SKILL.md"})
    assert result.state == gate.GateState.OPERATIONAL_ERROR
    assert result.reason.startswith("stability_manifest_error:")


@pytest.mark.parametrize("failure", ["walk", "stat", "read"])
def test_incomplete_stability_scan_cannot_pass(home, monkeypatch, failure):
    root = home / "skills/sample"
    child = root / "child"
    child.mkdir(parents=True)
    target = child / "SKILL.md"
    target.write_text("before")
    if failure == "walk":
        original = gate.os.scandir
        def broken_walk(path):
            if Path(path) == child:
                raise PermissionError("unreadable directory")
            return original(path)
        monkeypatch.setattr(gate.os, "scandir", broken_walk)
    elif failure == "stat":
        original = Path.lstat
        def broken_stat(path, *args, **kwargs):
            if path == root:
                raise OSError("stat failed")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", broken_stat)
    else:
        original = gate.os.open
        def broken_read(path, flags, *args, **kwargs):
            if Path(path) == target:
                raise PermissionError("unreadable file")
            return original(path, flags, *args, **kwargs)
        monkeypatch.setattr(gate.os, "open", broken_read)
    result = gate.run_validator(_validator(_code(0, "pass"), stability=("skills/sample",)),
                                "operation", {"skills/sample/child/SKILL.md"})
    assert result.state == gate.GateState.OPERATIONAL_ERROR
    assert result.reason.startswith("stability_manifest_error:")


def test_ledger_id_window_before_after_and_missing_cursor(home, monkeypatch):
    from tools import skill_ledger

    ledger = home / "skills/.curator_ledger.jsonl"
    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    monkeypatch.setattr(skill_ledger, "ledger_path", lambda: ledger)
    old_id = skill_ledger.append_entry("patch", "sample", actor="curator")
    cursor = gate.capture_ledger_cursor()
    assert cursor == (True, old_id)
    skill_ledger.append_entry(
        "patch", "sample", actor="curator", evidence={"session_id": "ours"},
        before=[{"path": str(home / "skills/sample/old.md"), "sha256": "old"}],
        after=[{"path": str(home / "skills/sample/new.md"), "sha256": "new"}],
    )
    assert gate.changed_paths_since(cursor, started_at="2025-12-31", session_id="ours") == {
        "skills/sample/old.md", "skills/sample/new.md",
    }
    ledger.write_text(ledger.read_text().splitlines()[-1] + "\n")
    assert gate.changed_paths_since(cursor, started_at="2025-12-31") is None
    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: False)
    assert gate.capture_ledger_cursor() == (False, None)
    assert gate.run_applicable_validators("operation", None)[0].state == gate.GateState.NOT_APPLICABLE


def test_ledger_empty_cursor_and_backwards_timestamp_are_complete(home, monkeypatch):
    from tools import skill_ledger

    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    cursor = gate.capture_ledger_cursor()
    assert cursor == (True, None)
    path = home / "skills/.curator_ledger.jsonl"
    target = home / "skills/sample/SKILL.md"
    row = {
        "id": "new", "ts": "2000-01-01T00:00:00+00:00",
        "actor": "curator", "action": "patch", "skill": "sample",
        "evidence": {"session_id": "ours"},
        "before": [], "after": [{"path": str(target)}],
    }
    path.write_text(json.dumps(row) + "\n")
    assert gate.changed_paths_since(cursor, started_at="2099-01-01", session_id="ours") == {
        "skills/sample/SKILL.md",
    }


@pytest.mark.parametrize(("actor", "recorded_session"), [
    ("agent", "ours"), ("curator", "other"), ("curator", 23),
])
def test_unattributed_ledger_row_interrupts_gate(home, monkeypatch, actor, recorded_session):
    from tools import skill_ledger

    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    skill_ledger.append_entry("patch", "sample", actor="curator")
    cursor = gate.capture_ledger_cursor()
    row = {
        "id": "foreign", "ts": "2000-01-01T00:00:00+00:00",
        "actor": actor, "action": "patch", "skill": "sample",
        "evidence": {"session_id": recorded_session},
        "before": [], "after": [{"path": str(home / "skills/sample/SKILL.md")}],
    }
    with skill_ledger.ledger_path().open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    operation = gate.GateOperation("op", "curator", "2099-01-01", cursor, "ours")
    assert gate.finish_operation(operation) == gate.GateState.INTERRUPTED
    assert operation.results[0].reason == "concurrent_or_unattributed_mutation"


@pytest.mark.parametrize("damage", ["malformed", "structural", "unreadable", "missing_cursor"])
def test_incomplete_ledger_window_never_passes(home, monkeypatch, damage):
    from tools import skill_ledger

    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    skill_ledger.append_entry("patch", "sample", actor="curator")
    cursor = gate.capture_ledger_cursor()
    path = skill_ledger.ledger_path()
    if damage == "malformed":
        with path.open("a") as stream:
            stream.write("{bad-json}\n")
    elif damage == "structural":
        with path.open("a") as stream:
            stream.write(json.dumps({"id": "partial", "actor": "curator"}) + "\n")
    elif damage == "unreadable":
        original = Path.read_bytes
        def unreadable(candidate):
            if candidate == path:
                raise PermissionError("ledger unreadable")
            return original(candidate)
        monkeypatch.setattr(Path, "read_bytes", unreadable)
    else:
        path.write_text("")
    paths = gate.changed_paths_since(cursor, started_at="2000-01-01")
    assert paths is None
    assert gate.run_applicable_validators("op", paths)[0].state != gate.GateState.PASS


@pytest.mark.parametrize("damage", ["older_malformed", "duplicate_cursor"])
def test_ledger_cursor_requires_one_strict_physical_history(home, monkeypatch, damage):
    from tools import skill_ledger

    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    ledger = home / "skills/.curator_ledger.jsonl"
    skill_ledger.append_entry("patch", "sample", actor="curator")
    cursor = gate.capture_ledger_cursor()
    original = ledger.read_text()
    skill_ledger.append_entry(
        "patch", "sample", actor="curator",
        after=[{"path": str(home / "skills/sample/SKILL.md")}],
    )
    if damage == "older_malformed":
        ledger.write_text("{bad-json}\n" + ledger.read_text())
    else:
        ledger.write_text(ledger.read_text() + original)
    paths = gate.changed_paths_since(cursor, started_at="2000-01-01")
    assert paths is None
    assert gate.run_applicable_validators("op", paths)[0].state != gate.GateState.PASS


@pytest.mark.parametrize(("exit_code", "validator_state"), [
    (0, "PASS"), (1, "FINDINGS"),
])
def test_ledger_write_during_validator_interrupts_and_preserves_verdict(
    home, monkeypatch, exit_code, validator_state,
):
    from tools import skill_ledger

    monkeypatch.setattr(skill_ledger, "ledger_enabled", lambda: True)
    target = home / "skills/sample/SKILL.md"
    target.parent.mkdir()
    target.write_text("skill")
    cursor = gate.capture_ledger_cursor()
    skill_ledger.append_entry(
        "patch", "sample", actor="curator", evidence={"session_id": "ours"},
        after=[{"path": str(target)}],
    )
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {
        "post_write_validators": {"check": {
            "enabled": True, "paths": ["skills/sample"],
            "command": [sys.executable, "-c",
                        _code(exit_code, "pass" if exit_code == 0 else "findings")],
        }},
    })
    run_validators = gate.run_applicable_validators

    def concurrent_write(operation_id, changed_paths):
        results = run_validators(operation_id, changed_paths)
        skill_ledger.append_entry(
            "patch", "sample", actor="agent", after=[{"path": str(target)}],
        )
        return results

    monkeypatch.setattr(gate, "run_applicable_validators", concurrent_write)
    operation = gate.GateOperation("op", "curator", gate._now(), cursor, "ours")
    assert gate.finish_operation(operation) == gate.GateState.INTERRUPTED
    result = operation.results[0]
    assert result.state == gate.GateState.INTERRUPTED
    artifact = json.loads(Path(result.artifact_path).read_text())
    assert artifact["gate"] == "INTERRUPTED"
    assert artifact["validator_state"] == validator_state
    assert artifact["validator_exit_code"] == exit_code
    assert artifact["validator_reason"] is None


def test_pending_recovery_and_terminal_state(home):
    with gate.acquire_operation_fence():
        prior = gate.begin_operation("curator")
    with gate.acquire_operation_fence():
        current = gate.begin_operation("background_review")
        pending = json.loads(gate.gate_state_path().read_text())
        assert pending["state"] == "PENDING"
        assert pending["last"]["operation_id"] == prior.operation_id
        assert pending["last"]["state"] == "INTERRUPTED"
        state = gate.finish_operation(current)
        assert state == gate.GateState.NOT_APPLICABLE
        assert json.loads(gate.gate_state_path().read_text())["state"] == "NOT_APPLICABLE"
