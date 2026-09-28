"""Post-write gates for autonomous skill operations.

The skill ledger is best-effort telemetry.  An absent ledger window cannot
prove that a write set passed validation; the gate sidecar is the durable
operation record.  Foreground skill writes do not acquire this fence.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import signal
import stat
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator, Mapping, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
FENCE_TIMEOUT_S = 5.0
MIN_TIMEOUT_S = 1
MAX_TIMEOUT_S = 600


class GateState(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PENDING = "PENDING"
    PASS = "PASS"
    FINDINGS = "FINDINGS"
    OPERATIONAL_ERROR = "OPERATIONAL_ERROR"
    INTERRUPTED = "INTERRUPTED"


class FenceBusy(RuntimeError):
    """A second autonomous operation could not acquire the shared fence."""


class ValidatorConfigError(ValueError):
    """A validator definition cannot be executed safely."""


class ConcurrentMutation(RuntimeError):
    """A complete ledger window contains a write not attributable to this run."""


@dataclass(frozen=True)
class Validator:
    name: str
    paths: tuple[str, ...]
    command: tuple[str, ...]
    timeout: int
    stability_paths: tuple[str, ...] = ()


@dataclass
class GateResult:
    validator_name: Optional[str]
    state: GateState
    operation_id: str
    exit_code: Optional[int] = None
    errors: Any = None
    warnings: Any = None
    info: Any = None
    reason: Optional[str] = None
    duration_s: float = 0.0
    timed_out: bool = False
    artifact_path: Optional[str] = None
    changed_paths_count: int = 0
    tool_release: Any = None
    validator_state: Optional[GateState] = None
    validator_reason: Optional[str] = None
    validator_exit_code: Optional[int] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fence_path() -> Path:
    return get_hermes_home() / "skills" / ".locks" / "self-improvement.fence"


def gate_state_path() -> Path:
    return get_hermes_home() / "skills" / ".post_write_validation_state.json"


@contextmanager
def acquire_operation_fence(timeout: float = FENCE_TIMEOUT_S) -> Iterator[Path]:
    """Bounded cross-process lock; never degrade to an unlocked write."""
    path = fence_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if fcntl is None and msvcrt is None:
        raise RuntimeError("cross-process file locking unavailable")
    fd = open(path, "a+b")
    acquired = False
    try:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # pragma: no cover - Windows
                    fd.seek(0)
                    if not fd.read(1):
                        fd.write(b" ")
                        fd.flush()
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except (OSError, IOError) as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise FenceBusy(f"self-improvement fence busy: {path}") from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield path
    finally:
        try:
            if acquired:
                if fcntl is not None:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                else:  # pragma: no cover - Windows
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            fd.close()


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValidatorConfigError("paths must be non-empty POSIX relative strings")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive
            or any(part in (".", "..") for part in value.split("/"))):
        raise ValidatorConfigError(f"invalid HERMES_HOME-relative path: {value!r}")
    normalized = path.as_posix()
    if normalized in ("", ".") or normalized != value.rstrip("/"):
        raise ValidatorConfigError(f"invalid HERMES_HOME-relative path: {value!r}")
    return normalized


def _path_list(name: str, value: Any, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (required and not value) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValidatorConfigError(f"{name} must be {'a non-empty' if required else 'a'} list[str]")
    return tuple(_relative_path(item) for item in value)


def parse_validators(config: Mapping[str, Any]) -> list[Validator]:
    raw = config.get("post_write_validators", {})
    if not isinstance(raw, dict):
        raise ValidatorConfigError("post_write_validators must be a mapping")
    validators: list[Validator] = []
    for name, item in raw.items():
        if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                or name in (".", "..")):
            raise ValidatorConfigError("validator names must be safe path components")
        if not isinstance(item, dict) or type(item.get("enabled")) is not bool:
            raise ValidatorConfigError(f"{name}: enabled must be bool")
        if not item["enabled"]:
            continue
        paths = _path_list(f"{name}.paths", item.get("paths"), required=True)
        command = item.get("command")
        if not isinstance(command, list) or not command or any(
            not isinstance(arg, str) or not arg for arg in command
        ):
            raise ValidatorConfigError(f"{name}.command must be a non-empty argv list[str]")
        if not Path(command[0]).is_absolute():
            raise ValidatorConfigError(f"{name}.command[0] must be absolute")
        timeout = item.get("timeout", 60)
        if type(timeout) is not int:
            raise ValidatorConfigError(f"{name}.timeout must be an integer")
        timeout = max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, timeout))
        stability = _path_list(
            f"{name}.stability_paths", item.get("stability_paths", []), required=False
        )
        validators.append(Validator(name, paths, tuple(command), timeout, stability))
    return validators


def validators_for_paths(config: Mapping[str, Any], changed_paths: set[str]) -> list[Validator]:
    def matches(path: str, root: str) -> bool:
        return path == root or path.startswith(root + "/")

    return [
        validator for validator in parse_validators(config)
        if any(matches(path, root) for path in changed_paths for root in validator.paths)
    ]


def _parse_gate_ledger_row(line: bytes) -> dict[str, Any]:
    """Validate the ledger shape without the audit reader's skip-bad-lines policy."""
    try:
        row = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed skill ledger JSON") from exc
    if not isinstance(row, dict) or not all(
        isinstance(row.get(key), str) and row[key]
        for key in ("id", "ts", "actor", "action", "skill")
    ):
        raise ValueError("invalid skill ledger row")
    if not isinstance(row.get("evidence"), dict):
        raise ValueError("invalid skill ledger evidence")
    for section in ("before", "after"):
        items = row.get(section)
        if not isinstance(items, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            for item in items
        ):
            raise ValueError(f"invalid skill ledger {section} paths")
    return row


def _gate_ledger_lines() -> Optional[list[bytes]]:
    """Return physical JSONL lines; None means the file cannot be trusted."""
    from tools import skill_ledger

    try:
        path = skill_ledger.ledger_path()
        if not path.exists():
            return []
        content = path.read_bytes()
        if content and not content.endswith(b"\n"):
            return None  # An append may have stopped partway through a row.
        return content.splitlines()
    except OSError as exc:
        logger.warning("skill ledger unavailable: %s", exc)
        return None


def capture_ledger_cursor() -> tuple[bool, Optional[str]]:
    """Capture the newest physical row ID before writes, including an empty ledger."""
    from tools import skill_ledger

    try:
        if not skill_ledger.ledger_enabled():
            return False, None
        lines = _gate_ledger_lines()
        if lines is None:
            return False, None
        if not lines:
            return True, None
        return True, _parse_gate_ledger_row(lines[-1])["id"]
    except (OSError, ValueError) as exc:
        logger.warning("skill ledger cursor unavailable: %s", exc)
        return False, None


def changed_paths_since(
    cursor: tuple[bool, Optional[str]], *, started_at: str, session_id: Optional[str] = None
) -> Optional[set[str]]:
    """Read a complete ID window, then attribute every mutation in that window."""
    from tools import skill_ledger

    available, cursor_id = cursor
    if not available:
        return None
    try:
        if not skill_ledger.ledger_enabled():
            return None
        lines = _gate_ledger_lines()
        if lines is None:
            return None
        rows = [_parse_gate_ledger_row(line) for line in lines]
        if cursor_id is None:
            window = rows
        else:
            positions = [index for index, row in enumerate(rows) if row["id"] == cursor_id]
            if len(positions) != 1:
                return None
            window = rows[positions[0] + 1:]
        paths: set[str] = set()
        for row in window:
            # The timestamp is diagnostic only. The ID cursor defines the
            # complete window, including foreground writes that bypass the fence.
            recorded_session = row["evidence"].get("session_id")
            if row["actor"] != "curator" or (
                recorded_session not in (None, "")
                and (not isinstance(recorded_session, str)
                     or not session_id or recorded_session != session_id)
            ):
                raise ConcurrentMutation("concurrent_or_unattributed_mutation")
            for section in ("before", "after"):
                for item in row[section]:
                    raw = item["path"]
                    try:
                        relative = Path(raw).relative_to(get_hermes_home()).as_posix()
                        paths.add(_relative_path(relative))
                    except (ValueError, ValidatorConfigError):
                        raise ConcurrentMutation("concurrent_or_unattributed_mutation")
        return paths
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("skill ledger window unavailable: %s", exc)
        return None


def stability_manifest(roots: tuple[str, ...]) -> dict[str, str]:
    """Hash every ordinary file under each root, or fail if a scan is incomplete."""
    home = get_hermes_home()
    output: dict[str, str] = {}

    def hash_file(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OSError(errno.EINVAL, f"not an ordinary file: {path}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        output[path.relative_to(home).as_posix()] = digest.hexdigest()

    def visit(directory: Path) -> None:
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            mode = entry.stat(follow_symlinks=False).st_mode
            path = Path(entry.path)
            if stat.S_ISLNK(mode):
                continue
            if stat.S_ISDIR(mode):
                visit(path)
            elif stat.S_ISREG(mode):
                hash_file(path)

    for root in sorted(set(roots)):
        path = home / _relative_path(root)
        relative = path.relative_to(home)
        for index in range(1, len(relative.parts) + 1):
            component = home / Path(*relative.parts[:index])
            if stat.S_ISLNK(component.lstat().st_mode):
                raise OSError(errno.ELOOP, f"symlink stability root: {component}")
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            hash_file(path)
        elif stat.S_ISDIR(mode):
            visit(path)
        else:
            raise OSError(errno.EINVAL, f"invalid stability root type: {path}")
    return dict(sorted(output.items()))


def _scrub_argv(argv: tuple[str, ...]) -> list[str]:
    secret_flags = ("token", "secret", "password", "api-key", "credential")
    scrubbed: list[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            scrubbed.append("[redacted]")
            hide_next = False
        elif arg.startswith("-") and any(key in arg.lower() for key in secret_flags):
            if "=" in arg:
                scrubbed.append(arg.split("=", 1)[0] + "=[redacted]")
            else:
                scrubbed.append(arg)
                hide_next = True
        else:
            scrubbed.append(arg)
    return scrubbed


def _terminate_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    def partial_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value if isinstance(value, str) else ""

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - Windows
        process.terminate()
    try:
        return process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired as first_timeout:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - Windows
            process.kill()
        try:
            return process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired as second_timeout:
            # A descendant may have escaped the process group while retaining
            # inherited pipe descriptors.  Do not block indefinitely on it.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
            return (partial_text(second_timeout.stdout or first_timeout.stdout),
                    partial_text(second_timeout.stderr or first_timeout.stderr))


def _json_agrees(payload: dict[str, Any], exit_code: int) -> bool:
    if "exit_code" in payload and payload["exit_code"] != exit_code:
        return False
    status = payload.get("status")
    if status is None:
        return True
    if not isinstance(status, str):
        return False
    expected = (
        GateState.PASS if exit_code == 0 else
        GateState.FINDINGS if exit_code == 1 else GateState.OPERATIONAL_ERROR
    )
    aliases = {
        GateState.PASS: {"pass", "passed", "ok", "success"},
        GateState.FINDINGS: {"findings", "fail", "failed"},
        GateState.OPERATIONAL_ERROR: {"error", "operational_error"},
    }
    return status.lower() in aliases[expected]


def run_validator(validator: Validator, operation_id: str, changed_paths: set[str]) -> GateResult:
    started = time.monotonic()
    stdout = stderr = ""
    payload: Optional[dict[str, Any]] = None
    exit_code: Optional[int] = None
    timed_out = False
    reason: Optional[str] = None
    stability: dict[str, list[str]] = {"new": [], "changed": [], "missing": []}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    manifest_available = True
    try:
        try:
            before = stability_manifest(validator.stability_paths)
        except (OSError, ValueError) as exc:
            manifest_available = False
            reason = f"stability_manifest_error: {exc}"
        if manifest_available:
            from tools.environments.local import served_profile_child_env

            env = served_profile_child_env(target_home=get_hermes_home(), inherit_credentials=True)
            process = subprocess.Popen(
                list(validator.command), shell=False, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=env, start_new_session=(os.name == "posix"),
            )
            try:
                stdout, stderr = process.communicate(timeout=validator.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stdout, stderr = _terminate_group(process)
                reason = "timeout"
            exit_code = process.returncode
    except (OSError, ValueError) as exc:
        reason = f"validator_execution_error: {exc}"
    finally:
        try:
            after = stability_manifest(validator.stability_paths)
            stability = {
                "new": sorted(after.keys() - before.keys()),
                "changed": sorted(key for key in before.keys() & after.keys() if before[key] != after[key]),
                "missing": sorted(before.keys() - after.keys()),
            }
        except (OSError, ValueError) as exc:
            manifest_available = False
            reason = f"stability_manifest_error: {exc}"

    state = GateState.OPERATIONAL_ERROR
    if reason is None:
        try:
            parsed = json.loads(stdout)
            if not isinstance(parsed, dict):
                raise ValueError("stdout must be a JSON object")
            payload = parsed
            if exit_code is None or not _json_agrees(payload, exit_code):
                reason = "process_json_disagreement"
            else:
                state = (GateState.PASS if exit_code == 0 else
                         GateState.FINDINGS if exit_code == 1 else GateState.OPERATIONAL_ERROR)
        except (json.JSONDecodeError, ValueError) as exc:
            reason = f"invalid_json: {exc}"
    validator_state = state
    validator_reason = reason
    if manifest_available and any(stability.values()):
        state = GateState.INTERRUPTED
        reason = "corpus_changed_during_validation"
    result = GateResult(
        validator_name=validator.name, state=state, operation_id=operation_id,
        exit_code=exit_code, errors=payload.get("errors") if payload else None,
        warnings=payload.get("warnings") if payload else None,
        info=payload.get("info") if payload else None, reason=reason,
        duration_s=round(time.monotonic() - started, 3), timed_out=timed_out,
        changed_paths_count=len(changed_paths),
        tool_release=(payload.get("release", payload.get("version")) if payload else None),
        validator_state=validator_state, validator_reason=validator_reason,
        validator_exit_code=exit_code,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact = get_hermes_home() / "logs" / "post-write-validation" / validator.name / f"{stamp}-{operation_id}.json"
    try:
        atomic_json_write(artifact, {
            "operation_id": operation_id, "validator": validator.name,
            "argv": _scrub_argv(validator.command), "exit_code": exit_code,
            "stdout": payload, "raw_stdout": None if payload is not None else stdout,
            "stderr": stderr, "gate": state.value, "reason": reason,
            "validator_state": validator_state.value,
            "validator_reason": validator_reason,
            "validator_exit_code": exit_code,
            "stability": stability, "duration_s": result.duration_s,
            "changed_paths": sorted(changed_paths), "timed_out": timed_out,
        }, indent=2, sort_keys=True)
        result.artifact_path = str(artifact)
    except Exception as exc:
        logger.error("validator artifact could not be retained: %s", exc)
        result.state = GateState.OPERATIONAL_ERROR
        result.reason = "artifact_persistence_failed"
    return result


def run_applicable_validators(operation_id: str, changed_paths: Optional[set[str]]) -> list[GateResult]:
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        parse_validators(config)
    except Exception as exc:
        return [GateResult(None, GateState.OPERATIONAL_ERROR, operation_id,
                           reason=f"validator_config_error: {exc}",
                           changed_paths_count=len(changed_paths or set()))]
    if changed_paths is None:
        return [GateResult(None, GateState.NOT_APPLICABLE, operation_id, reason="ledger_unavailable")]
    if not changed_paths:
        return [GateResult(None, GateState.NOT_APPLICABLE, operation_id, reason="no_changed_paths")]
    selected = validators_for_paths(config, changed_paths)
    if not selected:
        return [GateResult(None, GateState.NOT_APPLICABLE, operation_id,
                           reason="no_matching_validator", changed_paths_count=len(changed_paths))]
    return [run_validator(item, operation_id, changed_paths) for item in selected]


def _aggregate(results: list[GateResult]) -> GateState:
    for state in (GateState.INTERRUPTED, GateState.OPERATIONAL_ERROR, GateState.FINDINGS):
        if any(result.state == state for result in results):
            return state
    return GateState.PASS if any(result.state == GateState.PASS for result in results) else GateState.NOT_APPLICABLE


def _revise_results(
    results: list[GateResult], operation_id: str, state: GateState, reason: str,
) -> list[GateResult]:
    """Revise the gate verdict while retaining each validator's execution verdict."""
    if not results:
        return [GateResult(None, state, operation_id, reason=reason)]
    for result in results:
        result.validator_state = result.validator_state or result.state
        result.validator_reason = result.validator_reason or result.reason
        result.validator_exit_code = (
            result.validator_exit_code if result.validator_exit_code is not None else result.exit_code
        )
        result.state = state
        result.reason = reason
        if result.artifact_path:
            path = Path(result.artifact_path)
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact.update(
                gate=state.value, reason=reason,
                validator_state=result.validator_state.value,
                validator_reason=result.validator_reason,
                validator_exit_code=result.validator_exit_code,
            )
            atomic_json_write(path, artifact, indent=2, sort_keys=True)
    return results


def record_gate_state(
    operation_id: str, operation_kind: str, state: GateState, *,
    started_at: str, results: Optional[list[GateResult]] = None,
    reason: Optional[str] = None, session_id: Optional[str] = None,
    validator_candidates: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Atomically persist compact gate metadata; caller holds the fence."""
    results = results or []
    record = {
        "operation_id": operation_id, "operation_kind": operation_kind,
        "state": state.value, "started_at": started_at,
        "completed_at": None if state == GateState.PENDING else _now(),
        "session_id": session_id, "reason": reason,
        "validator_candidates": validator_candidates or [],
        "results": [{
            "validator_name": result.validator_name, "state": result.state.value,
            "exit_code": result.exit_code, "reason": result.reason,
            "artifact_path": result.artifact_path,
            "changed_paths_count": result.changed_paths_count,
            "validator_state": (result.validator_state.value
                                if result.validator_state else None),
            "validator_reason": result.validator_reason,
            "validator_exit_code": result.validator_exit_code,
        } for result in results],
    }
    path = gate_state_path()
    previous_last = None
    last_curator = None
    if path.exists():
        try:
            previous_record = json.loads(path.read_text(encoding="utf-8"))
            previous_last = previous_record.get("last")
            last_curator = previous_record.get("last_curator")
            if (previous_record.get("operation_kind") == "curator"
                    and previous_record.get("state") != GateState.PENDING.value):
                last_curator = {key: previous_record.get(key) for key in (
                    "operation_id", "operation_kind", "state", "started_at",
                    "completed_at", "reason", "results",
                )}
            if validator_candidates is None and state != GateState.PENDING:
                record["validator_candidates"] = previous_record.get("validator_candidates", [])
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if state == GateState.PENDING and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("state") == GateState.PENDING.value:
                previous["state"] = GateState.INTERRUPTED.value
                previous["completed_at"] = _now()
                previous["reason"] = "previous_operation_interrupted"
                atomic_json_write(path, previous, indent=2, sort_keys=True)
                previous_last = {key: previous.get(key) for key in (
                    "operation_id", "operation_kind", "state", "started_at",
                    "completed_at", "reason",
                )}
                if previous.get("operation_kind") == "curator":
                    last_curator = {key: previous.get(key) for key in (
                        "operation_id", "operation_kind", "state", "started_at",
                        "completed_at", "reason", "results",
                    )}
                logger.warning("recovered interrupted post-write operation %s", previous.get("operation_id"))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"cannot recover previous gate state: {exc}") from exc
    if last_curator is not None:
        record["last_curator"] = last_curator
    if previous_last is not None:
        record["last"] = previous_last
    elif state == GateState.PENDING and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            record["last"] = {key: previous.get(key) for key in (
                "operation_id", "operation_kind", "state", "started_at",
                "completed_at", "reason",
            )}
        except (OSError, ValueError, TypeError):
            pass
    atomic_json_write(path, record, indent=2, sort_keys=True)
    logger.info("post-write gate %s %s %s", operation_id, operation_kind, state.value)
    return record


@dataclass
class GateOperation:
    operation_id: str
    operation_kind: str
    started_at: str
    cursor: tuple[bool, Optional[str]]
    session_id: Optional[str] = None
    results: list[GateResult] = field(default_factory=list)
    persistence_error: Optional[str] = None


def begin_operation(kind: str, *, session_id: Optional[str] = None) -> GateOperation:
    """Call only after acquiring the fence and before the first corpus write."""
    operation = GateOperation(uuid.uuid4().hex, kind, _now(), capture_ledger_cursor(), session_id)
    try:
        from hermes_cli.config import load_config_readonly
        candidates = [item.name for item in parse_validators(load_config_readonly())]
    except Exception:
        candidates = []  # A malformed definition gets a terminal config error.
    record_gate_state(operation.operation_id, kind, GateState.PENDING,
                      started_at=operation.started_at, session_id=session_id,
                      validator_candidates=candidates)
    return operation


def finish_operation(operation: GateOperation, *, interrupted: bool = False,
                     failure: Optional[str] = None) -> GateState:
    """Run at most once, still under the fence; retain a terminal verdict."""
    if interrupted or failure is not None:
        state = GateState.INTERRUPTED if interrupted else GateState.OPERATIONAL_ERROR
        operation.results = [GateResult(None, state, operation.operation_id,
                                        reason=failure or "operation_interrupted")]
    else:
        try:
            # Capture the high-water row before selecting validators, then read
            # the complete ID window again after they return. A foreground
            # write can bypass the operation fence while a validator runs.
            pre_validation_highwater = capture_ledger_cursor()
            paths = changed_paths_since(operation.cursor, started_at=operation.started_at,
                                        session_id=operation.session_id)
            operation.results = run_applicable_validators(operation.operation_id, paths)
            state = _aggregate(operation.results)
            if any(result.validator_name is not None for result in operation.results):
                post_validation_highwater = capture_ledger_cursor()
                if not post_validation_highwater[0]:
                    state = GateState.OPERATIONAL_ERROR
                    operation.results = _revise_results(
                        operation.results, operation.operation_id, state,
                        "ledger_unavailable_after_validation",
                    )
                else:
                    final_paths = changed_paths_since(
                        operation.cursor, started_at=operation.started_at,
                        session_id=operation.session_id,
                    )
                    if final_paths is None:
                        state = GateState.OPERATIONAL_ERROR
                        operation.results = _revise_results(
                            operation.results, operation.operation_id, state,
                            "ledger_unavailable_after_validation",
                        )
                    elif pre_validation_highwater != post_validation_highwater or final_paths != paths:
                        operation.results = _revise_results(
                            operation.results, operation.operation_id, GateState.INTERRUPTED,
                            "ledger_changed_during_validation",
                        )
                        state = GateState.INTERRUPTED
        except ConcurrentMutation:
            state = GateState.INTERRUPTED
            operation.results = _revise_results(
                operation.results, operation.operation_id, state,
                "concurrent_or_unattributed_mutation",
            )
        except Exception as exc:
            state = GateState.OPERATIONAL_ERROR
            operation.results = [GateResult(None, state, operation.operation_id,
                                            reason=f"gate_execution_error: {exc}")]
    try:
        record_gate_state(operation.operation_id, operation.operation_kind, state,
                          started_at=operation.started_at, results=operation.results,
                          reason=operation.results[0].reason if operation.results else None,
                          session_id=operation.session_id)
    except Exception as exc:
        logger.exception("post-write terminal state could not be persisted")
        operation.persistence_error = str(exc)
        state = GateState.OPERATIONAL_ERROR
        operation.results.append(GateResult(
            None, state, operation.operation_id,
            reason="terminal_state_persistence_failed",
        ))
    return state
