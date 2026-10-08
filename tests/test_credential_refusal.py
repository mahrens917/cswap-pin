"""A refused token on /v1/messages is reported to cswap.

The owner proxy sees every `/v1/messages` reply. A 401 (or a 403 whose body
names an authentication or permission error) is the only sign a setup-token
died early, so the request's own bearer is handed to
`ClaudeAccountSwitcher.record_token_refused`, which charges it to the slot
whose stored credential carries that token and strikes it when that
credential is a setup-token (claude-swap's own tests cover that half). The
live slot is never consulted: a switch between the reply and the worker
moves the live login, and reading it dropped the refusal.
"""

import json
import socket
import types

from cswap_pin import proxy as pp

TOKEN = "sk-ant-oat01-live"
AUTH_403 = json.dumps(
    {"type": "error", "error": {"type": "permission_error", "message": "no"}}
).encode()
OTHER_403 = json.dumps(
    {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
).encode()


def _live_slot_read(*_a, **_k):
    raise AssertionError("the refusal path read the live slot")


def _wire(monkeypatch, *, recorder=True):
    """Stub cswap's switcher; returns the list of `record_token_refused`
    calls and the daemon log lines. Any read of the live slot fails."""
    calls: list = []
    logged: list = []
    attrs = dict(
        current_account_number=_live_slot_read,
        _read_credentials=_live_slot_read,
    )
    if recorder:
        attrs["record_token_refused"] = (
            lambda token, status: calls.append((token, status)) or True
        )
    fake_switcher = types.SimpleNamespace(
        ClaudeAccountSwitcher=lambda: types.SimpleNamespace(**attrs)
    )

    def _require(name):
        assert name == "switcher", name
        return fake_switcher

    monkeypatch.setattr(pp, "require", _require)
    monkeypatch.setattr(pp, "_spawn_usage_header_recorder", lambda fn: fn())
    monkeypatch.setattr(pp, "_log_lifecycle", logged.append)
    monkeypatch.setattr(pp, "_refusal_recorder_missing_warned", False)
    pp._refusal_spawn_seen.clear()
    return calls, logged


def _relay(status: bytes, body: bytes = b"{}", *, auth=f"Bearer {TOKEN}",
           path="/v1/messages?beta=true") -> bytes:
    up_a, up_b = socket.socketpair()
    cl_a, cl_b = socket.socketpair()
    try:
        head = (
            b"HTTP/1.1 " + status + b"\r\nContent-Type: application/json\r\n"
            + b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        )
        up_b.sendall(head + body)
        up_b.shutdown(socket.SHUT_WR)
        pp._relay_response(up_a, cl_a, 0, method="POST", path=path, auth=auth)
        cl_a.shutdown(socket.SHUT_WR)
        return cl_b.recv(65536)
    finally:
        for s in (up_a, up_b, cl_a, cl_b):
            s.close()


def test_a_401_reports_the_requests_own_token(monkeypatch):
    """Asserts: a 401 on /v1/messages reaches record_token_refused with the
    request's own bearer and the status, without reading the live slot, and
    the client still gets the 401."""
    calls, _ = _wire(monkeypatch)
    got = _relay(b"401 Unauthorized")
    assert got.startswith(b"HTTP/1.1 401"), got[:40]
    assert calls == [(TOKEN, 401)]


def test_a_bearer_cswap_switched_away_from_is_reported_too(monkeypatch):
    """Asserts: a 401 on a session still holding an account cswap switched
    away from is reported with that session's own token, so cswap charges
    it to the account that owns it rather than dropping it."""
    calls, _ = _wire(monkeypatch)
    _relay(b"401 Unauthorized", auth="Bearer sk-ant-oat01-old")
    assert calls == [("sk-ant-oat01-old", 401)]


def test_an_auth_typed_403_is_reported_after_its_body(monkeypatch):
    """Asserts: a 403 whose body's error type is permission_error is
    reported, and the body reaches the client unchanged."""
    calls, _ = _wire(monkeypatch)
    got = _relay(b"403 Forbidden", AUTH_403)
    assert got.endswith(AUTH_403), got
    assert calls == [(TOKEN, 403)]


def test_a_403_of_another_type_is_not_reported(monkeypatch):
    """Asserts: a 403 whose error type is not authentication or permission
    refuses nothing about the credential."""
    calls, _ = _wire(monkeypatch)
    _relay(b"403 Forbidden", OTHER_403)
    assert calls == []


def test_other_routes_are_not_reported(monkeypatch):
    """Asserts: a 403 off the usage endpoint (every setup-token's answer
    there) or a 401 off another route is not a refusal of the token."""
    calls, _ = _wire(monkeypatch)
    _relay(b"403 Forbidden", AUTH_403, path="/api/oauth/usage")
    _relay(b"401 Unauthorized", path="/api/oauth/profile")
    assert calls == []


def test_a_request_with_no_bearer_is_not_reported(monkeypatch):
    """Asserts: a 401 on a request that carried no bearer names no token."""
    calls, _ = _wire(monkeypatch)
    _relay(b"401 Unauthorized", auth="")
    assert calls == []


def test_a_200_is_not_reported(monkeypatch):
    """Asserts: a successful reply reports nothing."""
    calls, _ = _wire(monkeypatch)
    _relay(b"200 OK")
    assert calls == []


def test_a_host_without_the_method_warns_once(monkeypatch):
    """Asserts: an installed claude-swap without record_token_refused gets
    one daemon-log warning, not one per refusal."""
    _, logged = _wire(monkeypatch, recorder=False)
    _relay(b"401 Unauthorized")
    pp._refusal_spawn_seen.clear()
    _relay(b"401 Unauthorized")
    warnings = [line for line in logged if "record_token_refused" in line]
    assert len(warnings) == 1, logged
    assert warnings[0].startswith("warning:")


def test_a_burst_on_one_token_resolves_once(monkeypatch):
    """Asserts: many refusals of one token inside the throttle window are
    reported once."""
    calls, _ = _wire(monkeypatch)
    for _ in range(3):
        _relay(b"401 Unauthorized")
    assert len(calls) == 1
