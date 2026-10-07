"""A refused setup-token on /v1/messages is reported to cswap.

The owner proxy sees every `/v1/messages` reply. A 401 (or a 403 whose body
names an authentication or permission error) on a request whose bearer IS the
live slot's stored setup-token is the only sign that token died early, so it
is handed to `ClaudeAccountSwitcher.record_credential_refused`, which strikes
the slot out of rotation. A stale bearer (a session still on an account cswap
switched away from) and a browser login are never reported.
"""

import json
import socket
import types

import pytest

from claude_swap import oauth as real_oauth
from cswap_pin import proxy as pp

TOKEN = "sk-ant-oat01-live"
TOKEN_CREDS = json.dumps(
    {"claudeAiOauth": {"accessToken": TOKEN, "scopes": ["user:inference"]}}
)
BROWSER_CREDS = json.dumps(
    {
        "claudeAiOauth": {
            "accessToken": TOKEN,
            "refreshToken": "rt",
            "scopes": ["user:inference", "user:profile"],
        }
    }
)
AUTH_403 = json.dumps(
    {"type": "error", "error": {"type": "permission_error", "message": "no"}}
).encode()
OTHER_403 = json.dumps(
    {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
).encode()


def _wire(monkeypatch, *, live_creds=TOKEN_CREDS, live_num="2", recorder=True):
    """Stub cswap's switcher (the real `oauth` module stays); returns the
    list of `record_credential_refused` calls and the daemon log lines."""
    calls: list = []
    logged: list = []
    attrs = dict(
        current_account_number=lambda: live_num,
        _read_credentials=lambda: live_creds,
    )
    if recorder:
        attrs["record_credential_refused"] = (
            lambda num, status, fp: calls.append((num, status, fp)) or True
        )
    fake_switcher = types.SimpleNamespace(
        ClaudeAccountSwitcher=lambda: types.SimpleNamespace(**attrs)
    )

    def _require(name):
        return real_oauth if name == "oauth" else fake_switcher

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


def test_a_401_on_the_live_token_slot_is_reported(monkeypatch):
    """Asserts: a 401 on /v1/messages whose bearer is the live setup-token
    reaches record_credential_refused with the live slot, the status and the
    stored credential's fingerprint, and the client still gets the 401."""
    calls, _ = _wire(monkeypatch)
    got = _relay(b"401 Unauthorized")
    assert got.startswith(b"HTTP/1.1 401"), got[:40]
    assert calls == [("2", 401, real_oauth.credential_fingerprint(TOKEN_CREDS))]


def test_an_auth_typed_403_is_reported_after_its_body(monkeypatch):
    """Asserts: a 403 whose body's error type is permission_error is
    reported, and the body reaches the client unchanged."""
    calls, _ = _wire(monkeypatch)
    got = _relay(b"403 Forbidden", AUTH_403)
    assert got.endswith(AUTH_403), got
    assert [(n, s) for n, s, _ in calls] == [("2", 403)]


def test_a_403_of_another_type_is_not_reported(monkeypatch):
    """Asserts: a 403 whose error type is not authentication or permission
    refuses nothing about the credential."""
    calls, _ = _wire(monkeypatch)
    _relay(b"403 Forbidden", OTHER_403)
    assert calls == []


def test_a_stale_bearer_401_strikes_nothing(monkeypatch):
    """Asserts: a 401 for a bearer that is not the live slot's token (a
    session still on an account cswap switched away from) is never
    reported; the rebuild path owns it."""
    calls, _ = _wire(monkeypatch)
    _relay(b"401 Unauthorized", auth="Bearer sk-ant-oat01-old")
    assert calls == []


def test_a_browser_login_is_never_reported(monkeypatch):
    """Asserts: a 401 on a live browser login (refresh token present) is
    not reported; its refresh machinery owns its verdict."""
    calls, _ = _wire(monkeypatch, live_creds=BROWSER_CREDS)
    _relay(b"401 Unauthorized")
    assert calls == []


def test_other_routes_are_not_reported(monkeypatch):
    """Asserts: a 403 off the usage endpoint (every setup-token's answer
    there) or a 401 off another route is not a refusal of the token."""
    calls, _ = _wire(monkeypatch)
    _relay(b"403 Forbidden", AUTH_403, path="/api/oauth/usage")
    _relay(b"401 Unauthorized", path="/api/oauth/profile")
    assert calls == []


def test_a_200_is_not_reported(monkeypatch):
    """Asserts: a successful reply reports nothing."""
    calls, _ = _wire(monkeypatch)
    _relay(b"200 OK")
    assert calls == []


def test_a_host_without_the_method_warns_once(monkeypatch):
    """Asserts: an installed claude-swap without record_credential_refused
    gets one daemon-log warning, not one per refusal."""
    _, logged = _wire(monkeypatch, recorder=False)
    _relay(b"401 Unauthorized")
    pp._refusal_spawn_seen.clear()
    _relay(b"401 Unauthorized")
    warnings = [line for line in logged if "record_credential_refused" in line]
    assert len(warnings) == 1, logged
    assert warnings[0].startswith("warning:")


def test_a_burst_on_one_token_resolves_once(monkeypatch):
    """Asserts: many refusals of one token inside the throttle window
    resolve the live slot once."""
    calls, _ = _wire(monkeypatch)
    for _ in range(3):
        _relay(b"401 Unauthorized")
    assert len(calls) == 1


@pytest.mark.parametrize("raw", ["", "not json"])
def test_an_unreadable_live_credential_reports_nothing(monkeypatch, raw):
    """Asserts: with no readable live token nothing is attributed."""
    calls, _ = _wire(monkeypatch, live_creds=raw)
    _relay(b"401 Unauthorized")
    assert calls == []
