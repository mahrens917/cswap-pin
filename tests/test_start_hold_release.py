"""The owner proxy releases claude-swap's Remote Control start hold
(board row X3768) on the server's environment registration, and on
nothing else."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import pytest
from claude_swap.start_hold import (
    StartHold,
    read_start_hold,
    write_start_hold,
)
from test_proxy_server import (
    _make_certdir,
    _RecordingChain,
    _request_through_proxy,
    _StallableBridgeUpstream,
)

import cswap_pin.proxy as pp
from cswap_pin.proxy import PinProxy, write_upstream_hint

REGISTER = "/v1/environments/bridge"
OK = b"HTTP/1.1 200 OK"
RELEASED = "start hold released: Remote Control registered"


@pytest.fixture
def backup_dir(tmp_path) -> Path:
    """claude-swap's backup directory; the daemon's cert dir sits inside it
    as `pin-proxy`, the layout both spawn sites build."""
    return tmp_path


@pytest.fixture
def certdir(backup_dir) -> Path:
    d = backup_dir / "pin-proxy"
    d.mkdir()
    return _make_certdir(d)


@pytest.fixture
def lifecycle(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(pp, "_log_lifecycle", lines.append)
    return lines


def _hold(backup_dir: Path, set_at: float | None = None) -> StartHold:
    hold = StartHold(set_at=time.time() - 5 if set_at is None else set_at,
                     owner="1", previous="2")
    write_start_hold(backup_dir, hold)
    return hold


def _bare(certdir: Path) -> PinProxy:
    proxy = PinProxy.__new__(PinProxy)
    proxy._certdir = certdir
    return proxy


class TestTheReleaseDecision:
    def test_the_environment_registration_releases_the_hold(
            self, backup_dir, certdir, lifecycle):
        """Asserts: a 2xx `POST /v1/environments/bridge` deletes the hold
        and logs the release line once."""
        _hold(backup_dir)
        _bare(certdir)._note_remote_control_registered("POST", REGISTER, OK)
        assert read_start_hold(backup_dir) is None
        assert lifecycle == [RELEASED]

    @pytest.mark.parametrize("method,path,status", [
        ("POST", "/v1/code/sessions/cse_A/worker/register", OK),
        ("POST", "/v1/code/sessions/cse_A/bridge", OK),
        ("POST", "/v1/sessions/cse_A/bridge", OK),
        ("POST", REGISTER + "?beta=true", OK),
        ("POST", REGISTER + "/env_1", OK),
        ("DELETE", REGISTER + "/env_1", OK),
        ("POST", "/v1/environments/env_1/bridge/reconnect", OK),
        ("POST", "/v1/messages", OK),
        ("GET", REGISTER, OK),
        ("POST", REGISTER, b"HTTP/1.1 401 Unauthorized"),
        ("POST", REGISTER, b"HTTP/1.1 503 Service Unavailable"),
    ])
    def test_other_traffic_leaves_the_hold(
            self, backup_dir, certdir, lifecycle, method, path, status):
        """Asserts: worker and bridge registrations, the SDK's beta calls,
        other environment routes and a refused registration leave the hold
        in place and log nothing."""
        hold = _hold(backup_dir)
        _bare(certdir)._note_remote_control_registered(method, path, status)
        assert read_start_hold(backup_dir) == hold
        assert lifecycle == []

    def test_no_hold_logs_nothing(self, backup_dir, certdir, lifecycle):
        """Asserts: a registration with no hold set is silent."""
        _bare(certdir)._note_remote_control_registered("POST", REGISTER, OK)
        assert lifecycle == []

    def test_a_hold_set_after_the_registration_stays(
            self, backup_dir, certdir, lifecycle):
        """Asserts: a hold whose set time is later than the registration is
        not released by it."""
        hold = _hold(backup_dir, set_at=time.time() + 60)
        _bare(certdir)._note_remote_control_registered("POST", REGISTER, OK)
        assert read_start_hold(backup_dir) == hold
        assert lifecycle == []

    def test_a_corrupt_hold_is_one_log_line_not_a_raise(
            self, backup_dir, certdir, lifecycle):
        """Asserts: an unreadable hold never raises into the request path;
        it is one daemon log line naming the failure."""
        (backup_dir / "start_hold.json").write_text("not json")
        _bare(certdir)._note_remote_control_registered("POST", REGISTER, OK)
        assert len(lifecycle) == 1
        assert lifecycle[0].startswith("start hold release failed: StartHoldError")


class TestBothForwardingPathsRelease:
    def test_the_absolute_form_path_releases_on_the_registration(
            self, backup_dir, certdir, lifecycle):
        """Asserts: `claude remote-control`'s registration, sent in absolute
        form through this proxy, releases the hold."""
        _hold(backup_dir)
        chain = _RecordingChain(
            lambda req: b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        proxy = None
        try:
            write_upstream_hint(certdir, f"http://127.0.0.1:{chain.port}")
            proxy = PinProxy(certdir=certdir,
                             pin_token_provider=lambda: "PINTOKEN",
                             rediscover_chain=True)
            proxy.start()
            c = socket.create_connection(("127.0.0.1", proxy.port), timeout=10)
            try:
                c.sendall(
                    b"POST https://api.anthropic.com/v1/environments/bridge"
                    b" HTTP/1.1\r\nHost: api.anthropic.com\r\n"
                    b"Authorization: Bearer ACTIVE\r\nContent-Length: 0\r\n\r\n")
                got = b""
                c.settimeout(10)
                while b"\r\n\r\n" not in got:
                    d = c.recv(4096)
                    if not d:
                        break
                    got += d
                assert got.startswith(b"HTTP/1.1 200"), got[:60]
            finally:
                c.close()
        finally:
            if proxy:
                proxy.stop()
            chain.stop()
        assert read_start_hold(backup_dir) is None
        assert RELEASED in lifecycle

    def test_the_tls_path_releases_on_the_registration(
            self, backup_dir, certdir, lifecycle):
        """Asserts: the same registration through the CONNECT tunnel the
        proxy terminates releases the hold too."""
        _hold(backup_dir)
        upstream = _StallableBridgeUpstream(certdir)
        proxy = PinProxy(certdir=certdir, pin_token_provider=lambda: "TESTTOK",
                         upstream=("127.0.0.1", upstream.port))
        proxy.start()
        try:
            status = _request_through_proxy(
                proxy.port, certdir / "ca.pem", REGISTER, bearer="t")
        finally:
            proxy.stop()
            upstream.stop()
        assert status == 200
        assert read_start_hold(backup_dir) is None
        assert RELEASED in lifecycle

    def test_the_tls_path_leaves_the_hold_on_a_worker_registration(
            self, backup_dir, certdir, lifecycle):
        """Asserts: a worker registration through the CONNECT tunnel does
        not release the hold."""
        hold = _hold(backup_dir)
        upstream = _StallableBridgeUpstream(certdir)
        proxy = PinProxy(certdir=certdir, pin_token_provider=lambda: "TESTTOK",
                         upstream=("127.0.0.1", upstream.port))
        proxy.start()
        try:
            status = _request_through_proxy(
                proxy.port, certdir / "ca.pem",
                "/v1/code/sessions/cse_A/worker/register", bearer="t")
        finally:
            proxy.stop()
            upstream.stop()
        assert status == 200
        assert read_start_hold(backup_dir) == hold
        assert RELEASED not in lifecycle
