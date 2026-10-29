"""Downloader: yt-dlp option wiring and error classification.

Every test forces ALL ytdlp_* settings fields explicitly: the real .env may
set any of them, and these tests must stay hermetic regardless.
"""
from __future__ import annotations

import re
import sys
import types
import uuid
from pathlib import Path

import pytest

from app.config import settings
from app.services.downloader import (
    DownloadError,
    TransientDownloadError,
    _build_ydl,
    download,
)

TIKTOK_URL = "https://www.tiktok.com/@somehandle/video/7000000000000000001"

_YTDLP_FIELDS = (
    "ytdlp_cookies_file",
    "ytdlp_cookies_from_browser",
    "ytdlp_impersonate",
    "ytdlp_proxy",
    "ytdlp_user_agent",
)


def _clear_ytdlp_settings(monkeypatch) -> None:
    for name in _YTDLP_FIELDS:
        monkeypatch.setattr(settings, name, None)


class _FakeImpersonateTarget:
    """Stand-in for yt_dlp.networking.impersonate.ImpersonateTarget."""

    def __init__(self, value: str):
        self.value = value

    @classmethod
    def from_str(cls, value: str) -> "_FakeImpersonateTarget":
        return cls(value)


class _FakeYoutubeDLError(Exception):
    """Stand-in for yt_dlp.utils.YoutubeDLError."""


def _install_fake_ytdlp(monkeypatch, *, on_construct=None, extract_info=None):
    """Install a fake yt_dlp module in sys.modules and return it.

    `_build_ydl` only does `import yt_dlp` plus attribute lookups
    (yt_dlp.utils.YoutubeDLError, yt_dlp.networking.impersonate...), so a
    ModuleType with the right attributes is enough.
    """
    utils_mod = types.ModuleType("yt_dlp.utils")
    utils_mod.YoutubeDLError = _FakeYoutubeDLError

    impersonate_mod = types.ModuleType("yt_dlp.networking.impersonate")
    impersonate_mod.ImpersonateTarget = _FakeImpersonateTarget
    networking_mod = types.ModuleType("yt_dlp.networking")
    networking_mod.impersonate = impersonate_mod

    ydl_mod = types.ModuleType("yt_dlp")
    ydl_mod.utils = utils_mod
    ydl_mod.networking = networking_mod

    class FakeYoutubeDL:
        last_params = None

        def __init__(self, params):
            if on_construct is not None:
                on_construct(params, _FakeYoutubeDLError)
            self.params = params
            FakeYoutubeDL.last_params = params

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            if extract_info is not None:
                return extract_info(self, url, download)
            raise _FakeYoutubeDLError("no extract_info behaviour configured for this test")

    ydl_mod.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", ydl_mod)
    return ydl_mod


# ------------------------------------------------------------------ opts wiring


def test_cookies_file_is_passed_to_yt_dlp(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(settings, "ytdlp_cookies_file", str(cookies))

    _build_ydl({})
    assert fake.YoutubeDL.last_params["cookiefile"] == str(cookies)


def test_browser_cookies_parse_into_the_yt_dlp_tuple(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)

    monkeypatch.setattr(settings, "ytdlp_cookies_from_browser", "chrome")
    _build_ydl({})
    assert fake.YoutubeDL.last_params["cookiesfrombrowser"] == ("chrome", None, None, None)

    monkeypatch.setattr(settings, "ytdlp_cookies_from_browser", "chrome:Profile 1")
    _build_ydl({})
    assert fake.YoutubeDL.last_params["cookiesfrombrowser"] == (
        "chrome",
        "Profile 1",
        None,
        None,
    )

    # Whitespace around either part is tolerated.
    monkeypatch.setattr(settings, "ytdlp_cookies_from_browser", " chrome : Profile 1 ")
    _build_ydl({})
    assert fake.YoutubeDL.last_params["cookiesfrombrowser"] == (
        "chrome",
        "Profile 1",
        None,
        None,
    )


def test_browser_cookies_without_a_browser_name_is_a_config_error(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    _install_fake_ytdlp(monkeypatch)
    monkeypatch.setattr(settings, "ytdlp_cookies_from_browser", ":Profile 1")

    with pytest.raises(DownloadError) as excinfo:
        _build_ydl({})
    assert "YTDLP_COOKIES_FROM_BROWSER" in str(excinfo.value)


def test_proxy_user_agent_and_impersonate_are_passed_through(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)
    monkeypatch.setattr(settings, "ytdlp_proxy", "http://127.0.0.1:7890")
    monkeypatch.setattr(settings, "ytdlp_user_agent", "Mozilla/5.0 (test)")
    monkeypatch.setattr(settings, "ytdlp_impersonate", "chrome")

    _build_ydl({})
    params = fake.YoutubeDL.last_params
    assert params["proxy"] == "http://127.0.0.1:7890"
    assert params["user_agent"] == "Mozilla/5.0 (test)"
    # yt-dlp wants an ImpersonateTarget object built from the configured name.
    assert params["impersonate"].value == "chrome"


def test_no_session_opts_are_sent_when_nothing_is_configured(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)

    _build_ydl({})
    params = fake.YoutubeDL.last_params
    for key in ("cookiefile", "cookiesfrombrowser", "proxy", "user_agent", "impersonate"):
        assert key not in params


# ---------------------------------------------------- per-user cookies file


def test_per_user_cookiefile_is_used_and_the_browser_jar_disabled(monkeypatch, tmp_path):
    """The requesting user's own cookies win, and the operator's browser jar
    is switched off: yt-dlp would otherwise MERGE the two sessions."""
    _clear_ytdlp_settings(monkeypatch)

    def _stop(self, url, download):
        raise _FakeYoutubeDLError("stop after construction")

    fake = _install_fake_ytdlp(monkeypatch, extract_info=_stop)
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    cookies = tmp_path / "user-cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")

    with pytest.raises(TransientDownloadError):
        download(TIKTOK_URL, uuid.uuid4(), cookiefile=str(cookies))
    params = fake.YoutubeDL.last_params
    assert params["cookiefile"] == str(cookies)
    # Explicit override (the key is present and None), not merely absent.
    assert "cookiesfrombrowser" in params
    assert params.get("cookiesfrombrowser") is None


def test_per_user_cookiefile_wins_over_the_browser_setting(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)
    monkeypatch.setattr(settings, "ytdlp_cookies_from_browser", "chrome")
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    cookies = tmp_path / "user-cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")

    with pytest.raises(TransientDownloadError):
        download(TIKTOK_URL, uuid.uuid4(), cookiefile=str(cookies))
    params = fake.YoutubeDL.last_params
    assert params["cookiefile"] == str(cookies)
    assert params.get("cookiesfrombrowser") is None


def test_missing_per_user_cookiefile_falls_back_to_server_defaults(monkeypatch, tmp_path):
    """A vanished per-user file downgrades to the server session, not an error."""
    _clear_ytdlp_settings(monkeypatch)
    fake = _install_fake_ytdlp(monkeypatch)
    server_cookies = tmp_path / "server-cookies.txt"
    server_cookies.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(settings, "ytdlp_cookies_file", str(server_cookies))
    monkeypatch.setattr(settings, "media_root", str(tmp_path))
    missing = tmp_path / "deleted-user-cookies.txt"

    with pytest.raises(TransientDownloadError):
        download(TIKTOK_URL, uuid.uuid4(), cookiefile=str(missing))
    params = fake.YoutubeDL.last_params
    assert params["cookiefile"] == str(server_cookies)
    assert "cookiesfrombrowser" not in params


# ------------------------------------------------------------- config errors


def test_missing_cookies_file_is_permanent_and_leaks_no_path(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)
    _install_fake_ytdlp(monkeypatch)
    missing = tmp_path / "does-not-exist.txt"
    monkeypatch.setattr(settings, "ytdlp_cookies_file", str(missing))

    with pytest.raises(DownloadError) as excinfo:
        _build_ydl({})
    message = str(excinfo.value)
    # The operator's server path never reaches the chat error.
    assert str(tmp_path) not in message
    assert "missing on the server" in message


def test_missing_impersonation_dependency_is_a_permanent_error(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    monkeypatch.setattr(settings, "ytdlp_impersonate", "chrome")

    def _refuse(params, error_cls):
        if "impersonate" in params:
            raise error_cls(
                "Impersonate target 'chrome' is not available. "
                "You may be missing dependencies."
            )

    _install_fake_ytdlp(monkeypatch, on_construct=_refuse)

    with pytest.raises(DownloadError) as excinfo:
        _build_ydl({})
    message = str(excinfo.value)
    assert "curl-cffi" in message
    assert "YTDLP_IMPERSONATE=chrome" in message


def test_yt_dlp_not_installed_is_transient(monkeypatch):
    _clear_ytdlp_settings(monkeypatch)
    # None in sys.modules makes `import yt_dlp` raise ImportError.
    monkeypatch.setitem(sys.modules, "yt_dlp", None)

    with pytest.raises(TransientDownloadError) as excinfo:
        _build_ydl({})
    assert "yt-dlp is not installed" in str(excinfo.value)


# --------------------------------------------------------- error classification


def test_rate_limiting_is_classified_as_transient(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)

    def _throttled(self, url, download):
        raise _FakeYoutubeDLError("HTTP Error 429: Too Many Requests")

    _install_fake_ytdlp(monkeypatch, extract_info=_throttled)
    monkeypatch.setattr(settings, "media_root", str(tmp_path))

    with pytest.raises(TransientDownloadError) as excinfo:
        download(TIKTOK_URL, uuid.uuid4())
    assert not isinstance(excinfo.value, DownloadError)
    assert "rate-limiting" in str(excinfo.value)


def test_bot_block_is_permanent_and_carries_the_cookie_hint(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)

    def _blocked(self, url, download):
        raise _FakeYoutubeDLError("Unexpected response from webpage request")

    _install_fake_ytdlp(monkeypatch, extract_info=_blocked)
    monkeypatch.setattr(settings, "media_root", str(tmp_path))

    with pytest.raises(DownloadError) as excinfo:
        download(TIKTOK_URL, uuid.uuid4())
    message = str(excinfo.value)
    assert "blocked this request" in message
    assert "YTDLP_COOKIES_FILE" in message
    assert "YTDLP_IMPERSONATE" in message


# ----------------------------------------------------------------- happy path


def test_download_returns_a_complete_result(monkeypatch, tmp_path):
    _clear_ytdlp_settings(monkeypatch)

    description = " ".join(f"word{i}" for i in range(35))  # 35 words -> 20 kept

    def _extract(self, url, download):
        outtmpl = self.params["outtmpl"]["default"]
        assert outtmpl.endswith("media.%(ext)s")
        media = Path(outtmpl.replace("%(ext)s", "mp4"))
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"fake media bytes for the suite")
        return {
            "title": "A late-night walk along the riverside",
            "duration": 61.0,
            "thumbnail": "https://example.com/thumb.jpg",
            "extractor_key": "TikTok",
            "upload_date": "20261028",
            "uploader_id": "somehandle",
            "description": description,
            "requested_downloads": [{"filepath": str(media)}],
        }

    _install_fake_ytdlp(monkeypatch, extract_info=_extract)
    monkeypatch.setattr(settings, "media_root", str(tmp_path))

    result = download(TIKTOK_URL, uuid.uuid4())

    assert result.platform == "tiktok"
    assert result.author_handle == "@somehandle"
    assert result.date_posted == "2026-10-28"
    assert result.caption == " ".join(f"word{i}" for i in range(20))
    assert len(result.caption.split()) == 20
    assert re.fullmatch(r"[0-9a-f]{64}", result.sha256)
    assert result.path.exists()
    assert result.path.name == "media.mp4"
    assert result.file_size == len(b"fake media bytes for the suite")