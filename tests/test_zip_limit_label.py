"""The ZIP guard rails are a size answer, not a "nothing to export" answer.

`zip_export.build_folder_zip` signals both conditions with `ValueError`; the
route used to map every one of them to `409 nothing_to_export`. A folder that
merely exceeds the limits (5000 entries / 2 GiB, API_CONTRACT §3) is not
empty, and telling an operator it has "no downloadable media" hides the only
action that helps: export a sub-folder. These tests pin the distinct code,
message and details for each branch.
"""
from __future__ import annotations

from app.services import zip_export
from tests.helpers import make_folder, upload_video

API = "/api/v1"


def _folder_with_media(client, auth_headers, name: str, count: int = 1) -> dict:
    folder = make_folder(client, auth_headers, name=name)
    for index in range(count):
        upload_video(
            client, auth_headers, folder_id=folder["id"], title=f"unit {index}"
        )
    return folder


def test_exceeding_the_entry_limit_is_a_limit_error(client, auth_headers, monkeypatch):
    folder = _folder_with_media(client, auth_headers, "Huge", count=3)
    monkeypatch.setattr(zip_export, "MAX_ENTRIES", 2)

    resp = client.get(f"{API}/folders/{folder['id']}/download", headers=auth_headers)

    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "zip_too_large"
    assert error["code"] != "nothing_to_export"
    assert "2 entries" in error["message"], error["message"]
    assert error["details"][0]["folder_id"] == folder["id"]
    assert error["details"][0]["limit"] == "2 entries"


def test_exceeding_the_size_limit_is_a_limit_error(client, auth_headers, monkeypatch):
    folder = _folder_with_media(client, auth_headers, "Heavy")
    monkeypatch.setattr(zip_export, "MAX_TOTAL_BYTES", 1)

    resp = client.get(f"{API}/folders/{folder['id']}/download", headers=auth_headers)

    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "zip_too_large"
    assert error["code"] != "nothing_to_export"
    assert "1 byte" in error["message"], error["message"]


def test_the_real_limits_are_named_as_the_contract_states(client, auth_headers):
    """The live constants, not just the monkeypatched ones, match the contract."""
    assert zip_export.MAX_ENTRIES == 5000
    assert zip_export.MAX_TOTAL_BYTES == 2 * 1024 ** 3


def test_an_empty_folder_keeps_its_own_distinct_error(client, auth_headers):
    folder = make_folder(client, auth_headers, name="Empty")

    resp = client.get(f"{API}/folders/{folder['id']}/download", headers=auth_headers)

    assert resp.status_code == 409
    error = resp.json()["error"]
    assert error["code"] == "nothing_to_export"
    assert error["code"] != "zip_too_large"
    assert "no downloadable media" in error["message"]
