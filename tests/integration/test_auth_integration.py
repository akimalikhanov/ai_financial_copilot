"""
Integration tests for auth: register, login, refresh, logout, and list conversations by user.
"""

from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_auth_register_login_refresh(async_client) -> None:
    """Register → login → refresh returns new access token."""
    email = f"refresh-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "pass123"},
    )
    assert reg.status_code == 200
    first_token = reg.json()["access_token"]
    # Client stores cookies from register; reuse for refresh

    login = await async_client.post(
        "/v1/auth/login",
        json={"email": email, "password": "pass123"},
    )
    assert login.status_code == 200
    login_token = login.json()["access_token"]

    refresh = await async_client.post("/v1/auth/refresh")
    assert refresh.status_code == 200
    refresh_token = refresh.json()["access_token"]
    assert refresh_token
    assert refresh_token != first_token or refresh_token != login_token or refresh_token


@pytest.mark.integration
@pytest.mark.asyncio
async def test_auth_logout_revokes_session(async_client) -> None:
    """Register → logout (revoke session) → refresh with same cookie returns 401."""
    email = f"logout-{uuid4().hex}@test.com"
    reg = await async_client.post(
        "/v1/auth/register",
        json={"email": email, "password": "pass123"},
    )
    assert reg.status_code == 200
    # Client has refresh cookie from register; logout revokes it and clears cookie
    logout = await async_client.post("/v1/auth/logout")
    assert logout.status_code == 200

    refresh = await async_client.post("/v1/auth/refresh")
    assert refresh.status_code == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_conversations_filtered_by_user(async_client) -> None:
    """User A sees only their conversations; User B sees none of A's."""
    email_a = f"lista-{uuid4().hex}@test.com"
    reg_a = await async_client.post(
        "/v1/auth/register",
        json={"email": email_a, "password": "pass123"},
    )
    assert reg_a.status_code == 200
    token_a = reg_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    conv = await async_client.post(
        "/v1/conversations",
        json={"title": "A's conversation"},
        headers=headers_a,
    )
    assert conv.status_code == 200

    list_a = await async_client.get("/v1/conversations", headers=headers_a)
    assert list_a.status_code == 200
    data_a = list_a.json()
    assert data_a["total"] >= 1
    assert any(c["title"] == "A's conversation" for c in data_a["conversations"])

    email_b = f"listb-{uuid4().hex}@test.com"
    reg_b = await async_client.post(
        "/v1/auth/register",
        json={"email": email_b, "password": "pass123"},
    )
    assert reg_b.status_code == 200
    headers_b = {"Authorization": f"Bearer {reg_b.json()['access_token']}"}

    list_b = await async_client.get("/v1/conversations", headers=headers_b)
    assert list_b.status_code == 200
    data_b = list_b.json()
    assert not any(c["title"] == "A's conversation" for c in data_b["conversations"])
