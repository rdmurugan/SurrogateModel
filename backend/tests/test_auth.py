import pytest
from fastapi.testclient import TestClient


def test_login_success(client: TestClient, test_user):
    response = client.post("/api/v1/auth/login", data={
        "username": test_user.email,
        "password": "testpass"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == test_user.email


def test_login_invalid_password(client: TestClient, test_user):
    response = client.post("/api/v1/auth/login", data={
        "username": test_user.email,
        "password": "wrongpassword"
    })
    assert response.status_code == 401


def test_login_invalid_user(client: TestClient):
    response = client.post("/api/v1/auth/login", data={
        "username": "nonexistent@example.com",
        "password": "password"
    })
    assert response.status_code == 401


def test_logout(client: TestClient, auth_headers):
    response = client.post("/api/v1/auth/logout", headers=auth_headers)
    assert response.status_code == 200