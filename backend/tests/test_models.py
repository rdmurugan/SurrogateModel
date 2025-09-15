import pytest
from fastapi.testclient import TestClient


def test_list_models_empty(client: TestClient, auth_headers):
    response = client.get("/api/v1/models/", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_create_model(client: TestClient, auth_headers, db_session, test_tenant):
    from app.models.dataset import Dataset

    # Create a test dataset first
    dataset = Dataset(
        name="Test Dataset",
        file_path="test/path",
        input_columns=["x", "y"],
        output_columns=["z"],
        num_samples=100,
        tenant_id=test_tenant.id
    )
    db_session.add(dataset)
    db_session.commit()

    model_data = {
        "name": "Test Model",
        "description": "A test surrogate model",
        "dataset_id": dataset.id,
        "algorithm": "gaussian_process",
        "hyperparameters": {"kernel": "rbf"}
    }

    response = client.post("/api/v1/models/", json=model_data, headers=auth_headers)
    assert response.status_code == 200

    result = response.json()
    assert "id" in result
    assert result["status"] == "pending"


def test_create_model_invalid_dataset(client: TestClient, auth_headers):
    model_data = {
        "name": "Test Model",
        "dataset_id": 999,  # Non-existent dataset
        "algorithm": "gaussian_process"
    }

    response = client.post("/api/v1/models/", json=model_data, headers=auth_headers)
    assert response.status_code == 404


def test_get_model(client: TestClient, auth_headers, db_session, test_tenant):
    from app.models.dataset import Dataset
    from app.models.surrogate_model import SurrogateModel

    # Create dataset and model
    dataset = Dataset(
        name="Test Dataset",
        file_path="test/path",
        input_columns=["x", "y"],
        output_columns=["z"],
        num_samples=100,
        tenant_id=test_tenant.id
    )
    db_session.add(dataset)
    db_session.commit()

    model = SurrogateModel(
        name="Test Model",
        algorithm="gaussian_process",
        dataset_id=dataset.id,
        tenant_id=test_tenant.id
    )
    db_session.add(model)
    db_session.commit()

    response = client.get(f"/api/v1/models/{model.id}", headers=auth_headers)
    assert response.status_code == 200

    result = response.json()
    assert result["name"] == "Test Model"
    assert result["algorithm"] == "gaussian_process"