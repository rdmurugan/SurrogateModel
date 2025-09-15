import pytest
import io
from fastapi.testclient import TestClient


def test_list_datasets_empty(client: TestClient, auth_headers):
    response = client.get("/api/v1/datasets/", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_upload_dataset_csv(client: TestClient, auth_headers):
    # Create sample CSV content
    csv_content = """length,width,thickness,stress,displacement
10,5,1,100.5,0.001
20,10,2,200.3,0.002
30,15,3,300.1,0.003"""

    files = {"file": ("test.csv", io.StringIO(csv_content), "text/csv")}
    data = {
        "name": "Test Dataset",
        "description": "A test dataset",
        "input_columns": '["length", "width", "thickness"]',
        "output_columns": '["stress", "displacement"]'
    }

    response = client.post("/api/v1/datasets/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code == 200

    result = response.json()
    assert result["num_samples"] == 3
    assert "data_statistics" in result


def test_upload_dataset_invalid_columns(client: TestClient, auth_headers):
    csv_content = """length,width,stress
10,5,100.5
20,10,200.3"""

    files = {"file": ("test.csv", io.StringIO(csv_content), "text/csv")}
    data = {
        "name": "Test Dataset",
        "description": "A test dataset",
        "input_columns": '["length", "width", "thickness"]',  # thickness doesn't exist
        "output_columns": '["stress"]'
    }

    response = client.post("/api/v1/datasets/upload", files=files, data=data, headers=auth_headers)
    assert response.status_code == 400


def test_get_dataset(client: TestClient, auth_headers, db_session, test_tenant):
    from app.models.dataset import Dataset

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

    response = client.get(f"/api/v1/datasets/{dataset.id}", headers=auth_headers)
    assert response.status_code == 200

    result = response.json()
    assert result["name"] == "Test Dataset"
    assert result["num_samples"] == 100