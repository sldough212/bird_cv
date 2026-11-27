"""Tests for exporting Label Studio annotations."""

from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import bird_cv.client.get_label_studio_annotations as lsa


def test_get_project_id_from_name_found():
    """Test that `get_project_id_from_name` returns the correct ID if the project exists."""
    # Arrange
    project_mock = MagicMock()
    project_mock.title = "Test Project"
    project_mock.id = 42
    client = MagicMock()
    client.projects.list.return_value = [project_mock]

    # Act
    project_id = lsa.get_project_id_from_name(client, "Test Project")

    # Assert
    assert project_id == 42


def test_get_project_id_from_name_not_found():
    """Test that `get_project_id_from_name` raises ValueError if the project does not exist."""
    # Arrange
    client = MagicMock()
    client.projects.list.return_value = []

    # Act & Assert
    with pytest.raises(ValueError):
        lsa.get_project_id_from_name(client, "Nonexistent")


def test_export_label_studio_annotations(monkeypatch, tmp_path):
    """Test that `export_label_studio_annotations` writes the correct bytes to disk.

    Args:
        monkeypatch: pytest fixture to temporarily patch objects.
        tmp_path: pytest fixture providing a temporary directory.
    """
    # Arrange
    mock_client = MagicMock()

    # Export snapshot mock
    export_mock = MagicMock()
    export_mock.id = 1
    mock_client.projects.exports.create.return_value = export_mock

    # Job status mock
    job_mock = MagicMock()
    job_mock.status = "completed"
    mock_client.projects.exports.get.return_value = job_mock

    # Fake download generator
    def fake_download(id, export_pk, export_type, request_options):
        yield b"chunk1"
        yield b"chunk2"

    mock_client.projects.exports.download = fake_download

    out_path = tmp_path / "annotations.json"

    # Act
    lsa.export_label_studio_annotations(mock_client, 123, out_path)

    # Assert
    content = out_path.read_bytes()
    assert content == b"chunk1chunk2"


@patch("bird_cv.client.get_label_studio_annotations.close_server")
@patch("bird_cv.client.get_label_studio_annotations.export_label_studio_annotations")
@patch(
    "bird_cv.client.get_label_studio_annotations.get_project_id_from_name",
    return_value=42,
)
@patch(
    "bird_cv.client.get_label_studio_annotations.get_label_studio_client",
    return_value=MagicMock(),
)
@patch(
    "bird_cv.client.get_label_studio_annotations.find_open_port",
    return_value=8081,
)
def test_get_label_studio_annotations(
    mock_port, mock_client, mock_project_id, mock_export, mock_close
):
    """Test that `get_label_studio_annotations` performs the full flow successfully.

    Args:
        mock_port: Mock for find_open_port.
        mock_client: Mock for get_label_studio_client.
        mock_project_id: Mock for get_project_id_from_name.
        mock_export: Mock for export_label_studio_annotations.
        mock_close: Mock for close_server.
    """
    out_path = Path("/tmp/output.json")
    lsa.get_label_studio_annotations(
        host="localhost",
        port=8080,
        api_key="APIKEY",
        project_name="Test Project",
        output_path=out_path,
    )

    mock_port.assert_called_once()
    mock_client.assert_called_once()
    mock_project_id.assert_called_once()
    mock_export.assert_called_once()
    mock_close.assert_called_once()
