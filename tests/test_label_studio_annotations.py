"""Functions for testing label-studio API requests."""

import socket
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import bird_cv.get_label_studio_annotations as lsa


def test_is_port_available_free(monkeypatch):
    """Test that `is_port_available` returns True for a free port.

    This test patches `socket.socket` to simulate a port that can be bound
    successfully, indicating that the port is available.

    Args:
        monkeypatch: pytest fixture to temporarily patch objects.
    """

    # Arrange
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def settimeout(self, t):
            pass

        def bind(self, addr):
            return True

    # Act
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DummySocket())

    # Assert
    assert lsa.is_port_available("localhost", 12345) is True


def test_is_port_available_busy(monkeypatch):
    """Test that `is_port_available` returns False for a busy port.

    This test patches `socket.socket` to simulate a port that raises
    OSError when binding, indicating that the port is already in use.

    Args:
        monkeypatch: pytest fixture to temporarily patch objects.
    """

    # Arrange
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def settimeout(self, t):
            pass

        def bind(self, addr):
            raise OSError("Address already in use")

    # Act
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: DummySocket())

    # Assert
    assert lsa.is_port_available("localhost", 12345) is False


def test_find_open_port(monkeypatch):
    """Test that `find_open_port` returns the first available port.

    This test patches `is_port_available` to simulate the first port being
    unavailable and the second port being available.

    Args:
        monkeypatch: pytest fixture to temporarily patch objects.
    """
    # Arrange
    calls = []

    def fake_is_port_available(host, port):
        calls.append(port)
        return port == 12346

    monkeypatch.setattr(lsa, "is_port_available", fake_is_port_available)

    # Act
    port = lsa.find_open_port(12345, "localhost")

    # Assert
    assert port == 12346
    assert calls == [12345, 12346]


@patch("subprocess.run")
def test_close_server_calls_lsof(mock_run):
    """Test that `close_server` calls lsof and kills processes on the port.

    Args:
        mock_run: Mock for subprocess.run to capture calls.
    """
    # Arrange
    mock_run.return_value.stdout = "123\n456"

    # Act
    lsa.close_server(8080)

    # Assert
    mock_run.assert_any_call(["lsof", "-ti", ":8080"], capture_output=True, text=True)
    mock_run.assert_any_call(["kill", "-9", "123"])
    mock_run.assert_any_call(["kill", "-9", "456"])


@patch("bird_cv.get_label_studio_annotations.time.sleep", return_value=None)
@patch("bird_cv.get_label_studio_annotations.subprocess.Popen")
@patch("bird_cv.get_label_studio_annotations.LabelStudio")
def test_get_label_studio_client_success(mock_label_studio, mock_popen, mock_sleep):
    """Test that `get_label_studio_client` returns a client and verifies connection.

    Args:
        mock_label_studio: Mock for LabelStudio class.
        mock_popen: Mock for subprocess.Popen.
        mock_sleep: Mock for time.sleep to skip delays.
    """
    # Arrange: mock client
    mock_client = MagicMock()
    mock_client.users.whoami.return_value = {"username": "test"}
    mock_label_studio.return_value = mock_client

    # Act
    client = lsa.get_label_studio_client("localhost", 8080, "APIKEY")

    # Assert
    assert client.users.whoami() == {"username": "test"}
    mock_label_studio.assert_called()  # LabelStudio was instantiated
    mock_popen.assert_called()  # subprocess was called
    mock_sleep.assert_not_called()  # should succeed immediately


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


@patch("bird_cv.get_label_studio_annotations.close_server")
@patch("bird_cv.get_label_studio_annotations.export_label_studio_annotations")
@patch("bird_cv.get_label_studio_annotations.get_project_id_from_name", return_value=42)
@patch(
    "bird_cv.get_label_studio_annotations.get_label_studio_client",
    return_value=MagicMock(),
)
@patch("bird_cv.get_label_studio_annotations.find_open_port", return_value=8081)
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
