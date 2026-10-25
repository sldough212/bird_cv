import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for serializing NumPy arrays."""

    def default(self, obj):
        """Converts unsupported objects into JSON-serializable formats.

        Args:
            obj (Any): The object to serialize.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def extract_camera_video(video_str: str) -> tuple[str, str]:
    """Extract camera ID and video ID from a video path string.

    Args:
        video_str: Path to a video file where the parent directory name is the
            URL-encoded camera ID and the filename is the video ID.

    Returns:
        A tuple of (camera_id, video_id) where camera_id has URL encoding
        decoded (e.g. ``%2C`` → ``,``) and video_id is the filename.
    """
    video_path = Path(video_str)
    video_id = video_path.name
    camera_id = video_path.parent.stem
    camera_id = camera_id.replace("%2C", ",")
    return camera_id, video_id
