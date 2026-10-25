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


def init_job_ray() -> None:
    """Start a private Ray instance sized to this SLURM job, if none is running.

    Ray otherwise sizes itself to the whole node (all its cores), starting
    far more worker processes than the job's CPUs, and a fixed-port
    ``ray start --head`` collides with other jobs on the node. A local
    instance on free ports, with the job's CPU count and a small object
    store, avoids both; it shuts down with the process.
    """
    import os

    import ray

    if not ray.is_initialized():
        ray.init(
            address="local",
            num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
            include_dashboard=False,
            # Tuning passes almost no data through Ray; a small object store
            # keeps two jobs on one node within shared memory.
            object_store_memory=2 * 1024**3,
        )
