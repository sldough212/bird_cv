import json
import numpy as np


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
