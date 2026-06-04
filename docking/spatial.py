"""Spatial-neighbor helpers with an optional SciPy acceleration path."""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import cKDTree as cKDTree  # type: ignore
except ImportError:

    class cKDTree:
        """Small NumPy fallback implementing the cKDTree methods used here."""

        def __init__(self, data):
            self.data = np.asarray(data, dtype=float)
            if self.data.ndim != 2:
                raise ValueError("Spatial index data must be a 2D coordinate array")

        def query_ball_point(self, points, radius):
            query = np.asarray(points, dtype=float)
            scalar = query.ndim == 1
            query = np.atleast_2d(query)
            result = []
            for point in query:
                distances = np.linalg.norm(self.data - point, axis=1)
                result.append(np.flatnonzero(distances <= radius).tolist())
            return result[0] if scalar else result

        def query(self, points, k=1):
            query = np.asarray(points, dtype=float)
            scalar = query.ndim == 1
            query = np.atleast_2d(query)
            k = max(1, min(int(k), len(self.data)))
            all_distances = []
            all_indices = []
            for point in query:
                distances = np.linalg.norm(self.data - point, axis=1)
                indices = np.argsort(distances)[:k]
                all_distances.append(distances[indices])
                all_indices.append(indices)
            distances = np.asarray(all_distances)
            indices = np.asarray(all_indices)
            if k == 1:
                distances = distances[:, 0]
                indices = indices[:, 0]
            if scalar:
                return distances[0], indices[0]
            return distances, indices
