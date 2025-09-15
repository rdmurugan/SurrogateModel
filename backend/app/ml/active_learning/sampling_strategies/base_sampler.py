from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseSampler(ABC):
    """
    Abstract base class for all sampling strategies in active learning.

    This class defines the interface that all sampling strategies must implement,
    ensuring consistency across different sampling approaches.
    """

    def __init__(self):
        """Initialize the base sampler."""
        self.history = []
        self.metadata = {}

    @abstractmethod
    def sample(self,
               model,
               X_candidates: np.ndarray,
               n_samples: int = 1,
               acquisition_function=None,
               **kwargs) -> Dict[str, Any]:
        """
        Sample points from candidates using the specific strategy.

        Args:
            model: The trained surrogate model
            X_candidates: Array of candidate points to sample from
            n_samples: Number of points to sample
            acquisition_function: Optional acquisition function to guide sampling
            **kwargs: Additional strategy-specific parameters

        Returns:
            Dictionary containing:
                - selected_points: The selected sample points
                - selected_indices: Indices of selected points in X_candidates
                - Additional strategy-specific information
        """
        pass

    def update_history(self, sample_result: Dict[str, Any]) -> None:
        """
        Update the sampling history with the latest results.

        Args:
            sample_result: Result dictionary from the sample method
        """
        self.history.append({
            'iteration': len(self.history),
            'n_samples': len(sample_result.get('selected_points', [])),
            'sample_result': sample_result
        })

    def get_history(self) -> list:
        """Get the complete sampling history."""
        return self.history

    def reset(self) -> None:
        """Reset the sampler state."""
        self.history = []
        self.metadata = {}

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the sampler."""
        self.metadata[key] = value

    def get_metadata(self, key: str = None) -> Any:
        """Get metadata from the sampler."""
        if key is None:
            return self.metadata
        return self.metadata.get(key)