"""Abstract base class for video quality scorers."""

from abc import ABC, abstractmethod

import numpy as np


class BaseScorer(ABC):
    """
    Interface for video quality scorers.

    Each scorer is used as a Ray map_batches callable class:
      - __init__ loads model weights (runs once per Ray actor)
      - __call__ processes a batch dict and returns it with new score columns

    The batch dict has at minimum:
      - "video_path": np.ndarray of str
      - "category": np.ndarray of str
    Scorers add their own column(s) and return the dict.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Load any model weights. Called once per Ray actor."""
        ...

    @abstractmethod
    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Score a batch of videos. Add score column(s) and return the dict."""
        ...

    #: Names of columns this scorer adds to the batch. Subclasses must set this.
    score_columns: list[str]
