"""brain brain modules."""

from __future__ import annotations

__all__ = ["main"]

from .subthalamic_nucleus import SubthalamicNucleus
from .supplementary_motor_area import SupplementaryMotorArea


def main() -> None:
    """Entry point for running the integrated brain."""
    from .brain import main as run

    run()
