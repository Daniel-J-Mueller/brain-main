"""Global arousal controller."""


class ReticularActivatingSystem:
    """Toggle arousal based on salience."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.arousal = 1.0

    def update(self, salience: float) -> None:
        self.arousal = 1.0 if salience > self.threshold else 0.0

    def is_awake(self) -> bool:
        return self.arousal > 0.0
