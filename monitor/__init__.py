"""Driver Monitoring System package."""

BEHAVIORS = ["Normal", "Drowsy", "Distracted", "Using Phone", "Eating", "Drinking"]
RISKY = set(BEHAVIORS) - {"Normal"}
