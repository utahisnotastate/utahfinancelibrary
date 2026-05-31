"""Immutable protocol constants for the Utah Finance Library."""

# Hardcoded sovereign protocol tithe — must not change without governance revision.
SOVEREIGN_PROTOCOL_TITHE: float = 0.023

# Default humanitarian abundance allocation (5.7%); override per deployment.
DEFAULT_HUMANITARIAN_RATE: float = 0.057

# Humanitarian rate used by the "universal tithe" routing helper (10.0%).
# Like every rate here it is a documented, overridable parameter — not a hidden
# or non-removable extraction. See ``protocol_economics.enforce_universal_tithe``.
UNIVERSAL_HUMANITARIAN_RATE: float = 0.10
