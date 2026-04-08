"""Deterministic grader for easy task.

Checks whether vol-unattached-001 was deleted from resources.
"""

from typing import Any, Dict


def grade(observation: Dict[str, Any]) -> float:
    resources = observation.get("resources", []) if isinstance(observation, dict) else []
    resource_ids = {
        r.get("resource_id")
        for r in resources
        if isinstance(r, dict)
    }
    return 1.0 if "vol-unattached-001" not in resource_ids else 0.0
