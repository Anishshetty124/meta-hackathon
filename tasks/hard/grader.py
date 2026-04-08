"""Deterministic grader for hard task.

Checks whether i-expensive-prod was right-sized to t3.large.
"""

from typing import Any, Dict


def grade(observation: Dict[str, Any]) -> float:
    resources = observation.get("resources", []) if isinstance(observation, dict) else []
    for resource in resources:
        if isinstance(resource, dict) and resource.get("resource_id") == "i-expensive-prod":
            return 1.0 if resource.get("instance_type") == "t3.large" else 0.0
    return 0.0
