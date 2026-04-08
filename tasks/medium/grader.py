"""Deterministic grader for medium task.

Checks whether s3-public-bucket is no longer public.
"""

from typing import Any, Dict


def grade(observation: Dict[str, Any]) -> float:
    resources = observation.get("resources", []) if isinstance(observation, dict) else []
    for resource in resources:
        if isinstance(resource, dict) and resource.get("resource_id") == "s3-public-bucket":
            return 1.0 if resource.get("is_public") is False else 0.0
    return 0.0
