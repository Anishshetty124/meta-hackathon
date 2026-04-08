"""Local checker for filesystem-based task grader detection.

Validates at least 3 tasks exist and each has a grader script.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tasks" / "tasks_manifest.json"


def main() -> int:
    if not MANIFEST.exists():
        print("FAIL: tasks_manifest.json missing")
        return 1

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or len(tasks) < 3:
        print("FAIL: fewer than 3 tasks in manifest")
        return 1

    ok = True
    for task in tasks:
        name = task.get("name")
        has_grader = bool(task.get("has_grader"))
        grader_path = task.get("grader_path")
        grader_file = ROOT / grader_path if isinstance(grader_path, str) else None

        if not has_grader:
            print(f"FAIL: task {name} has_grader=false")
            ok = False
            continue

        if grader_file is None or not grader_file.exists():
            print(f"FAIL: task {name} grader file missing: {grader_path}")
            ok = False
            continue

        print(f"PASS: task {name} grader found at {grader_path}")

    if ok:
        print("PASS: 3+ tasks with graders detected")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
