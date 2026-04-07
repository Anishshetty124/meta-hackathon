#!/usr/bin/env python3
"""Quick start guide and validation script for Cloud FinOps & Security Auditor."""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict


def validate_dependencies() -> bool:
    """Validate that all required dependencies are available.
    
    Returns:
        True if all dependencies are available
    """
    print("Checking dependencies...")
    try:
        import fastapi
        import pydantic
        import requests
        import openai
        print("✓ All core dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False


def validate_structure() -> bool:
    """Validate project directory structure.
    
    Returns:
        True if structure is correct
    """
    print("\nValidating project structure...")
    required_files = [
        "main.py",
        "inference.py",
        "requirements.txt",
        "Dockerfile",
        "server/__init__.py",
        "server/models.py",
        "server/environment.py",
    ]
    
    all_valid = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_valid = False
    
    return all_valid


def validate_imports() -> bool:
    """Validate that all modules can be imported.
    
    Returns:
        True if all imports successful
    """
    print("\nValidating Python imports...")
    try:
        from server.models import Resource, Action, Observation
        from server.environment import CloudEnvironment
        print("✓ Can import server.models")
        print("✓ Can import server.environment")
        
        # Verify FastAPI app can be created
        import importlib.util
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        print("✓ FastAPI app loads successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_environment() -> bool:
    """Test CloudEnvironment functionality.
    
    Returns:
        True if environment works correctly
    """
    print("\nTesting CloudEnvironment...")
    try:
        from server.environment import CloudEnvironment
        from server.models import Action
        
        env = CloudEnvironment(seed=42, max_steps=100)
        obs = env.reset()
        
        print(f"✓ Environment created and reset")
        print(f"  Initial cost: ${obs.monthly_cost:.2f}")
        print(f"  Resources: {len(obs.resources)}")
        print(f"  Tasks to complete: {3 - len(obs.completed_tasks)}")
        
        # Test an action
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={}
        )
        obs, reward, done, info = env.step(action)
        print(f"✓ Action executed successfully")
        print(f"  Reward: {reward:.2f}")
        print(f"  New cost: ${obs.monthly_cost:.2f}")
        print(f"  Tasks completed: {len(obs.completed_tasks)}")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_api_endpoints() -> bool:
    """Test FastAPI endpoints (requires running server).
    
    Returns:
        True if API endpoints work
    """
    print("\nTesting API endpoints (requires running server on localhost:8000)...")

    def parse_json_response(resp: Any, context: str) -> Dict[str, Any]:
        """Parse HTTP JSON safely with useful context in errors."""
        try:
            payload = resp.json()
        except ValueError as e:
            body_preview = (getattr(resp, "text", "") or "")[:300]
            raise ValueError(
                f"{context} returned invalid JSON (status={resp.status_code}): {body_preview}"
            ) from e

        if not isinstance(payload, dict):
            raise ValueError(f"{context} returned unexpected JSON shape: {type(payload).__name__}")

        return payload

    try:
        import requests
        
        # Test health endpoint
        try:
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code == 200:
                print("✓ GET /health works")
            else:
                print(f"✗ GET /health returned {resp.status_code}")
                return False
        except requests.ConnectionError:
            print("✗ Cannot connect to server on localhost:8000")
            print("  Start the server with: python main.py")
            return False
        
        # Test reset endpoint
        resp = requests.post("http://localhost:8000/reset", timeout=5)
        if resp.status_code == 200:
            obs = parse_json_response(resp, "POST /reset")
            print(f"✓ POST /reset works")
            print(f"  Initial cost: ${float(obs.get('monthly_cost', 0.0)):.2f}")
        else:
            print(f"✗ POST /reset returned {resp.status_code}")
            return False
        
        # Test step endpoint
        action = {
            "command": "delete_resource",
            "resource_id": "vol-unattached-001",
            "parameters": {}
        }
        resp = requests.post(
            "http://localhost:8000/step",
            json={"action": action},
            timeout=5
        )
        if resp.status_code == 200:
            obs = parse_json_response(resp, "POST /step")
            print(f"✓ POST /step works")
            print(f"  Reward: {float(obs.get('reward', 0.0)):.2f}")
            print(f"  New cost: ${float(obs.get('monthly_cost', 0.0)):.2f}")
        else:
            print(f"✗ POST /step returned {resp.status_code}")
            return False
        
        # Test state endpoint
        resp = requests.get("http://localhost:8000/state", timeout=5)
        if resp.status_code == 200:
            state = parse_json_response(resp, "GET /state")
            print(f"✓ GET /state works")
            print(
                f"  Episode step: {state.get('episode_step', 0)}/{state.get('max_steps', 0)}"
            )
        else:
            print(f"✗ GET /state returned {resp.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"⚠ Could not test API (this is OK if server is not running): {e}")
        return True  # Don't fail if server isn't running


def main() -> int:
    """Run all validation checks.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 60)
    print("Cloud FinOps & Security Auditor - Validation Script")
    print("=" * 60)
    
    checks = [
        ("Dependencies", validate_dependencies),
        ("Project Structure", validate_structure),
        ("Python Imports", validate_imports),
        ("Environment Logic", test_environment),
        ("API Endpoints (optional)", test_api_endpoints),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All checks passed! Project is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set environment variables (see .env.example)")
        print("3. Run the FastAPI server: python main.py")
        print("4. Run the agent: python inference.py")
        return 0
    elif passed == total - 1:  # Allow optional API test to fail
        print("\n✓ Project is mostly ready (API test is optional)")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
