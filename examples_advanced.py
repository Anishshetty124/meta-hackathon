"""Advanced usage examples for Cloud FinOps environment.

Demonstrates advanced patterns, best practices, and integration strategies
for working with the Cloud FinOps simulator.
"""

import asyncio
import json
from typing import List, Tuple

import requests

from server.environment import CloudEnvironment
from server.models import Action
from server.config import ApplicationConfig
from server.metrics import MetricsCollector
from server.exceptions import ActionExecutionError


# ============================================================================
# Example 1: Basic Environment Usage
# ============================================================================

def example_basic_environment():
    """Demonstrate basic environment initialization and stepping."""
    print("=" * 60)
    print("Example 1: Basic Environment Usage")
    print("=" * 60)
    
    # Create environment with reproducible seed
    env = CloudEnvironment(max_steps=100, random_seed=42)
    env.reset()
    
    # Get initial state
    obs = env._build_observation()
    print(f"\nInitial State:")
    print(f"  Monthly Cost: ${obs.monthly_cost:.2f}")
    print(f"  Resources: {len(obs.resources)}")
    print(f"  Description: {obs.description}\n")
    
    # Execute a simple action
    action = Action(
        command="delete_resource",
        resource_id="vol-unattached-001",
        parameters={},
    )
    
    obs, reward, done, info = env.step(action)
    reward = obs.reward
    print(f"After Delete Unattached Volume:")
    print(f"  Reward: {reward:.2f}")
    print(f"  New Cost: ${obs.monthly_cost:.2f}")
    print(f"  Tasks Completed: {obs.completed_tasks}")


# ============================================================================
# Example 2: Configuration Management
# ============================================================================

def example_configuration_management():
    """Demonstrate configuration loading and validation."""
    print("\n" + "=" * 60)
    print("Example 2: Configuration Management")
    print("=" * 60)
    
    # Load configuration from environment (requires env vars set)
    try:
        config = ApplicationConfig.from_environment()
        
        print(f"\nLoaded Configuration:")
        print(f"  App: {config.app_name} v{config.app_version}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Model: {config.llm.model_name}")
        print(f"  API: {config.api.api_base_url}")
        print(f"  Max Steps: {config.environment.max_steps}")
        
        # Print config as dict (with secrets masked)
        config_dict = config.to_dict()
        print(f"\n  Full Config (safe): {json.dumps(config_dict, indent=2)}")
        
    except Exception as e:
        print(f"  (Configuration requires env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN)")
        print(f"  Error: {e}")


# ============================================================================
# Example 3: Metrics Collection
# ============================================================================

def example_metrics_collection():
    """Demonstrate metrics collection and analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Metrics Collection")
    print("=" * 60)
    
    # Create metrics collector
    collector = MetricsCollector()
    
    # Simulate an episode
    env = CloudEnvironment(max_steps=100, random_seed=42)
    env.reset()
    
    initial_cost = env._build_observation().monthly_cost
    
    # Start episode tracking
    collector.start_episode(episode_number=1)
    
    # Execute multiple actions and record metrics
    actions_to_execute = [
        Action(command="delete_resource", resource_id="vol-unattached-001", parameters={}),
        Action(command="make_private", resource_id="s3-public-bucket", parameters={}),
        Action(command="downsize_instance", resource_id="i-expensive-prod", 
               parameters={"instance_type": "t3.large"}),
    ]
    
    total_reward = 0.0
    for i, action in enumerate(actions_to_execute):
        obs, reward, done, info = env.step(action)
        reward = obs.reward
        total_reward += reward
        
        collector.record_action(
            command=action.command,
            resource_id=action.resource_id,
            success=True,
            reward=reward,
            duration_ms=10.0,  # Mock duration
        )
    
    # End episode
    final_cost = env._build_observation().monthly_cost
    final_obs = env._build_observation()
    
    episode_metrics = collector.end_episode(
        cost_initial=initial_cost,
        cost_final=final_cost,
        tasks_completed=len(final_obs.completed_tasks),
        success=all(final_obs.progress.values()),
        duration_seconds=5.0,
    )
    
    # Display metrics
    if episode_metrics:
        print(f"\nEpisode Metrics:")
        print(f"  Episode: {episode_metrics.episode_number}")
        print(f"  Steps: {episode_metrics.total_steps}")
        print(f"  Total Reward: {episode_metrics.total_reward:.2f}")
        print(f"  Cost Savings: ${episode_metrics.cost_savings:.2f}")
        print(f"  Cost Savings %: {episode_metrics.cost_savings_pct:.1f}%")
        print(f"  Tasks Completed: {episode_metrics.tasks_completed}/3")
        print(f"  Avg Reward/Step: {episode_metrics.avg_reward_per_step:.2f}")
        print(f"  Duration: {episode_metrics.duration_seconds:.1f}s")


# ============================================================================
# Example 4: Error Handling with Custom Exceptions
# ============================================================================

def example_error_handling():
    """Demonstrate robust error handling."""
    print("\n" + "=" * 60)
    print("Example 4: Error Handling with Custom Exceptions")
    print("=" * 60)
    
    env = CloudEnvironment(max_steps=100, random_seed=42)
    env.reset()
    
    # Try various error scenarios
    test_cases = [
        ("Nonexistent Resource", "delete_resource", "no-such-resource"),
        ("Invalid Command", "invalid_command", "vol-unattached-001"),
        ("Valid Action", "delete_resource", "vol-unattached-001"),
    ]
    
    for test_name, command, resource_id in test_cases:
        try:
            action = Action(command=command, resource_id=resource_id, parameters={})
            obs, reward, done, info = env.step(action)
            reward = obs.reward
            print(f"\n{test_name}:")
            print(f"  ✓ Success (reward={reward:.2f})")
        except ActionExecutionError as e:
            print(f"\n{test_name}:")
            print(f"  ✗ Error: {e.error_code} - {e.message}")
        except Exception as e:
            print(f"\n{test_name}:")
            print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")


# ============================================================================
# Example 5: HTTP API Integration
# ============================================================================

def example_http_api_integration():
    """Demonstrate HTTP API usage."""
    print("\n" + "=" * 60)
    print("Example 5: HTTP API Integration (requires running server)")
    print("=" * 60)
    
    api_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.status_code == 200:
            print(f"\n✓ Server is healthy")
            print(f"  {resp.json()}")
        
        # Reset environment
        resp = requests.post(f"{api_url}/reset", json={"seed": None}, timeout=5)
        if resp.status_code == 200:
            initial_obs = resp.json()
            print(f"\n✓ Reset successful")
            print(f"  Initial Cost: ${initial_obs['monthly_cost']:.2f}")
            print(f"  Resources: {len(initial_obs['resources'])}")
        
        # Step with action
        action_data = {
            "action": {
                "command": "delete_resource",
                "resource_id": "vol-unattached-001",
                "parameters": {}
            }
        }
        resp = requests.post(f"{api_url}/step", json=action_data, timeout=5)
        if resp.status_code == 200:
            step_obs = resp.json()
            print(f"\n✓ Step successful")
            print(f"  Reward: {step_obs['reward']:.2f}")
            print(f"  New Cost: ${step_obs['monthly_cost']:.2f}")
        
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to API at {api_url}")
        print(f"  Start server with: python -m uvicorn main:app --reload")
    except Exception as e:
        print(f"\n✗ Error: {e}")


# ============================================================================
# Example 6: Complete Episode Execution
# ============================================================================

def example_complete_episode():
    """Execute a complete episode attempting all three tasks."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Episode Execution")
    print("=" * 60)
    
    env = CloudEnvironment(max_steps=100, random_seed=42)
    env.reset()
    
    # Define optimal action sequence for all tasks
    optimal_actions = [
        ("Easy", "delete_resource", "vol-unattached-001", {}),
        ("Medium", "make_private", "s3-public-bucket", {}),
        ("Hard", "downsize_instance", "i-expensive-prod", {"instance_type": "t3.large"}),
    ]
    
    print(f"\nExecuting optimal action sequence:")
    total_reward = 0.0
    
    for task_name, command, resource_id, params in optimal_actions:
        action = Action(command=command, resource_id=resource_id, parameters=params)
        obs, reward, done, info = env.step(action)
        reward = obs.reward
        total_reward += reward
        
        print(f"\n  [{task_name}] {command} on {resource_id}")
        print(f"    Reward: {reward:+.2f} | Total: {total_reward:.2f}")
        print(f"    Cost: ${obs.monthly_cost:.2f} | Tasks: {len(obs.completed_tasks)}/3")
    
    # Final summary
    final_obs = env._build_observation()
    print(f"\nEpisode Complete:")
    print(f"  Final Reward: {total_reward:.2f} / 1.45 (optimal)")
    print(f"  Final Cost: ${final_obs.monthly_cost:.2f} / $50 (optimal)")
    print(f"  Tasks Completed: {len(final_obs.completed_tasks)}/3")
    print(f"  Episode Done: {final_obs.done}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    """Run all examples."""
    
    example_basic_environment()
    example_configuration_management()
    example_metrics_collection()
    example_error_handling()
    example_complete_episode()
    example_http_api_integration()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
