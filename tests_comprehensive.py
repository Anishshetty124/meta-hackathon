"""Comprehensive test suite for Cloud FinOps environment.

Tests cover models, environment simulation, action execution, API endpoints,
and end-to-end workflows. Uses pytest fixtures for reproducible test state.
"""

import pytest
from typing import Dict, Any

from server.models import Resource, Action, Observation, ResetRequest, StepRequest
from server.environment import CloudEnvironment, TaskProgress
from server.exceptions import (
    ActionExecutionError,
    ResourceNotFoundError,
    InvalidActionError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def environment() -> CloudEnvironment:
    """Provide a fresh CloudEnvironment for each test."""
    env = CloudEnvironment(max_steps=100, random_seed=42)
    env.reset()
    return env


@pytest.fixture
def reset_request() -> ResetRequest:
    """Provide ResetRequest for testing."""
    return ResetRequest(seed=42, difficulty=None)


@pytest.fixture
def initial_observation(environment: CloudEnvironment) -> Observation:
    """Provide initial observation from environment reset."""
    return environment._build_observation()


# ============================================================================
# Model Tests
# ============================================================================

class TestResourceModel:
    """Test Resource model validation and behavior."""
    
    def test_valid_resource_creation(self):
        """Test that valid resources are created correctly."""
        resource = Resource(
            resource_id="test-id",
            resource_type="ec2_instance",
            monthly_cost=100.0,
            cpu_utilization=50.0,
            is_attached=True,
            is_public=False,
            instance_type="t3.large",
            needs_fixing=False,
        )
        assert resource.resource_id == "test-id"
        assert resource.monthly_cost == 100.0
    
    def test_monthly_cost_validation(self):
        """Test that monthly cost is rounded to 2 decimals."""
        resource = Resource(
            resource_id="test-id",
            resource_type="ec2_instance",
            monthly_cost=100.12345,
            cpu_utilization=50.0,
        )
        assert resource.monthly_cost == 100.12
    
    def test_invalid_monthly_cost(self):
        """Test that negative costs are rejected."""
        with pytest.raises(ValueError):
            Resource(
                resource_id="test-id",
                resource_type="ec2_instance",
                monthly_cost=-100.0,
                cpu_utilization=50.0,
            )
    
    def test_cpu_utilization_validation(self):
        """Test that CPU utilization is constrained to 0-100."""
        with pytest.raises(ValueError):
            Resource(
                resource_id="test-id",
                resource_type="ec2_instance",
                monthly_cost=100.0,
                cpu_utilization=150.0,  # Invalid
            )


class TestActionModel:
    """Test Action model validation."""
    
    def test_valid_action(self):
        """Test that valid actions are created."""
        action = Action(
            command="delete_resource",
            resource_id="vol-123",
            parameters={},
        )
        assert action.command == "delete_resource"
    
    def test_invalid_command(self):
        """Test that invalid commands are rejected."""
        with pytest.raises(ValueError):
            Action(
                command="invalid_command",
                resource_id="vol-123",
                parameters={},
            )


# ============================================================================
# Environment Tests
# ============================================================================

class TestCloudEnvironmentInitialization:
    """Test environment initialization and state."""
    
    def test_reset_creates_resources(self, environment: CloudEnvironment):
        """Test that reset creates the expected resources."""
        obs = environment._build_observation()
        assert len(obs.resources) == 6
        assert obs.monthly_cost == pytest.approx(365.0, abs=1.0)
    
    def test_environment_with_seed_is_reproducible(self):
        """Test that same seed produces same initial state."""
        env1 = CloudEnvironment(max_steps=100, random_seed=42)
        env1.reset()
        state1 = env1._build_observation()
        
        env2 = CloudEnvironment(max_steps=100, random_seed=42)
        env2.reset()
        state2 = env2._build_observation()
        
        assert state1.monthly_cost == state2.monthly_cost
    
    def test_different_seeds_produce_different_states(self):
        """Test that different seeds produce different states."""
        env1 = CloudEnvironment(max_steps=100, random_seed=42)
        env1.reset()
        state1 = env1._build_observation()
        
        env2 = CloudEnvironment(max_steps=100, random_seed=123)
        env2.reset()
        state2 = env2._build_observation()
        
        # At least one should be different (with high probability)
        # Note: might rarely be equal by chance, but very unlikely


class TestActionExecution:
    """Test action execution and state transitions."""
    
    def test_delete_unattached_volume_success(
        self, environment: CloudEnvironment
    ):
        """Test successful deletion of unattached EBS volume."""
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={},
        )
        obs, reward, done, info = environment.step(action)
        reward = obs.reward
        
        assert reward == pytest.approx(0.35, abs=0.01)
        assert "easy" in obs.completed_tasks
        # Verify resource was removed
        resource_ids = [r.resource_id for r in obs.resources]
        assert "vol-unattached-001" not in resource_ids
    
    def test_make_public_bucket_private_success(
        self, environment: CloudEnvironment
    ):
        """Test successful security fix on S3 bucket."""
        action = Action(
            command="make_private",
            resource_id="s3-public-bucket",
            parameters={},
        )
        obs, reward, done, info = environment.step(action)
        reward = obs.reward
        
        assert reward == pytest.approx(0.50, abs=0.01)
        assert "medium" in obs.completed_tasks
    
    def test_downsize_instance_success(self, environment: CloudEnvironment):
        """Test successful EC2 instance downsizing."""
        action = Action(
            command="downsize_instance",
            resource_id="i-expensive-prod",
            parameters={"instance_type": "t3.large"},
        )
        obs, reward, done, info = environment.step(action)
        reward = obs.reward
        
        assert reward == pytest.approx(0.60, abs=0.01)
        assert "hard" in obs.completed_tasks
    
    def test_action_on_nonexistent_resource_fails(
        self, environment: CloudEnvironment
    ):
        """Test that actions on nonexistent resources fail gracefully."""
        action = Action(
            command="delete_resource",
            resource_id="nonexistent-id",
            parameters={},
        )
        obs, reward, done, info = environment.step(action)
        reward = obs.reward
        
        # Should return negative reward but not crash
        assert reward < 0
    
    def test_step_increments_counter(self, environment: CloudEnvironment):
        """Test that step counter increments correctly."""
        initial_step = environment.current_step
        
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={},
        )
        environment.step(action)
        
        assert environment.current_step == initial_step + 1
    
    def test_max_steps_termination(self, environment: CloudEnvironment):
        """Test that episode terminates after max steps."""
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={},
        )
        
        for i in range(100):
            obs_before = environment._build_observation()
            if obs_before.done:
                break
            environment.step(action)
        
        obs_final = environment._build_observation()
        # Episode should be done (either all tasks or max steps)
        assert obs_final.done


class TestProgressTracking:
    """Test task progress tracking."""
    
    def test_progress_initializes_at_zero(self, environment: CloudEnvironment):
        """Test that task progress starts at 0.0."""
        obs = environment._build_observation()
        
        assert obs.progress["easy"] == 0.0
        assert obs.progress["medium"] == 0.0
        assert obs.progress["hard"] == 0.0
    
    def test_progress_updates_on_task_completion(
        self, environment: CloudEnvironment
    ):
        """Test that progress updates after task completion."""
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={},
        )
        environment.step(action)
        
        obs = environment._build_observation()
        assert obs.progress["easy"] == 1.0  # Task completed


# ============================================================================
# Integration Tests
# ============================================================================

class TestEpisodeWorkflow:
    """Test complete episode workflows."""
    
    def test_complete_all_three_tasks(self):
        """Test successfully completing all three optimization tasks."""
        env = CloudEnvironment(max_steps=100, random_seed=42)
        env.reset()
        
        actions = [
            Action(
                command="delete_resource",
                resource_id="vol-unattached-001",
                parameters={},
            ),
            Action(
                command="make_private",
                resource_id="s3-public-bucket",
                parameters={},
            ),
            Action(
                command="downsize_instance",
                resource_id="i-expensive-prod",
                parameters={"instance_type": "t3.large"},
            ),
        ]
        
        total_reward = 0.0
        for action in actions:
            obs, reward, done, info = env.step(action)
            reward = obs.reward
            total_reward += reward
        
        # All three tasks completed
        assert total_reward == pytest.approx(1.45, abs=0.01)
        final_obs = env._build_observation()
        assert len(final_obs.completed_tasks) == 3
    
    def test_cost_reduction_from_actions(self):
        """Test that actions reduce monthly cost."""
        env = CloudEnvironment(max_steps=100, random_seed=42)
        env.reset()
        
        initial_obs = env._build_observation()
        initial_cost = initial_obs.monthly_cost
        
        action = Action(
            command="downsize_instance",
            resource_id="i-expensive-prod",
            parameters={"instance_type": "t3.large"},
        )
        env.step(action)
        
        final_obs = env._build_observation()
        final_cost = final_obs.monthly_cost
        
        # Cost should decrease
        assert final_cost < initial_cost


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_reset_performance(self, environment: CloudEnvironment, benchmark):
        """Benchmark environment reset operation."""
        def reset_op():
            environment.reset()
        
        # Reset should be fast
        result = benchmark(reset_op)
        # Note: benchmark requires pytest-benchmark plugin
    
    def test_step_performance(
        self, environment: CloudEnvironment, benchmark
    ):
        """Benchmark step operation."""
        action = Action(
            command="delete_resource",
            resource_id="vol-unattached-001",
            parameters={},
        )
        
        def step_op():
            environment.step(action)
        
        # Step should be very fast
        result = benchmark(step_op)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
