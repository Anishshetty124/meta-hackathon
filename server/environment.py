"""Cloud FinOps & Security Auditor environment simulation.

Implements a realistic cloud infrastructure management simulator where agents
learn to optimize costs and fix security issues. Provides realistic financial
models and task completion mechanics.
"""

import logging
import random
from dataclasses import dataclass, field as dataclass_field
from typing import Dict, List, Tuple, Optional, Any

from server.models import (
    Resource,
    ResourceType,
    TaskType,
    Action,
    Observation,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskProgress:
    """Tracks progress towards completing a specific optimization task.

    Attributes:
        task_type: Category of this task (easy/medium/hard)
        progress: Fractional completion between 0.0 and 1.0
        completed: Whether task has reached full completion
        description: Human-readable task description
    """
    task_type: TaskType
    progress: float = 0.0
    completed: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Validate task progress on initialization."""
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"Task progress must be 0.0-1.0, got {self.progress}")



class CloudEnvironment:
    """Simulates cloud infrastructure management for cost and security optimization.

    Manages a mock cloud environment with EC2 instances, S3 buckets, and EBS volumes.
    Agents can delete resources, fix security issues, and downsize instances.
    Provides cost tracking and rewards for task completion.
    """

    # Static configuration for instance pricing
    INSTANCE_TYPE_COSTS: Dict[str, float] = {
        "t3.micro": 8.0,
        "t3.small": 15.0,
        "t3.medium": 30.0,
        "t3.large": 65.0,
        "t3.xlarge": 130.0,
        "m5.large": 96.0,
        "m5.xlarge": 192.0,
        "m5.2xlarge": 180.0,
    }

    # Task reward structure
    TASK_REWARDS: Dict[TaskType, float] = {
        TaskType.EASY: 0.35,
        TaskType.MEDIUM: 0.50,
        TaskType.HARD: 0.60,
    }

    def __init__(self, seed: Optional[int] = None, max_steps: int = 100) -> None:
        """Initialize the cloud environment.

        Args:
            seed: Random seed for reproducible episode generation
            max_steps: Maximum steps allowed before episode termination

        Raises:
            ValueError: If max_steps is negative
        """
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")

        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
            logger.debug(f"Environment initialized with seed {seed}")

        self.max_steps = max_steps
        self.current_step = 0
        self.resources: Dict[str, Resource] = {}
        self.initial_cost = 0.0
        self.action_history: List[Tuple[str, str]] = []
        self.last_action_signature: Optional[Tuple[str, str]] = None
        self.repeat_action_streak = 0

        # Initialize task tracking
        self.task_progress: Dict[TaskType, TaskProgress] = {
            TaskType.EASY: TaskProgress(
                TaskType.EASY,
                description="Delete an unattached disk"
            ),
            TaskType.MEDIUM: TaskProgress(
                TaskType.MEDIUM,
                description="Fix a public S3 bucket"
            ),
            TaskType.HARD: TaskProgress(
                TaskType.HARD,
                description="Downsize expensive EC2 instance"
            ),
        }

        self.reset()

    def reset(self) -> Observation:
        """Reset environment to initial state.

        Clears all resources, resets step counter, and reinitializes task tracking.
        Should be called at the start of each episode.

        Returns:
            Initial observation of the environment
        """
        self.current_step = 0
        self.resources = self._initialize_resources()
        self.initial_cost = self._calculate_total_cost()
        self.action_history = []
        self.last_action_signature = None
        self.repeat_action_streak = 0

        # Reset task progress
        for task_type in TaskType:
            self.task_progress[task_type].progress = 0.0
            self.task_progress[task_type].completed = False

        logger.info(
            f"Environment reset with {len(self.resources)} resources, "
            f"initial cost ${self.initial_cost:.2f}"
        )

        return self._build_observation(reward=0.0)

    def _initialize_resources(self) -> Dict[str, Resource]:
        """Create initial cloud infrastructure resources.

        Sets up a realistic multi-cloud environment with optimization opportunities:
        - One unattached EBS volume (easy task)
        - One public S3 bucket (medium task)
        - One oversized EC2 instance (hard task)
        - Supporting resources for context

        Returns:
            Dictionary mapping resource IDs to Resource objects
        """
        resources = {}

        # Task 1: Unattached EBS volume (Easy) - $15/month savings
        resources["vol-unattached-001"] = Resource(
            resource_id="vol-unattached-001",
            resource_type=ResourceType.EBS_VOLUME,
            name="Orphaned Data Volume",
            monthly_cost=15.0,
            is_attached=False,
            storage_gb=100.0,
            needs_fixing=True,
        )

        # Task 2: Public S3 bucket (Medium) - Security issue
        resources["s3-public-bucket"] = Resource(
            resource_id="s3-public-bucket",
            resource_type=ResourceType.S3_BUCKET,
            name="World Readable Bucket",
            monthly_cost=25.0,
            is_public=True,
            storage_gb=500.0,
            needs_fixing=True,
        )

        # Task 3: Expensive EC2 instance (Hard) - $180/month with low utilization
        resources["i-expensive-prod"] = Resource(
            resource_id="i-expensive-prod",
            resource_type=ResourceType.EC2_INSTANCE,
            name="Production Web Server",
            monthly_cost=self.INSTANCE_TYPE_COSTS["m5.2xlarge"],
            instance_type="m5.2xlarge",
            cpu_utilization=12.5,
            needs_fixing=True,
        )

        # Supporting resource: Application server (healthy, low cost)
        resources["i-moderate-app"] = Resource(
            resource_id="i-moderate-app",
            resource_type=ResourceType.EC2_INSTANCE,
            name="Application Server",
            monthly_cost=self.INSTANCE_TYPE_COSTS["t3.large"],
            instance_type="t3.large",
            cpu_utilization=45.0,
        )

        # Supporting resource: Private S3 bucket for logs (healthy)
        resources["s3-private-logs"] = Resource(
            resource_id="s3-private-logs",
            resource_type=ResourceType.S3_BUCKET,
            name="Application Logs",
            monthly_cost=30.0,
            is_public=False,
            storage_gb=2000.0,
        )

        # Supporting resource: Attached data volume (healthy)
        resources["vol-attached-data"] = Resource(
            resource_id="vol-attached-data",
            resource_type=ResourceType.EBS_VOLUME,
            name="Production Data Volume",
            monthly_cost=50.0,
            is_attached=True,
            storage_gb=500.0,
        )

        logger.debug(f"Initialized {len(resources)} resources")
        return resources

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Processes the agent's action, updates infrastructure state, calculates
        rewards, and checks for episode termination.

        Args:
            action: Agent action to execute

        Returns:
            Tuple of (observation, reward, done, info) aligned with OpenEnv step semantics
        """
        self.current_step += 1
        logger.debug(f"Step {self.current_step}: executing {action.command} on {action.resource_id}")

        pre_cost = self._calculate_total_cost()
        pre_open_issues = self._count_open_issues()

        reward = 0.0
        info: Dict[str, Any] = {}

        # Validate resource exists
        if action.resource_id not in self.resources:
            logger.warning(f"Action targeted non-existent resource: {action.resource_id}")
            reward = -0.1
            info["error"] = f"Resource {action.resource_id} not found"
        else:
            reward, info = self._execute_action(action)

        shaping_reward, shaping_info = self._calculate_shaping_reward(pre_cost, pre_open_issues)
        reward += shaping_reward
        info.update(shaping_info)

        repeat_penalty = self._calculate_repeat_penalty(action)
        reward += repeat_penalty
        if repeat_penalty < 0:
            info["repeat_action_penalty"] = round(repeat_penalty, 2)

        task_scores = self._grade_tasks()
        info["task_scores"] = task_scores
        info["operational_scores"] = self._calculate_operational_scores()

        reward = round(max(-1.0, min(1.0, reward)), 2)

        # Check if episode is complete
        done = self._is_episode_complete()

        observation = self._build_observation(reward=reward, info=info, done=done)
        return observation, reward, done, info

    def _count_open_issues(self) -> int:
        """Return number of open optimization and security issues."""
        return sum(1 for resource in self.resources.values() if resource.needs_fixing)

    def _calculate_shaping_reward(self, pre_cost: float, pre_open_issues: int) -> Tuple[float, Dict[str, Any]]:
        """Provide dense reward for incremental cost and risk improvements."""
        post_cost = self._calculate_total_cost()
        post_open_issues = self._count_open_issues()

        cost_saved = max(0.0, pre_cost - post_cost)
        issues_reduced = max(0, pre_open_issues - post_open_issues)

        cost_reward = min(0.08, cost_saved / 200.0)
        risk_reward = min(0.08, issues_reduced * 0.04)
        shaping_reward = round(cost_reward + risk_reward, 2)

        return shaping_reward, {
            "cost_shaping_reward": round(cost_reward, 2),
            "risk_shaping_reward": round(risk_reward, 2),
            "cost_saved_this_step": round(cost_saved, 2),
            "issues_reduced_this_step": int(issues_reduced),
        }

    def _calculate_repeat_penalty(self, action: Action) -> float:
        """Penalize repeated action loops to discourage unproductive behavior."""
        signature = (action.command.lower(), action.resource_id)
        self.action_history.append(signature)

        if self.last_action_signature == signature:
            self.repeat_action_streak += 1
        else:
            self.repeat_action_streak = 0

        self.last_action_signature = signature

        if self.repeat_action_streak <= 0:
            return 0.0

        return round(-0.03 * min(4, self.repeat_action_streak), 2)

    def _grade_tasks(self) -> Dict[str, float]:
        """Programmatic deterministic graders for easy/medium/hard tasks."""
        easy_score = 0.0 if "vol-unattached-001" in self.resources else 1.0

        medium_score = 0.0
        medium_resource = self.resources.get("s3-public-bucket")
        if medium_resource is None or not medium_resource.is_public:
            medium_score = 1.0

        hard_score = 0.0
        hard_resource = self.resources.get("i-expensive-prod")
        if hard_resource is not None:
            target_cost = self.INSTANCE_TYPE_COSTS["t3.large"]
            current_cost = hard_resource.monthly_cost
            if hard_resource.instance_type == "t3.large":
                hard_score = 1.0
            elif current_cost > target_cost:
                hard_score = min(0.95, max(0.0, (180.0 - current_cost) / (180.0 - target_cost)))

        return {
            "easy": round(easy_score, 2),
            "medium": round(medium_score, 2),
            "hard": round(hard_score, 2),
        }

    def _calculate_operational_scores(self) -> Dict[str, float]:
        """Return practical KPI-style scores for cost and governance quality."""
        current_cost = self._calculate_total_cost()
        cost_reduction = max(0.0, self.initial_cost - current_cost)
        cost_efficiency = min(1.0, cost_reduction / 130.0)

        public_bucket_count = sum(
            1
            for resource in self.resources.values()
            if resource.resource_type == ResourceType.S3_BUCKET and resource.is_public
        )
        security_posture = 1.0 if public_bucket_count == 0 else 0.0

        orphaned_volume_count = sum(
            1
            for resource in self.resources.values()
            if resource.resource_type == ResourceType.EBS_VOLUME and not resource.is_attached
        )
        hygiene_score = 1.0 if orphaned_volume_count == 0 else 0.0

        overall = (cost_efficiency + security_posture + hygiene_score) / 3.0
        return {
            "cost_efficiency": round(cost_efficiency, 2),
            "security_posture": round(security_posture, 2),
            "resource_hygiene": round(hygiene_score, 2),
            "overall": round(overall, 2),
        }

    def state(self) -> Observation:
        """Return the current environment state without mutating it."""
        done = self._is_episode_complete()
        return self._build_observation(
            reward=0.0,
            info={"state_query": True},
            done=done,
        )

    def _execute_action(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        """Execute an action and compute reward.

        Dispatches to appropriate handler based on action command.

        Args:
            action: Action to execute

        Returns:
            Tuple of (reward, info dict)
        """
        command = action.command.lower()
        resource_id = action.resource_id
        resource = self.resources[resource_id]
        reward = 0.0
        info: Dict[str, Any] = {"command": command, "resource_id": resource_id}

        try:
            if command == "delete_resource":
                reward, info = self._handle_delete_resource(resource, info)
            elif command == "make_private":
                reward, info = self._handle_make_private(resource, info)
            elif command == "downsize_instance":
                new_instance_type = action.parameters.get("instance_type", "t3.medium")
                reward, info = self._handle_downsize_instance(resource, new_instance_type, info)
            else:
                reward = -0.05
                info["error"] = f"Unknown command: {command}"
        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            reward = -0.1
            info["error"] = str(e)

        return reward, info

    def _handle_delete_resource(
        self, resource: Resource, info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Handle resource deletion action.

        Validates that deletion is appropriate and removes resource from inventory.
        Awards reward for successfully deleting unattached volumes (Easy task).

        Args:
            resource: Resource to potentially delete
            info: Action info dict to update

        Returns:
            Tuple of (reward, updated info dict)
        """
        reward = 0.0

        if resource.resource_type == ResourceType.EBS_VOLUME:
            if not resource.is_attached:
                # Successfully deleted unattached volume - Easy task complete
                cost_saved = resource.monthly_cost
                del self.resources[resource.resource_id]

                self.task_progress[TaskType.EASY].progress = 1.0
                self.task_progress[TaskType.EASY].completed = True

                reward = self.TASK_REWARDS[TaskType.EASY]
                info.update({
                    "success": True,
                    "cost_saved": round(cost_saved, 2),
                    "task_completed": "easy",
                })
                logger.info(f"Deleted unattached volume {resource.resource_id}, saved ${cost_saved:.2f}")
            else:
                # Cannot delete attached volume - invalid action
                reward = -0.15
                info["error"] = "Cannot delete volume that is currently attached"
                logger.warning(f"Attempted to delete attached volume {resource.resource_id}")
        else:
            # Cannot delete non-volume resources
            reward = -0.1
            info["error"] = f"Can only delete EBS volumes, not {resource.resource_type}"
            logger.debug(f"Invalid delete on {resource.resource_type}")

        return reward, info

    def _handle_make_private(
        self, resource: Resource, info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Handle making a resource private (security fix).

        Updates resource access settings to restrict public access.
        Awards reward for fixing public S3 buckets (Medium task).

        Args:
            resource: Resource to make private
            info: Action info dict to update

        Returns:
            Tuple of (reward, updated info dict)
        """
        reward = 0.0

        if resource.resource_type == ResourceType.S3_BUCKET:
            if resource.is_public:
                # Successfully fixed public bucket - Medium task complete
                resource.is_public = False
                resource.needs_fixing = False

                self.task_progress[TaskType.MEDIUM].progress = 1.0
                self.task_progress[TaskType.MEDIUM].completed = True

                reward = self.TASK_REWARDS[TaskType.MEDIUM]
                info.update({
                    "success": True,
                    "security_fixed": True,
                    "task_completed": "medium",
                })
                logger.info(f"Made bucket {resource.resource_id} private - security issue fixed")
            else:
                # Bucket already private - wasted action
                reward = -0.05
                info["warning"] = "Bucket is already private"
                logger.debug(f"Attempted to make already-private bucket {resource.resource_id} private")
        else:
            # Wrong resource type
            reward = -0.1
            info["error"] = f"Can only make S3 buckets private, not {resource.resource_type}"
            logger.debug(f"Invalid make_private on {resource.resource_type}")

        return reward, info

    def _handle_downsize_instance(
        self,
        resource: Resource,
        new_instance_type: str,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Handle EC2 instance downsizing action.

        Validates target instance type exists and is cheaper, then updates costs.
        Awards reward for downsizing the oversized production instance (Hard task).

        Args:
            resource: EC2 instance to downsize
            new_instance_type: Target instance type
            info: Action info dict to update

        Returns:
            Tuple of (reward, updated info dict)
        """
        reward = 0.0

        if resource.resource_type != ResourceType.EC2_INSTANCE:
            reward = -0.1
            info["error"] = f"Can only downsize EC2 instances, not {resource.resource_type}"
            logger.debug(f"Invalid downsize on {resource.resource_type}")
            return reward, info

        # Validate instance type exists
        if new_instance_type not in self.INSTANCE_TYPE_COSTS:
            reward = -0.1
            info["error"] = f"Unknown instance type: {new_instance_type}"
            logger.warning(f"Downsize to unknown type {new_instance_type}")
            return reward, info

        # Check if new type is cheaper
        new_cost = self.INSTANCE_TYPE_COSTS[new_instance_type]
        current_cost = resource.monthly_cost

        if new_cost >= current_cost:
            reward = -0.15
            info["error"] = f"Instance type {new_instance_type} (${new_cost:.2f}) is not cheaper than current (${current_cost:.2f})"
            logger.debug(f"Attempted non-cost-saving downsize: {current_cost:.2f} -> {new_cost:.2f}")
            return reward, info

        # Valid downsize - update instance
        cost_saved = current_cost - new_cost
        old_type = resource.instance_type

        resource.instance_type = new_instance_type
        resource.monthly_cost = new_cost
        resource.needs_fixing = False

        # Award reward based on which instance was downsized
        if resource.resource_id == "i-expensive-prod":
            # Downsized the specific expensive instance - Hard task complete
            self.task_progress[TaskType.HARD].progress = 1.0
            self.task_progress[TaskType.HARD].completed = True
            reward = self.TASK_REWARDS[TaskType.HARD]
            info["task_completed"] = "hard"
            logger.info(
                f"Downsized i-expensive-prod: {old_type} -> {new_instance_type}, "
                f"saved ${cost_saved:.2f}/month"
            )
        else:
            # Downsized other instance - partial credit for Hard task
            self.task_progress[TaskType.HARD].progress = min(
                1.0,
                self.task_progress[TaskType.HARD].progress + 0.25,
            )
            reward = 0.25
            logger.info(
                f"Downsized {resource.resource_id}: {old_type} -> {new_instance_type}, "
                f"saved ${cost_saved:.2f}/month"
            )

        info.update({
            "success": True,
            "cost_saved": round(cost_saved, 2),
            "old_instance_type": old_type,
            "new_instance_type": new_instance_type,
        })

        return reward, info

    def _is_episode_complete(self) -> bool:
        """Check if episode should terminate.

        Episode ends when:
        1. All three tasks are completed, or
        2. Maximum steps have been reached

        Returns:
            True if episode is complete
        """
        all_tasks_done = all(task.completed for task in self.task_progress.values())
        max_steps_reached = self.current_step >= self.max_steps

        if all_tasks_done:
            logger.info(f"Episode complete: all tasks finished at step {self.current_step}")
            return True

        if max_steps_reached:
            logger.info(f"Episode complete: reached max steps {self.max_steps}")
            return True

        return False

    def _calculate_total_cost(self) -> float:
        """Calculate total monthly infrastructure cost.

        Sums monthly costs of all resources currently in inventory.

        Returns:
            Total monthly cost in USD
        """
        total = sum(resource.monthly_cost for resource in self.resources.values())
        return round(total, 2)

    def _all_tasks_complete(self) -> bool:
        """Check if all tasks are completed.
        
        Returns:
            True if all tasks completed
        """
        return all(task.completed for task in self.task_progress.values())

    def _build_observation(
        self,
        reward: float = 0.0,
        info: Optional[Dict[str, Any]] = None,
        done: bool = False,
    ) -> Observation:
        """Construct environment observation for agent.

        Builds complete state representation including resources, costs,
        progress toward tasks, and reward for last action.

        Args:
            reward: Reward for the last action
            info: Additional step information

        Returns:
            Observation object ready for agent consumption
        """
        if info is None:
            info = {}

        current_cost = self._calculate_total_cost()

        # Calculate cost reduction percentage
        if self.initial_cost > 0:
            cost_reduction_pct = ((self.initial_cost - current_cost) / self.initial_cost) * 100
        else:
            cost_reduction_pct = 0.0

        # Count issues and completed tasks
        completed_task_types = [
            task_type for task_type, task in self.task_progress.items()
            if task.completed
        ]
        num_issues = sum(1 for r in self.resources.values() if r.needs_fixing)
        num_resources = len(self.resources)

        # Build progress dictionary
        progress_dict = {
            task_type: task.progress
            for task_type, task in self.task_progress.items()
        }

        # Create human-readable description
        description = (
            f"Step {self.current_step}/{self.max_steps}: "
            f"Monthly cost ${current_cost:.2f} ({cost_reduction_pct:.1f}% reduction), "
            f"{num_resources} resources ({num_issues} with issues), "
            f"{len(completed_task_types)}/3 tasks completed"
        )

        # Add financial tracking to info
        info.update({
            "initial_cost": round(self.initial_cost, 2),
            "current_cost": current_cost,
            "cost_reduction_percent": round(cost_reduction_pct, 2),
        })

        return Observation(
            description=description,
            resources=list(self.resources.values()),
            monthly_cost=current_cost,
            reward=reward,
            done=done,
            completed_tasks=completed_task_types,
            progress=progress_dict,
            info=info,
        )
