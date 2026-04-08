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
    MIN_TASK_SCORE = 0.01
    MAX_TASK_SCORE = 0.99

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
        self.last_action_signature: Optional[Tuple[str, str]] = None
        self.no_progress_steps = 0

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
        self.last_action_signature = None
        self.no_progress_steps = 0

        # Reset task progress
        for task_type in TaskType:
            self.task_progress[task_type].progress = self.MIN_TASK_SCORE
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

        reward = 0.0
        info: Dict[str, Any] = {}
        previous_cost = self._calculate_total_cost()

        # Validate resource exists
        if action.resource_id not in self.resources:
            logger.warning(f"Action targeted non-existent resource: {action.resource_id}")
            reward = -0.1
            info["error"] = f"Resource {action.resource_id} not found"
        else:
            reward, info = self._execute_action(action)

        reward, shaping_info = self._apply_reward_shaping(
            action=action,
            base_reward=reward,
            previous_cost=previous_cost,
            info=info,
        )
        info.update(shaping_info)

        # Check if episode is complete
        done = self._is_episode_complete()

        observation = self._build_observation(reward=reward, info=info, done=done)
        return observation, reward, done, info

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
            elif command == "upsize_instance":
                new_instance_type = action.parameters.get("instance_type", "m5.large")
                reward, info = self._handle_upsize_instance(resource, new_instance_type, info)
            elif command == "attach_resource":
                target_instance_id = action.parameters.get("instance_id", "i-moderate-app")
                reward, info = self._handle_attach_resource(resource, target_instance_id, info)
            elif command == "detach_resource":
                reward, info = self._handle_detach_resource(resource, info)
            else:
                reward = -0.05
                info["error"] = f"Unknown command: {command}"
        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            reward = -0.1
            info["error"] = str(e)

        return reward, info

    def _handle_upsize_instance(
        self,
        resource: Resource,
        new_instance_type: str,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Handle EC2 upsizing for reliability-sensitive workloads."""
        if resource.resource_type != ResourceType.EC2_INSTANCE:
            return -0.1, {**info, "error": f"Can only upsize EC2 instances, not {resource.resource_type}"}

        if new_instance_type not in self.INSTANCE_TYPE_COSTS:
            return -0.1, {**info, "error": f"Unknown instance type: {new_instance_type}"}

        new_cost = self.INSTANCE_TYPE_COSTS[new_instance_type]
        current_cost = resource.monthly_cost
        if new_cost <= current_cost:
            return -0.12, {
                **info,
                "error": f"Upsize target {new_instance_type} must be more expensive than current type",
            }

        old_type = resource.instance_type
        resource.instance_type = new_instance_type
        resource.monthly_cost = new_cost
        cost_increase = new_cost - current_cost

        # Only high-utilization workloads justify upsizing; otherwise penalize waste.
        cpu_util = resource.cpu_utilization or 0.0
        if cpu_util >= 70.0:
            reward = 0.08
            reliability_note = "High utilization justified capacity increase"
        else:
            reward = -0.12
            reliability_note = "Upsize increased cost without utilization evidence"

        info.update({
            "success": True,
            "old_instance_type": old_type,
            "new_instance_type": new_instance_type,
            "cost_increase": round(cost_increase, 2),
            "reliability_note": reliability_note,
        })
        return reward, info

    def _handle_attach_resource(
        self,
        resource: Resource,
        target_instance_id: str,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Attach an unattached EBS volume to an existing instance."""
        if resource.resource_type != ResourceType.EBS_VOLUME:
            return -0.1, {**info, "error": f"Can only attach EBS volumes, not {resource.resource_type}"}

        if resource.is_attached:
            return -0.05, {**info, "warning": "Volume is already attached"}

        target = self.resources.get(target_instance_id)
        if not target or target.resource_type != ResourceType.EC2_INSTANCE:
            return -0.1, {**info, "error": f"Target instance {target_instance_id} not found"}

        resource.is_attached = True
        # Attaching a previously orphaned volume resolves waste and risk signal.
        was_issue = resource.needs_fixing
        resource.needs_fixing = False

        reward = 0.12 if was_issue else 0.03
        info.update({
            "success": True,
            "attached_to": target_instance_id,
            "issue_resolved": was_issue,
        })
        return reward, info

    def _handle_detach_resource(
        self,
        resource: Resource,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Detach an attached EBS volume from a workload."""
        if resource.resource_type != ResourceType.EBS_VOLUME:
            return -0.1, {**info, "error": f"Can only detach EBS volumes, not {resource.resource_type}"}

        if not resource.is_attached:
            return -0.05, {**info, "warning": "Volume is already detached"}

        resource.is_attached = False
        # Detached-but-not-deleted resources tend to become cost leaks.
        resource.needs_fixing = True
        info.update({"success": True, "warning": "Detached volume may become orphaned cost"})
        return -0.02, info

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

                self.task_progress[TaskType.EASY].progress = self.MAX_TASK_SCORE
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

                self.task_progress[TaskType.MEDIUM].progress = self.MAX_TASK_SCORE
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
            self.task_progress[TaskType.HARD].progress = self.MAX_TASK_SCORE
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
                self.MAX_TASK_SCORE,
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

    def _apply_reward_shaping(
        self,
        action: Action,
        base_reward: float,
        previous_cost: float,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply trajectory-level shaping signals for better agent learning.

        Adds:
        - cost-efficiency bonus for measurable savings
        - repeated-action penalty to discourage loops
        - stalled-episode penalty after repeated no-progress steps
        - reliability penalty for unsafe production downsizing
        """
        shaped_reward = base_reward
        current_cost = self._calculate_total_cost()
        cost_saved = max(0.0, previous_cost - current_cost)
        shaping_info: Dict[str, Any] = {}

        if cost_saved > 0.0 and shaped_reward >= 0.0:
            efficiency_bonus = min(0.06, cost_saved / 2000.0)
            shaped_reward += efficiency_bonus
            shaping_info["efficiency_bonus"] = round(efficiency_bonus, 3)

        action_signature = (action.command, action.resource_id)
        if self.last_action_signature == action_signature:
            shaped_reward -= 0.03
            shaping_info["repeat_action_penalty"] = -0.03
        self.last_action_signature = action_signature

        task_completed = info.get("task_completed")
        if task_completed is None and cost_saved == 0.0:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = 0

        if self.no_progress_steps >= 5:
            shaped_reward -= 0.05
            shaping_info["stalled_episode_penalty"] = -0.05

        if (
            action.command == "downsize_instance"
            and action.resource_id == "i-expensive-prod"
            and info.get("success")
            and info.get("new_instance_type") in {"t3.micro", "t3.small"}
        ):
            shaped_reward -= 0.12
            shaping_info["reliability_risk_penalty"] = -0.12

        shaped_reward = max(-1.0, min(1.0, round(shaped_reward, 4)))
        return shaped_reward, shaping_info

    def _task_scorecard(self) -> Dict[str, float]:
        """Return deterministic task scores strictly inside (0, 1)."""
        return {
            task_type.value: round(
                min(self.MAX_TASK_SCORE, max(self.MIN_TASK_SCORE, task.progress)),
                3,
            )
            for task_type, task in self.task_progress.items()
        }

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
        last_action_error = info.get("error")
        info.update({
            "initial_cost": round(self.initial_cost, 2),
            "current_cost": current_cost,
            "cost_reduction_percent": round(cost_reduction_pct, 2),
            "task_scores": self._task_scorecard(),
            "overall_score": round(sum(progress_dict.values()) / 3.0, 3),
            "open_issues": num_issues,
            "last_action_error": last_action_error,
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
