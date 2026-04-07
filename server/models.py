from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class ResourceType(str, Enum):
    """Cloud resource types."""
    EC2_INSTANCE = "ec2_instance"
    S3_BUCKET = "s3_bucket"
    EBS_VOLUME = "ebs_volume"


class TaskType(str, Enum):
    """Task difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Resource(BaseModel):
    """Represents a cloud infrastructure resource (EC2, S3, EBS).

    Encapsulates all operational and financial details about a cloud resource,
    including its lifecycle status, security posture, utilization metrics, and
    current cost. Supports tracking across heterogeneous resource types.
    """
    resource_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique resource identifier (e.g., vol-12345678, i-abcdef01)"
    )
    resource_type: ResourceType = Field(
        ...,
        description="Type of cloud resource: EC2 instance, S3 bucket, or EBS volume"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable resource name for display and identification"
    )
    monthly_cost: float = Field(
        ...,
        ge=0.0,
        le=100000.0,
        description="Monthly cost in USD (range: 0-100000)"
    )
    is_attached: bool = Field(
        default=True,
        description="Attachment status: True if resource is in use, False if orphaned"
    )
    is_public: bool = Field(
        default=False,
        description="Access status: True if publicly accessible (security concern)"
    )
    instance_type: Optional[str] = Field(
        None,
        description="EC2 instance type code (t3.large, m5.2xlarge, etc.) for compute resources"
    )
    storage_gb: Optional[float] = Field(
        None,
        ge=0.0,
        le=100000.0,
        description="Storage capacity in GB for storage resources"
    )
    cpu_utilization: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Current CPU utilization as percentage (0-100) for compute resources"
    )
    needs_fixing: bool = Field(
        default=False,
        description="Critical flag: True if resource has performance/security issues"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata for future attributes and tags"
    )

    @field_validator('monthly_cost')
    @classmethod
    def round_cost_to_cents(cls, value: float) -> float:
        """Round monetary values to standard two-decimal places."""
        if value < 0:
            raise ValueError("Monthly cost cannot be negative")
        return round(value, 2)

    @field_validator('cpu_utilization')
    @classmethod
    def validate_cpu_percentage(cls, value: Optional[float]) -> Optional[float]:
        """Ensure CPU utilization is within valid percentage range."""
        if value is not None and not (0.0 <= value <= 100.0):
            raise ValueError(f"CPU utilization must be 0-100%, got {value}")
        return value


class Action(BaseModel):
    """Agent action to modify infrastructure state.

    Represents a discrete action that an agent can execute to change the cloud
    infrastructure. Each action targets a specific resource and includes optional
    parameters for command-specific configuration.
    """
    command: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Action command name: delete_resource, make_private, downsize_instance, etc."
    )
    resource_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target resource ID that this action will affect"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command-specific parameters (e.g., instance_type for downsize actions)"
    )

    @field_validator('command')
    @classmethod
    def validate_command_is_recognized(cls, value: str) -> str:
        """Ensure command is one of the supported action types."""
        valid_actions = {
            'delete_resource', 'make_private', 'downsize_instance',
            'upsize_instance', 'attach_resource', 'detach_resource'
        }
        normalized_command = value.lower()
        if normalized_command not in valid_actions:
            raise ValueError(
                f"Unknown command '{value}'. "
                f"Supported commands: {', '.join(sorted(valid_actions))}"
            )
        return normalized_command

    class Config:
        json_schema_extra = {
            "example": {
                "command": "delete_resource",
                "resource_id": "vol-12345678",
                "parameters": {}
            }
        }


class Reward(BaseModel):
    """Typed reward model for step outcomes.

    Keeps reward semantics explicit for grading and baseline reporting while
    preserving the scalar signal used by the OpenEnv observation payload.
    """
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Scalar reward score from -1.0 to 1.0"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short reason describing why this reward was produced"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured reward components"
    )


class Observation(BaseModel):
    """Environment state observed after an action is executed.

    Provides complete feedback including infrastructure state, costs, task
    progress, and reward signals. This is the primary learning signal for
    the training agent.
    """
    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Human-readable description of current environment state and recent changes"
    )
    resources: List[Resource] = Field(
        default_factory=list,
        description="Complete list of current cloud infrastructure resources"
    )
    monthly_cost: float = Field(
        ...,
        ge=0.0,
        le=1000000.0,
        description="Total monthly cost of all infrastructure in USD"
    )
    reward: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Reward signal from -1.0 (very bad) to 1.0 (excellent) for this action"
    )
    done: bool = Field(
        default=False,
        description="Episode termination flag: True when all tasks complete or max steps reached"
    )
    completed_tasks: List[TaskType] = Field(
        default_factory=list,
        description="List of task types that have been successfully completed"
    )
    progress: Dict[TaskType, float] = Field(
        default_factory=dict,
        description="Fractional progress toward each task (0.0 = none, 1.0 = complete)"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostic information including errors and cost savings"
    )

    @field_validator('progress')
    @classmethod
    def validate_progress_range(cls, value: Dict[TaskType, float]) -> Dict[TaskType, float]:
        """Ensure all progress values are valid probabilities between 0.0 and 1.0."""
        for task_type, progress_val in value.items():
            if not isinstance(progress_val, (int, float)):
                raise ValueError(f"Progress value for {task_type} must be numeric")
            if not (0.0 <= progress_val <= 1.0):
                raise ValueError(
                    f"Progress for {task_type} must be 0.0-1.0, got {progress_val}"
                )
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Step 1/100: Monthly cost $350.00 (4.1% reduction), 6 resources (2 with issues), 1/3 tasks completed",
                "resources": [],
                "monthly_cost": 350.0,
                "reward": 0.35,
                "done": False,
                "completed_tasks": ["easy"],
                "progress": {"easy": 1.0, "medium": 0.0, "hard": 0.0},
                "info": {"cost_saved": 15.0, "task_completed": "easy"}
            }
        }


class ResetRequest(BaseModel):
    """HTTP request to reset environment to initial state.

    Optionally specifies a random seed for reproducible simulations and
    a difficulty level for task filtering.
    """
    seed: Optional[int] = Field(
        None,
        ge=0,
        le=2**31 - 1,
        description="Random seed for reproducible episode generation (None for random)"
    )
    difficulty: Optional[TaskType] = Field(
        None,
        description="Optional task difficulty level for filtering (easy, medium, hard)"
    )


class StepRequest(BaseModel):
    """HTTP request to execute one environment step.

    Contains the agent's action that will be processed to update the
    infrastructure state and return new observations.
    """
    action: Action = Field(
        ...,
        description="Agent action to execute in the environment"
    )

    @field_validator('action')
    @classmethod
    def validate_action_completeness(cls, value: Action) -> Action:
        """Ensure action has required fields."""
        if not value.command or not value.resource_id:
            raise ValueError("Action must specify both command and resource_id")
        return value


class StateResponse(BaseModel):
    """HTTP response containing current environment state.

    Provides complete state information without executing actions,
    allowing agents to plan multiple steps ahead if needed.
    """
    observation: Observation = Field(
        ...,
        description="Current environment observation including all resource state"
    )
    episode_step: int = Field(
        ...,
        ge=0,
        description="Current step number within the episode (0-indexed)"
    )
    max_steps: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Maximum allowed steps before episode automatic termination"
    )
