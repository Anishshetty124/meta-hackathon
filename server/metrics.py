"""Metrics and observability for Cloud FinOps environment.

Provides performance monitoring, metrics collection, and observability
features for production deployments, including timing, resource utilization,
and custom business metrics.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from collections import defaultdict
from datetime import datetime


@dataclass
class TimingMetrics:
    """Timing metrics for operations.
    
    Tracks execution time of various operations for performance monitoring.
    """
    operation_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    
    def complete(self) -> None:
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def __str__(self) -> str:
        return f"{self.operation_name}: {self.duration_ms:.2f}ms"


@dataclass
class ActionMetrics:
    """Metrics for a single action execution.
    
    Tracks action success/failure, reward, and timing information.
    """
    action_id: int
    command: str
    resource_id: str
    success: bool
    reward: float
    duration_ms: float
    error_message: Optional[str] = None
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for an entire episode.
    
    Summarizes performance across all actions in an episode.
    """
    episode_number: int
    total_steps: int
    total_reward: float
    cost_initial: float
    cost_final: float
    cost_savings: float
    cost_savings_pct: float
    tasks_completed: int
    success: bool
    duration_seconds: float
    actions: list = field(default_factory=list)
    
    @property
    def avg_reward_per_step(self) -> float:
        """Average reward earned per step."""
        return self.total_reward / max(self.total_steps, 1)
    
    @property
    def avg_action_duration_ms(self) -> float:
        """Average action execution time."""
        if not self.actions:
            return 0.0
        total_ms = sum(a.duration_ms for a in self.actions)
        return total_ms / len(self.actions)


class MetricsCollector:
    """Collects and aggregates metrics across episodes and actions.
    
    Provides observability into environment performance with support
    for timing metrics, action tracking, and episode summaries.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.episodes: Dict[int, EpisodeMetrics] = {}
        self.current_episode: Optional[EpisodeMetrics] = None
        self.action_counter = 0
        self.operation_timings: Dict[str, list] = defaultdict(list)
    
    def start_episode(self, episode_number: int) -> None:
        """Initialize metrics for new episode.
        
        Args:
            episode_number: Episode identifier
        """
        self.current_episode = EpisodeMetrics(
            episode_number=episode_number,
            total_steps=0,
            total_reward=0.0,
            cost_initial=0.0,
            cost_final=0.0,
            cost_savings=0.0,
            cost_savings_pct=0.0,
            tasks_completed=0,
            success=False,
            duration_seconds=0.0,
        )
    
    def record_action(
        self,
        command: str,
        resource_id: str,
        success: bool,
        reward: float,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> int:
        """Record metrics for a single action.
        
        Args:
            command: Action command name
            resource_id: Target resource identifier
            success: Whether action succeeded
            reward: Reward earned from action
            duration_ms: Action execution time in milliseconds
            error: Error message if action failed
            
        Returns:
            Action ID for correlation
        """
        self.action_counter += 1
        
        action_metric = ActionMetrics(
            action_id=self.action_counter,
            command=command,
            resource_id=resource_id,
            success=success,
            reward=reward,
            duration_ms=duration_ms,
            error_message=error,
        )
        
        if self.current_episode:
            self.current_episode.actions.append(action_metric)
            self.current_episode.total_steps += 1
            self.current_episode.total_reward += reward
        
        return self.action_counter
    
    def record_timing(self, operation_name: str, duration_ms: float) -> None:
        """Record timing for an operation.
        
        Args:
            operation_name: Name of operation
            duration_ms: Duration in milliseconds
        """
        self.operation_timings[operation_name].append(duration_ms)
    
    def end_episode(
        self,
        cost_initial: float,
        cost_final: float,
        tasks_completed: int,
        success: bool,
        duration_seconds: float,
    ) -> Optional[EpisodeMetrics]:
        """Finalize metrics for current episode.
        
        Args:
            cost_initial: Initial monthly cost
            cost_final: Final monthly cost
            tasks_completed: Number of tasks completed
            success: Whether episode was successful
            duration_seconds: Total episode duration
            
        Returns:
            Completed EpisodeMetrics
        """
        if not self.current_episode:
            return None
        
        self.current_episode.cost_initial = cost_initial
        self.current_episode.cost_final = cost_final
        self.current_episode.cost_savings = max(0, cost_initial - cost_final)
        self.current_episode.cost_savings_pct = (
            (self.current_episode.cost_savings / cost_initial * 100)
            if cost_initial > 0 else 0
        )
        self.current_episode.tasks_completed = tasks_completed
        self.current_episode.success = success
        self.current_episode.duration_seconds = duration_seconds
        
        self.episodes[self.current_episode.episode_number] = self.current_episode
        
        result = self.current_episode
        self.current_episode = None
        return result
    
    def get_episode_metrics(self, episode_number: int) -> Optional[EpisodeMetrics]:
        """Retrieve metrics for specific episode.
        
        Args:
            episode_number: Episode identifier
            
        Returns:
            EpisodeMetrics if found, None otherwise
        """
        return self.episodes.get(episode_number)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all episodes.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.episodes:
            return {}
        
        episodes_list = list(self.episodes.values())
        
        return {
            "total_episodes": len(episodes_list),
            "successful_episodes": sum(1 for e in episodes_list if e.success),
            "total_reward": sum(e.total_reward for e in episodes_list),
            "avg_reward_per_episode": sum(e.total_reward for e in episodes_list) / len(episodes_list),
            "total_cost_savings": sum(e.cost_savings for e in episodes_list),
            "avg_cost_savings_per_episode": sum(e.cost_savings for e in episodes_list) / len(episodes_list),
            "avg_tasks_completed_per_episode": sum(e.tasks_completed for e in episodes_list) / len(episodes_list),
            "total_duration_seconds": sum(e.duration_seconds for e in episodes_list),
        }
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation.
        
        Args:
            operation_name: Name of operation to analyze
            
        Returns:
            Dictionary with timing statistics
        """
        timings = self.operation_timings.get(operation_name, [])
        
        if not timings:
            return {}
        
        return {
            "count": len(timings),
            "total_ms": sum(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "avg_ms": sum(timings) / len(timings),
            "p50_ms": sorted(timings)[len(timings) // 2],
            "p95_ms": sorted(timings)[int(len(timings) * 0.95)] if len(timings) > 1 else timings[0],
            "p99_ms": sorted(timings)[int(len(timings) * 0.99)] if len(timings) > 1 else timings[0],
        }
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        self.episodes.clear()
        self.current_episode = None
        self.action_counter = 0
        self.operation_timings.clear()
