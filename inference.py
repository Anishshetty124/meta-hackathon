"""Cloud FinOps AI agent using OpenAI-compatible inference.

Implements a hybrid agent that uses LLM recommendations with fast heuristic
fallbacks for cloud infrastructure optimization. Balances intelligence with
reliability to complete three optimization tasks.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
from openai import OpenAI

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
API_REQUEST_TIMEOUT = 10  # seconds
LLM_MAX_TOKENS = 500
DEFAULT_STEP_DELAY = 0.5  # seconds between steps for rate limiting
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds


def load_configuration() -> Tuple[str, str, str, str, bool]:
    """Load and validate required environment configuration.

    Reads environment endpoint, LLM endpoint, model name, and authentication token.
    Ensures all required variables are present before continuing.

    Returns:
        Tuple of (env_base_url, llm_base_url, model_name, api_token, heuristic_only)

    Raises:
        ValueError: If any required environment variable is missing
    """
    # Checker-friendly semantics:
    # - API_BASE_URL is the LLM endpoint
    # - ENV_BASE_URL is the environment endpoint (step/reset/state)
    llm_base_url = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/openai/v1").strip()
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    heuristic_only = os.getenv("BASELINE_MODE", "").strip().lower() == "heuristic"

    # Accept either HF_TOKEN (hackathon instruction) or OPENAI_API_KEY (common OpenAI client pattern).
    api_token = hf_token or openai_api_key

    errors = []
    if not llm_base_url and not heuristic_only:
        errors.append("API_BASE_URL environment variable is required unless BASELINE_MODE=heuristic")
    if not env_base_url:
        errors.append("ENV_BASE_URL environment variable is required")
    if not model_name:
        errors.append("MODEL_NAME environment variable is required")
    if not api_token and not heuristic_only:
        errors.append("Either HF_TOKEN or OPENAI_API_KEY environment variable is required unless BASELINE_MODE=heuristic")

    if errors:
        raise ValueError("; ".join(errors))

    logger.info(
        "Configuration loaded successfully "
        f"(model={model_name}, env={env_base_url}, llm={llm_base_url}, heuristic_only={heuristic_only})"
    )
    return env_base_url, llm_base_url, model_name, api_token, heuristic_only


def create_openai_client(model_name: str, api_token: str, llm_base_url: str) -> OpenAI:
    """Create OpenAI-compatible client for HuggingFace inference API.

    Configures the client to use HuggingFace's OpenAI-compatible endpoint
    with proper authentication headers.

    Args:
        model_name: Hugging Face model identifier
        api_token: API authentication token (HF token or OpenAI-style key)
        llm_base_url: OpenAI-compatible LLM endpoint

    Returns:
        Configured OpenAI client instance

    Raises:
        ValueError: If credentials are invalid or missing
    """
    if not api_token or not model_name:
        raise ValueError("Model name and HuggingFace token are required")

    client = OpenAI(
        api_key=api_token,
        base_url=llm_base_url,
    )
    logger.debug(f"OpenAI client created for model {model_name}")
    return client


def reset_environment(api_base_url: str) -> Dict[str, Any]:
    """Reset environment to initial state with fresh episode.

    Makes synchronized HTTP request to reset endpoint. Handles network
    errors with exponential backoff retry logic.

    Args:
        api_base_url: Base URL of environment API

    Returns:
        Initial observation object

    Raises:
        requests.RequestException: If reset fails after retries
    """
    url = f"{api_base_url}/reset"
    payload = {"seed": None, "difficulty": None}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=API_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            logger.info(f"Environment reset successfully (attempt {attempt + 1})")
            return response.json()

        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    f"Reset failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Reset failed after {MAX_RETRIES} attempts: {e}")
                raise


def get_environment_state(api_base_url: str) -> Dict[str, Any]:
    """Query current environment state without modifying it.

    Allows agent to plan multiple steps ahead without executing actions.

    Args:
        api_base_url: Base URL of environment API

    Returns:
        Current state dictionary with observation and metadata

   Raises:
        requests.RequestException: If state query fails
    """
    url = f"{api_base_url}/state"
    try:
        response = requests.get(url, timeout=API_REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to get environment state: {e}")
        raise


def execute_action(
    api_base_url: str,
    command: str,
    resource_id: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a single action in the environment.

    Sends action to environment API endpoint and receives updated observation.
    Handles network failures with immediate feedback to agent.

    Args:
        api_base_url: Base URL of environment API
        command: Action command name
        resource_id: Target resource identifier
        parameters: Command-specific parameters (default: empty dict)

    Returns:
        Updated observation object with reward signal

    Raises:
        ValueError: If action parameters are invalid
        requests.RequestException: If action execution fails
    """
    url = f"{api_base_url}/step"
    payload = {
        "action": {
            "command": command,
            "resource_id": resource_id,
            "parameters": parameters or {},
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=API_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        logger.debug(f"Action executed: {command} on {resource_id}")
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            logger.warning(f"Invalid action rejected by environment: {e}")
            raise ValueError(str(e))
        else:
            logger.error(f"Server error executing action: {e}")
            raise
    except requests.RequestException as e:
        logger.error(f"Network error executing action: {e}")
        raise


def identify_easy_task_opportunities(observation: Dict[str, Any]) -> list:
    """Find unattached EBS volumes that can be deleted (Easy task).

    Args:
        observation: Current observation

    Returns:
        List of unattached volume resource IDs with issues
    """
    opportunities = []
    for resource in observation.get("resources", []):
        is_volume = resource.get("resource_type") == "ebs_volume"
        is_unattached = not resource.get("is_attached", True)
        needs_attention = resource.get("needs_fixing", False)

        if is_volume and is_unattached and needs_attention:
            opportunities.append(resource["resource_id"])

    return opportunities


def identify_medium_task_opportunities(observation: Dict[str, Any]) -> list:
    """Find public S3 buckets needing security fixes (Medium task).

    Args:
        observation: Current observation

    Returns:
        List of public bucket resource IDs with issues
    """
    opportunities = []
    for resource in observation.get("resources", []):
        is_bucket = resource.get("resource_type") == "s3_bucket"
        is_public = resource.get("is_public", False)
        needs_attention = resource.get("needs_fixing", False)

        if is_bucket and is_public and needs_attention:
            opportunities.append(resource["resource_id"])

    return opportunities


def identify_hard_task_opportunities(observation: Dict[str, Any]) -> list:
    """Find oversized EC2 instances suitable for downsizing (Hard task).

    Identifies instances with low CPU utilization that waste resources.

    Args:
        observation: Current observation

    Returns:
        List of (resource_id, instance_type, monthly_cost) tuples
    """
    opportunities = []
    for resource in observation.get("resources", []):
        is_instance = resource.get("resource_type") == "ec2_instance"
        cpu_util = resource.get("cpu_utilization", 100.0)
        is_underutilized = cpu_util < 50.0
        needs_attention = resource.get("needs_fixing", False)

        if is_instance and is_underutilized and needs_attention:
            opportunities.append((
                resource["resource_id"],
                resource.get("instance_type", ""),
                resource.get("monthly_cost", 0.0),
            ))

    return opportunities


def get_model_recommendation(
    client: OpenAI,
    model_name: str,
    observation: Dict[str, Any],
    task_completion: Dict[str, bool],
) -> Optional[Dict[str, Any]]:
    """Request LLM to recommend next action based on current state.

    Sends infrastructure state to LLM and parses structured JSON response.
    Includes graceful error handling for malformed or invalid responses.

    Args:
        client: OpenAI-compatible client
        model_name: Model identifier
        observation: Current environment observation
        task_completion: Dictionary tracking which tasks are done

    Returns:
        Parsed action dictionary or None if LLM fails
    """
    completed_tasks = [name for name, done in task_completion.items() if done]
    resources_json = json.dumps(observation["resources"], indent=2)

    prompt = f"""You are a Cloud FinOps & Security Auditor. Analyze this infrastructure and recommend one optimization action.

Infrastructure Status:
{resources_json}

Completed Optimization Tasks: {completed_tasks}

Three tasks need completion:
1. EASY: Delete orphaned unattached disk (vol-unattached-001)
2. MEDIUM: Secure public S3 bucket (s3-public-bucket) by making it private
3. HARD: Right-size expensive instance (i-expensive-prod) from m5.2xlarge to t3.large

Current monthly cost: ${observation.get('monthly_cost', 0):.2f}

Respond with ONLY valid JSON (no other text):
{{
  "command": "delete_resource|make_private|downsize_instance",
  "resource_id": "the-target-resource-id",
  "parameters": {{"instance_type": "target-type-for-downsize"}}
}}"""

    try:
        logger.debug(f"Requesting recommendation from {model_name}")
        response = client.messages.create(
            model=model_name,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_response = response.content[0].text if response.content else ""
        logger.debug(f"LLM response: {raw_response[:100]}...")

        # Extract JSON from response
        json_start = raw_response.find("{")
        json_end = raw_response.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            parsed = json.loads(json_str)
            logger.info(f"LLM recommended: {parsed.get('command')} on {parsed.get('resource_id')}")
            return parsed

        logger.warning("LLM response did not contain valid JSON")
        return None

    except Exception as e:
        logger.warning(f"LLM inference failed ({type(e).__name__}): {e}")
        return None


def select_heuristic_action(
    observation: Dict[str, Any],
    task_completion: Dict[str, bool],
) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
    """Select action using deterministic heuristics when LLM unavailable.

    Implements task-priority scheduling: easy → medium → hard to ensure
    systematic progress even when LLM fails.

    Args:
        observation: Current environment observation
        task_completion: Dictionary tracking task completion status

    Returns:
        Tuple of (command, resource_id, parameters) or (None, None, {}) if no action available
    """
    logger.debug("Using heuristic action selection")

    # Priority 1: Complete easy task
    if not task_completion.get("easy", False):
        easy_targets = identify_easy_task_opportunities(observation)
        if easy_targets:
            resource_id = easy_targets[0]
            logger.info(f"Heuristic selected easy task: delete {resource_id}")
            return "delete_resource", resource_id, {}

    # Priority 2: Complete medium task
    if not task_completion.get("medium", False):
        medium_targets = identify_medium_task_opportunities(observation)
        if medium_targets:
            resource_id = medium_targets[0]
            logger.info(f"Heuristic selected medium task: make {resource_id} private")
            return "make_private", resource_id, {}

    # Priority 3: Complete hard task
    if not task_completion.get("hard", False):
        hard_targets = identify_hard_task_opportunities(observation)
        if hard_targets:
            resource_id, instance_type, cost = hard_targets[0]
            logger.info(f"Heuristic selected hard task: downsize {resource_id}")
            return "downsize_instance", resource_id, {"instance_type": "t3.large"}

    logger.debug("No heuristic action available - all tasks may be complete")
    return None, None, {}


def run_training_episode(
    env_base_url: str,
    client: Optional[OpenAI],
    model_name: str,
    episode_number: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Execute one complete training episode.

    Orchestrates agent-environment interaction loop: reset → observe → act
    → receive reward → repeat until episode termination.

    Args:
        env_base_url: Base URL of environment API
        client: OpenAI-compatible client, or None for heuristic-only mode
        model_name: Model identifier
        episode_number: Current episode number (1-indexed)
        max_steps: Maximum steps allowed per episode

    Returns:
        Episode statistics dictionary
    """
    logger.info(f"===== Episode {episode_number} Starting =====")

    episode_stats = {
        "episode": episode_number,
        "steps": 0,
        "total_reward": 0.0,
        "initial_cost": 0.0,
        "final_cost": 0.0,
        "cost_savings": 0.0,
        "tasks_completed": [],
        "success": False,
    }

    try:
        # Initialize episode
        initial_obs = reset_environment(env_base_url)
        episode_stats["initial_cost"] = initial_obs["monthly_cost"]

        task_completion = {"easy": False, "medium": False, "hard": False}
        current_observation = initial_obs

        logger.info(f"Episode started - Initial cost: ${episode_stats['initial_cost']:.2f}")

        # Action loop
        for step in range(1, max_steps + 1):
            # Update task completion status
            for completed_task in current_observation.get("completed_tasks", []):
                task_completion[completed_task.lower()] = True

            # Check early termination
            if all(task_completion.values()):
                logger.info("All tasks completed - episode terminating")
                episode_stats["success"] = True
                break

            logger.info(
                f"Step {step}/{max_steps} - Cost: ${current_observation['monthly_cost']:.2f}, "
                f"Tasks: {len([t for t in task_completion.values() if t])}/3"
            )

            # Decide action: try LLM first, fall back to heuristic
            recommendation = None
            if client is not None:
                recommendation = get_model_recommendation(
                    client, model_name, current_observation, task_completion
                )

            if recommendation:
                try:
                    command = recommendation.get("command", "")
                    resource_id = recommendation.get("resource_id", "")
                    parameters = recommendation.get("parameters", {})

                    if not command or not resource_id:
                        logger.warning("LLM recommendation missing required fields, using heuristic")
                        command, resource_id, parameters = select_heuristic_action(
                            current_observation, task_completion
                        )
                except Exception as e:
                    logger.warning(f"Error processing LLM recommendation: {e}, using heuristic")
                    command, resource_id, parameters = select_heuristic_action(
                        current_observation, task_completion
                    )
            else:
                command, resource_id, parameters = select_heuristic_action(
                    current_observation, task_completion
                )

            # Execute action if one was selected
            if command and resource_id:
                try:
                    current_observation = execute_action(
                        env_base_url, command, resource_id, parameters
                    )
                    step_reward = current_observation.get("reward", 0.0)
                    episode_stats["total_reward"] += step_reward
                    episode_stats["steps"] = step

                    logger.info(
                        f"  → Action: {command} on {resource_id}, "
                        f"Reward: {step_reward:+.2f}, Total: {episode_stats['total_reward']:+.2f}"
                    )

                    if current_observation.get("done", False):
                        logger.info("Episode done signal received")
                        episode_stats["success"] = True
                        break

                except ValueError as e:
                    logger.warning(f"Invalid action rejected: {e}")
                    continue
                except requests.RequestException as e:
                    logger.error(f"Action execution failed: {e}")
                    break

                time.sleep(DEFAULT_STEP_DELAY)  # Rate limiting
            else:
                logger.warning("No valid action could be determined - stopping episode")
                break

        # Record final stats
        episode_stats["final_cost"] = current_observation.get("monthly_cost", 0.0)
        episode_stats["cost_savings"] = max(0, episode_stats["initial_cost"] - episode_stats["final_cost"])
        episode_stats["tasks_completed"] = [k for k, v in task_completion.items() if v]

        logger.info(
            f"Episode {episode_number} complete - "
            f"Reward: {episode_stats['total_reward']:+.2f}, "
            f"Cost savings: ${episode_stats['cost_savings']:.2f}, "
            f"Tasks: {episode_stats['tasks_completed']}"
        )

    except requests.RequestException as e:
        logger.error(f"Network error during episode: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during episode: {e}", exc_info=True)

    return episode_stats


def main() -> int:
    """Main entry point for the inference agent script.

    Returns:
        Exit code (0 for success, 1 for configuration/fatal error)
    """
    try:
        # Load configuration
        env_base_url, llm_base_url, model_name, api_token, heuristic_only = load_configuration()

        # Create client unless explicitly running deterministic heuristic baseline.
        client: Optional[OpenAI] = None
        if not heuristic_only:
            client = create_openai_client(model_name, api_token, llm_base_url)
        else:
            logger.info("Running in heuristic-only baseline mode (BASELINE_MODE=heuristic)")

        # Run training episodes
        num_episodes = 3
        logger.info(f"Starting {num_episodes} training episodes")

        episode_results = []
        for episode_num in range(1, num_episodes + 1):
            stats = run_training_episode(
                env_base_url,
                client,
                model_name,
                episode_num,
                max_steps=100,
            )
            episode_results.append(stats)
            
            # Brief pause between episodes
            if episode_num < num_episodes:
                logger.info("Pausing between episodes...")
                time.sleep(2)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        total_reward = sum(s["total_reward"] for s in episode_results)
        total_savings = sum(s["cost_savings"] for s in episode_results)
        successful_episodes = sum(1 for s in episode_results if s["success"])

        logger.info(f"Episodes: {successful_episodes}/{num_episodes} successful")
        logger.info(f"Total reward: {total_reward:+.2f}")
        logger.info(f"Total cost savings: ${total_savings:.2f}")

        for result in episode_results:
            logger.info(
                f"  Episode {result['episode']}: "
                f"Reward={result['total_reward']:+.2f}, "
                f"Steps={result['steps']}, "
                f"Savings=${result['cost_savings']:.2f}, "
                f"Tasks={len(result['tasks_completed'])}/3"
            )

        logger.info("Agent training complete!")
        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
