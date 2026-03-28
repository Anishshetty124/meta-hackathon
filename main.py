"""FastAPI application implementing OpenEnv specification.

Provides HTTP endpoints for cloud infrastructure FinOps optimization simulation.
Manages environment lifecycle and coordinates agent interactions with the simulator.
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from server.models import ResetRequest, StepRequest, StateResponse, Observation
from server.environment import CloudEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global environment instance
environment: Optional[CloudEnvironment] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifecycle.

    Initializes the CloudEnvironment on startup and cleans up on shutdown.
    """
    global environment
    logger.info("Starting Cloud FinOps & Security Auditor application")
    environment = CloudEnvironment(max_steps=100)
    yield
    logger.info("Shutting down application")
    environment = None


app = FastAPI(
    title="Cloud FinOps & Security Auditor",
    description="OpenEnv-compatible simulator for cloud cost and security optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset", response_model=Observation)
async def reset(request: Optional[ResetRequest] = None) -> Observation:
    """Reset environment to initial state.

    Creates a new episode with optional reproducibility seed. Can be called
    at any time to start a fresh optimization task.

    Args:
        request: Optional reset parameters containing seed and difficulty

    Returns:
        Initial observation of the environment

    Raises:
        HTTPException: If environment initialization fails
    """
    global environment

    try:
        if request and request.seed is not None:
            logger.info(f"Resetting environment with seed {request.seed}")
            environment = CloudEnvironment(seed=request.seed, max_steps=100)
        else:
            logger.info("Resetting environment with random seed")
            environment = CloudEnvironment(max_steps=100)

        observation = environment.reset()
        logger.info(
            f"Environment reset successfully: "
            f"{len(observation.resources)} resources, "
            f"${observation.monthly_cost:.2f} initial cost"
        )
        return observation

    except ValueError as e:
        logger.error(f"Invalid reset request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid reset parameters: {str(e)}")
    except Exception as e:
        logger.error(f"Environment reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to reset environment")


@app.post("/step", response_model=Observation)
async def step(request: StepRequest) -> Observation:
    """Execute one step in the environment.

    Processes the agent's action, updates infrastructure state, and returns
    the new observation with reward signal.

    Args:
        request: Step request containing the action to execute

    Returns:
        Updated observation with new state and reward

    Raises:
        HTTPException: If environment uninitialized or step execution fails
    """
    global environment

    if environment is None:
        logger.warning("Step attempted before environment initialization")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )

    try:
        logger.debug(
            f"Executing action: {request.action.command} on {request.action.resource_id}"
        )
        observation, done = environment.step(request.action)

        # Ensure done flag is properly set
        observation.done = done

        if done:
            logger.info("Episode complete")

        return observation

    except ValueError as e:
        logger.error(f"Invalid action: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
    except Exception as e:
        logger.error(f"Step execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to execute step")


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """Retrieve current environment state without modifying it.

    Allows agents to query the state and plan actions without execution.
    Useful for lookahead planning strategies.

    Returns:
        Current state with observation and episode metadata

    Raises:
        HTTPException: If environment uninitialized
    """
    global environment

    if environment is None:
        logger.warning("State query attempted before environment initialization")
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call POST /reset first."
        )

    try:
        observation = environment._build_observation()

        return StateResponse(
            observation=observation,
            episode_step=environment.current_step,
            max_steps=environment.max_steps,
        )

    except Exception as e:
        logger.error(f"Failed to get state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve state")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for monitoring.

    Returns basic application status. Can be used for load balancer checks
    and container orchestration health probes.

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "environment_initialized": environment is not None,
        "application": "Cloud FinOps & Security Auditor",
        "version": "1.0.0"
    }


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information and navigation.

    Returns:
        API metadata and endpoint descriptions
    """
    return {
        "name": "Cloud FinOps & Security Auditor",
        "description": "OpenEnv-compatible simulator for cloud optimization",
        "version": "1.0.0",
        "endpoints": {
            "POST /reset": "Initialize or reset the environment to start an episode",
            "POST /step": "Execute an action and observe new state",
            "GET /state": "Query current state without executing actions",
            "GET /health": "Health check for monitoring",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /openapi.json": "OpenAPI specification",
        },
        "example_reset": {
            "method": "POST",
            "url": "/reset",
            "body": {"seed": None, "difficulty": None}
        },
        "example_step": {
            "method": "POST",
            "url": "/step",
            "body": {
                "action": {
                    "command": "delete_resource",
                    "resource_id": "vol-unattached-001",
                    "parameters": {}
                }
            }
        }
    }


if __name__ == "__main__":
    logger.info("Starting Uvicorn server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
