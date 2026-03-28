"""Comprehensive API Documentation for Cloud FinOps Environment.

This document provides detailed API specifications, usage patterns, and
integration guidelines for consuming the Cloud FinOps OpenEnv environment.
"""

# ============================================================================
# API ENDPOINTS
# ============================================================================

"""
All endpoints follow the OpenEnv specification and return typed JSON responses.

Base URL: http://localhost:8000 (or deployed HuggingFace Space URL)
"""


# ============================================================================
# POST /reset
# ============================================================================

"""
Initialize or reset the environment to a fresh state.

Request:
    {
        "seed": null | integer,      # Optional: Random seed for reproducibility
        "difficulty": null | string  # Optional: Task difficulty (not used currently)
    }

Response (Observation):
    {
        "description": "Step 0/100: Monthly cost $365.00 (0.0% reduction), 6 resources (3 with issues), 0/3 tasks completed",
        "resources": [
            {
                "resource_id": "vol-unattached-001",
                "resource_type": "ebs_volume",
                "monthly_cost": 15.00,
                "is_attached": false,
                "is_public": false,
                "cpu_utilization": 0.0,
                "instance_type": null,
                "needs_fixing": true
            },
            // ... more resources
        ],
        "monthly_cost": 365.00,
        "reward": 0.0,
        "done": false,
        "completed_tasks": [],
        "progress": {
            "easy": 0.0,
            "medium": 0.0,
            "hard": 0.0
        },
        "info": {}
    }

Status Codes:
    200 OK       - Environment reset successful
    400 BadRequest - Invalid request parameters
    500 ServerError - Environment initialization failed
"""


# ============================================================================
# POST /step
# ============================================================================

"""
Execute an action in the environment and receive updated state.

Request:
    {
        "action": {
            "command": "delete_resource" | "make_private" | "downsize_instance" | 
                       "upsize_instance" | "attach_resource" | "detach_resource",
            "resource_id": "resource-id-string",
            "parameters": {
                // Command-specific parameters
                "instance_type": "t3.large"  // For downsize_instance
            }
        }
    }

Response (Observation):
    Same structure as /reset response, with updated state reflecting action execution

Status Codes:
    200 OK       - Action executed successfully
    400 BadRequest - Invalid action or parameters
    404 NotFound - Target resource not found
    500 ServerError - Action execution failed
"""


# ============================================================================
# GET /state
# ============================================================================

"""
Query the current environment state without executing any action.

Request: (GET, no body)

Response:
    {
        "observation": { ... },  // Current Observation object
        "episode_step": 5,      // Current step number
        "max_steps": 100        // Maximum steps for this episode
    }

Status Codes:
    200 OK       - State retrieved successfully
    500 ServerError - Failed to get state
"""


# ============================================================================
# GET /health
# ============================================================================

"""
Health check endpoint for monitoring and load balancing.

Request: (GET, no body)

Response:
    {
        "status": "healthy",
        "environment_initialized": true,
        "application": "Cloud FinOps & Security Auditor",
        "version": "1.0.0"
    }

Status Codes:
    200 OK       - Service is healthy and ready
    503 Unavailable - Service is not ready
"""


# ============================================================================
# GET /docs
# ============================================================================

"""
Interactive API documentation (Swagger UI).

Access: http://localhost:8000/docs

Provides:
    - Endpoint documentation
    - Interactive request/response testing
    - Schema exploration
"""


# ============================================================================
# DATA MODELS
# ============================================================================

"""
RESOURCE OBJECT:

{
    "resource_id": string,           // Unique identifier (1-100 chars)
    "resource_type": "ec2_instance" | "s3_bucket" | "ebs_volume",
    "monthly_cost": number,          // USD, 0-384, rounded to 2 decimals
    "is_attached": boolean,          // For EBS: attached to instance?
    "is_public": boolean,            // For S3: publicly readable?
    "cpu_utilization": number,       // For EC2: 0-100 percent
    "instance_type": string | null,  // For EC2: t3.large, m5.2xlarge, etc.
    "needs_fixing": boolean          // Security/optimization issue exists?
}

RESOURCE TYPES:
    - ec2_instance: EC2 compute instances
    - s3_bucket: S3 storage buckets
    - ebs_volume: Elastic Block Store volumes

OBSERVATION OBJECT:

{
    "description": string,           // Human-readable state description
    "resources": Resource[],         // List of cloud resources
    "monthly_cost": number,          // Total monthly cost in USD
    "reward": number,                // Reward from last action
    "done": boolean,                 // Episode termination flag
    "completed_tasks": string[],     // ['easy', 'medium', 'hard']
    "progress": {                    // Task completion (0.0-1.0)
        "easy": 0.0-1.0,
        "medium": 0.0-1.0,
        "hard": 0.0-1.0
    },
    "info": object                   // Additional metadata
}

ACTION OBJECT:

{
    "command": string,               // Action command
    "resource_id": string,           // Target resource
    "parameters": object             // Command-specific params
}

VALID COMMANDS:

1. delete_resource
   - Purpose: Remove a resource from infrastructure
   - Target: Unattached EBS volumes
   - Parameters: {}
   - Example:
     {
         "command": "delete_resource",
         "resource_id": "vol-unattached-001",
         "parameters": {}
     }

2. make_private
   - Purpose: Fix security vulnerability
   - Target: Public S3 buckets
   - Parameters: {}
   - Example:
     {
         "command": "make_private",
         "resource_id": "s3-public-bucket",
         "parameters": {}
     }

3. downsize_instance
   - Purpose: Reduce instance size for underutilized resources
   - Target: EC2 instances with low CPU utilization
   - Parameters: {"instance_type": "t3.large"}
   - Example:
     {
         "command": "downsize_instance",
         "resource_id": "i-expensive-prod",
         "parameters": {"instance_type": "t3.large"}
     }

4. upsize_instance
   - Purpose: Increase instance size (rarely optimal)
   - Target: Small EC2 instances
   - Parameters: {"instance_type": "m5.2xlarge"}

5. attach_resource
   - Purpose: Attach volume to instance
   - Target: Unattached EBS volumes
   - Parameters: {}

6. detach_resource
   - Purpose: Detach volume from instance
   - Target: Attached EBS volumes
   - Parameters: {}
"""


# ============================================================================
# REWARD SPECIFICATION
# ============================================================================

"""
The reward function provides scalar feedback for agent optimization.

TASK REWARDS (Optimal Behavior):
    - Easy Task (Delete EBS):     +0.35
    - Medium Task (Secure S3):    +0.50
    - Hard Task (Downsize EC2):   +0.60
    
PENALTIES (Suboptimal Behavior):
    - Invalid command:            -0.05
    - Resource not found:         -0.10
    - Invalid parameters:         -0.15

MAXIMUM POSSIBLE REWARD: 1.45 (all 3 tasks completed)

REWARD CHARACTERISTICS:
    - Scalar signal per step
    - Bounded between -0.15 and +0.60
    - Provides partial credit (not just terminal reward)
    - Incentivizes efficient task completion
"""


# ============================================================================
# ENVIRONMENT STATE
# ============================================================================

"""
INITIAL INFRASTRUCTURE:

Resource              | Type       | Cost    | Issue
----------------------|------------|---------|--------------------
vol-unattached-001    | EBS        | $15/mo  | Not attached (EASY)
s3-public-bucket      | S3         | $25/mo  | Publicly readable (MEDIUM)
i-expensive-prod      | EC2        | $180/mo | Oversized @ 12.5% CPU (HARD)
i-moderate-app        | EC2        | $65/mo  | Optimal
s3-private-logs       | S3         | $30/mo  | Healthy
vol-attached-data     | EBS        | $50/mo  | Healthy

TOTAL INITIAL COST: $365/mo

OPTIMAL FINAL COST: $50/mo (after all optimizations)
OPTIMAL TOTAL SAVINGS: $315/mo (86% reduction)
"""


# ============================================================================
# USAGE PATTERNS
# ============================================================================

"""
PATTERN 1: Synchronous HTTP Client

import requests

# Initialize
response = requests.post(
    "http://localhost:8000/reset",
    json={"seed": 42, "difficulty": None},
    timeout=10
)
observation = response.json()

# Step
action = {
    "command": "delete_resource",
    "resource_id": "vol-unattached-001",
    "parameters": {}
}
response = requests.post(
    "http://localhost:8000/step",
    json={"action": action},
    timeout=10
)
next_observation = response.json()

# Check state
response = requests.get("http://localhost:8000/state", timeout=5)
state = response.json()


PATTERN 2: LLM Agent Integration

from openai import OpenAI

# Configure OpenAI-compatible client
client = OpenAI(
    api_key="your-hf-token",
    base_url="https://api-inference.huggingface.co/openai/v1"
)

# Get model recommendation
response = client.messages.create(
    model="meta-llama/Llama-2-7b-chat",
    max_tokens=500,
    messages=[{
        "role": "user",
        "content": f"Recommend optimization for: {json.dumps(observation)}"
    }]
)


PATTERN 3: Error Handling

from server.exceptions import (
    FinOpsException,
    ActionExecutionError,
    ResourceNotFoundError
)

try:
    # Request
    response = requests.post(..., timeout=10)
    response.raise_for_status()
    observation = response.json()
except ResourceNotFoundError as e:
    # Handle specific error
    print(f"Resource not found: {e.resource_id}")
except ActionExecutionError as e:
    # Handle action errors
    print(f"Action failed: {e.message}")
except FinOpsException as e:
    # Handle all FinOps errors
    print(f"Environment error ({e.error_code}): {e.message}")
"""


# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================

"""
ENDPOINT LATENCIES (Typical):
    - GET /health:      < 10ms
    - POST /reset:      10-50ms
    - POST /step:       10-100ms
    - GET /state:       < 10ms
    - LLM inference:    5-30 seconds (model dependent)

RESOURCE REQUIREMENTS:
    - CPU:    2 vCPU minimum (optimized)
    - Memory: 2-4 GB (1GB for environment, 1-3GB for LLM)
    - Disk:   < 500MB (code + dependencies)
    - Network: HTTP requests (no streaming)

THROUGHPUT:
    - Episodes/minute: ~6-12 (with LLM inference)
    - Actions/second: ~10 (without LLM)
"""


# ============================================================================
# INTEGRATION GUIDELINES
# ============================================================================

"""
DOCKER DEPLOYMENT:

    docker build -t finops-auditor:latest .
    
    docker run -p 8000:8000 \\
      -e API_BASE_URL=http://localhost:8000 \\
      -e MODEL_NAME=meta-llama/Llama-2-7b-chat \\
      -e HF_TOKEN=<your-token> \\
      finops-auditor:latest

HUGGINGFACE SPACES:

    1. Create Space at huggingface.co/spaces
    2. Select Docker template
    3. Push code with valid Dockerfile
    4. Set Environment Variables:
       - API_BASE_URL
       - MODEL_NAME  
       - HF_TOKEN
    5. Space will auto-deploy and provide public URL

KUBERNETES DEPLOYMENT:

    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: finops-auditor
    spec:
      replicas: 3
      template:
        spec:
          containers:
          - name: finops
            image: finops-auditor:latest
            ports:
            - containerPort: 8000
            env:
            - name: API_BASE_URL
              value: "http://localhost:8000"
            - name: MODEL_NAME
              value: "meta-llama/Llama-2-7b-chat"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-credentials
                  key: token
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 10
              periodSeconds: 30
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
ISSUE: Connection refused on localhost:8000
SOLUTION: 
    - Verify server is running: python -m uvicorn main:app --reload
    - Check port 8000 is not in use: lsof -i :8000

ISSUE: Timeout on LLM API calls
SOLUTION:
    - Increase timeout: export LLM_TIMEOUT=60
    - Check HF token is valid
    - Verify model endpoint is accessible

ISSUE: Resource not found error
SOLUTION:
    - Verify resource_id exactly matches observation
    - Use GET /state to check current resources
    - Some resources may not exist in initial state

ISSUE: Low reward despite correct actions
SOLUTION:
    - Ensure preconditions are met (e.g., CPU < 50% for downsize)
    - Check needs_fixing flag is true for target
    - Review action parameters for typos

ISSUE: Episode terminates early
SOLUTION:
    - Check max_steps setting (export ENV_MAX_STEPS=...)
    - All tasks may be completed (valid termination)
    - Network errors may have occurred (check logs)
"""
