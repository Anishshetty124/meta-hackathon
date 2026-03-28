"""Development guide for Cloud FinOps environment.

This document provides instructions for setting up a development environment,
running tests, and contributing changes to the project.
"""

# ============================================================================
# SETUP
# ============================================================================

"""
PREREQUISITES:
    - Python 3.10, 3.11, or 3.12
    - Git
    - HuggingFace account (for LLM testing)

INSTALLATION:

    1. Clone the repository:
       git clone https://github.com/your-username/cloud-finops-auditor.git
       cd cloud-finops-auditor

    2. Create virtual environment:
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\\Scripts\\activate

    3. Install development dependencies:
       pip install -r requirements-full.txt

    4. Configure environment:
       cp .env.example .env
       # Edit .env with your credentials

    5. Verify installation:
       pytest -v
       python -m uvicorn main:app --reload
"""

# ============================================================================
# RUNNING TESTS
# ============================================================================

"""
RUN ALL TESTS:
    pytest -v

RUN SPECIFIC TEST FILE:
    pytest tests_comprehensive.py -v

RUN SPECIFIC TEST:
    pytest tests_comprehensive.py::TestResourceModel::test_valid_resource_creation -v

RUN TESTS WITH COVERAGE:
    pytest --cov=server --cov-report=html

RUN TESTS WITH MARKERS:
    pytest -m unit -v          # Only unit tests
    pytest -m integration -v   # Only integration tests
    pytest -m "not slow" -v    # All except slow tests

RUN TESTS WITH SPECIFIC TIMEOUT:
    pytest --timeout=10 -v     # Fail tests that take > 10 seconds

BENCHMARKING:
    pytest tests_comprehensive.py::TestPerformance -v --benchmark-only
"""

# ============================================================================
# CODE QUALITY
# ============================================================================

"""
TYPE CHECKING:
    mypy . --strict

CODE FORMATTING:
    black . --line-length=100

SORTING IMPORTS:
    isort . --profile=black --line-length=100

LINTING:
    flake8 . --max-line-length=100 --count --show-source
    pylint server/ main.py inference.py

RUNNING ALL CHECKS:
    black . --check --line-length=100
    isort . --check-only --profile=black --line-length=100
    mypy . --strict
    pylint server/ main.py inference.py
    pytest -v --cov=server

FIX FORMATTING AUTOMATICALLY:
    black . --line-length=100
    isort . --profile=black --line-length=100
"""

# ============================================================================
# RUNNING LOCALLY
# ============================================================================

"""
START THE SERVER:
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Server will be available at http://localhost:8000
Interactive docs at http://localhost:8000/docs

RUN THE AGENT:
    export API_BASE_URL=http://localhost:8000
    export MODEL_NAME=meta-llama/Llama-2-7b-chat
    export HF_TOKEN=your-token-here
    python inference.py

RUN EXAMPLES:
    python examples_advanced.py

BUILD DOCKER IMAGE:
    docker build -t finops-auditor:dev .

RUN DOCKER CONTAINER:
    docker run -p 8000:8000 \\
      -e API_BASE_URL=http://localhost:8000 \\
      -e MODEL_NAME=meta-llama/Llama-2-7b-chat \\
      -e HF_TOKEN=<token> \\
      finops-auditor:dev
"""

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

"""
cloud-finops-auditor/
├── server/                          # Main package
│   ├── __init__.py
│   ├── models.py                   # Pydantic data models
│   ├── environment.py              # CloudEnvironment simulation
│   ├── config.py                   # Configuration management
│   ├── metrics.py                  # Metrics & observability
│   ├── exceptions.py               # Custom exception hierarchy
│   └── py.typed                    # Type hints marker
├── main.py                         # FastAPI application
├── inference.py                    # LLM agent script
├── validate.py                     # Validation utilities
├── examples_advanced.py            # Usage examples
├── tests_comprehensive.py          # Test suite
├── requirements.txt                # Production dependencies
├── requirements-full.txt           # Dev + test dependencies
├── Dockerfile                      # Container configuration
├── openenv.yaml                    # OpenEnv specification
├── pyproject.toml                  # Project configuration
├── pytest.ini                      # Test configuration
├── API_DOCUMENTATION.md            # API reference
├── README.md                       # Project overview
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
└── .dockerignore                   # Docker ignore rules
"""

# ============================================================================
# ADDING FEATURES
# ============================================================================

"""
PROCESS FOR ADDING NEW FEATURES:

1. CREATE A BRANCH:
   git checkout -b feature/description

2. IMPLEMENT FEATURE:
   - Write code following existing patterns
   - Add comprehensive docstrings
   - Include type hints throughout
   - Add appropriate error handling

3. WRITE TESTS:
   - Unit tests in tests_comprehensive.py
   - Use fixtures for test data
   - Aim for >80% code coverage
   - Test both happy path and error cases

4. RUN QUALITY CHECKS:
   pytest -v --cov=server
   black . --check
   mypy .
   flake8 .

5. FIX ANY ISSUES:
   black .
   isort .
   mypy . (manual fixes may be needed)

6. COMMIT AND PUSH:
   git add .
   git commit -m "Add feature: description"
   git push origin feature/description

7. SUBMIT PULL REQUEST

EXAMPLE: Adding a new action type

1. Update server/models.py:
   - Add command to Action.command enum
   - Add docstring explaining new command

2. Update server/environment.py:
   - Add new action handler method
   - Add logging
   - Handle error cases

3. Update tests_comprehensive.py:
   - Add test for new action handler
   - Test error conditions

4. Update API_DOCUMENTATION.md:
   - Document new command
   - Include examples

5. Update README.md if user-facing change
"""

# ============================================================================
# DOCUMENTATION
# ============================================================================

"""
DOCSTRING FORMAT:

Use Google-style docstrings:

    def function_name(param1: str, param2: int) -> bool:
        '''Short description (one line).
        
        Longer description if needed. Explain the purpose,
        behavior, and any important details.
        
        Args:
            param1: Description of param1
            param2: Description of param2
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: When something is invalid
            RuntimeError: When something fails at runtime
        
        Examples:
            >>> result = function_name("test", 42)
            >>> assert result is True
        '''

UPDATE README FOR USER-FACING CHANGES
UPDATE API_DOCUMENTATION.md FOR API CHANGES
UPDATE pyproject.toml FOR DEPENDENCY CHANGES
"""

# ============================================================================
# DEBUGGING
# ============================================================================

"""
ENABLE DEBUG LOGGING:
    export DEBUG=true
    export LOG_LEVEL=DEBUG
    python -m uvicorn main:app --reload

USE PYTHON DEBUGGER:
    import pdb; pdb.set_trace()
    # Then step through code with n, s, c, etc.

INSPECT ENVIRONMENT STATE:
    env = CloudEnvironment(max_steps=100, random_seed=42)
    obs = env._build_observation()
    print(f"Resources: {obs.resources}")
    print(f"Progress: {obs.progress}")
    print(f"Cost: ${obs.monthly_cost}")

RUN HTTP REQUESTS MANUALLY:
    # In another terminal after starting server
    curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"seed": null}'
    
    curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \\
      -d '{"action": {"command": "delete_resource", "resource_id": "vol-unattached-001", "parameters": {}}}'
    
    curl http://localhost:8000/state
"""

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

"""
PROFILING:
    python -m cProfile -s cumulative inference.py > profile.txt
    cat profile.txt

MEMORY PROFILING:
    pip install memory-profiler
    python -m memory_profiler examples_advanced.py

BENCHMARKING:
    pytest tests_comprehensive.py::TestPerformance --benchmark-only

OPTIMIZATION CHECKLIST:
    ✓ Use async/await for I/O-bound operations
    ✓ Cache frequently accessed data
    ✓ Batch API calls when possible
    ✓ Use generators for large datasets
    ✓ Profile before optimizing
    ✓ Monitor production performance
"""
