"""Custom exception hierarchy for Cloud FinOps environment.

Provides structured error handling with specific exception types for
different failure modes, enabling precise error recovery and debugging.
"""


class FinOpsException(Exception):
    """Base exception for all FinOps environment errors.
    
    All environment-specific exceptions inherit from this, allowing
    callers to catch all FinOps errors with a single except clause.
    """
    
    def __init__(self, message: str, error_code: str = "UNKNOWN_ERROR"):
        """Initialize exception with message and error code.
        
        Args:
            message: Human-readable error description
            error_code: Machine-readable error identifier
        """
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class ConfigurationError(FinOpsException):
    """Raised when environment configuration is invalid or incomplete.
    
    Examples:
        - Missing required environment variables
        - Invalid API endpoints
        - Authentication credentials not available
    """
    
    def __init__(self, message: str):
        super().__init__(message, "CONFIG_ERROR")


class EnvironmentError(FinOpsException):
    """Raised when environment state becomes invalid.
    
    Examples:
        - Episode max steps exceeded
        - Invalid environment initialization
        - Corrupted internal state
    """
    
    def __init__(self, message: str):
        super().__init__(message, "ENVIRONMENT_ERROR")


class ActionExecutionError(FinOpsException):
    """Raised when action execution fails.
    
    Examples:
        - Invalid action command
        - Target resource not found
        - Action preconditions not met
    """
    
    def __init__(self, message: str, action_command: str = "", resource_id: str = ""):
        self.action_command = action_command
        self.resource_id = resource_id
        super().__init__(message, "ACTION_ERROR")


class ResourceNotFoundError(ActionExecutionError):
    """Raised when target resource doesn't exist.
    
    This is a specific case of ActionExecutionError for when the
    requested resource_id cannot be found in the environment.
    """
    
    def __init__(self, resource_id: str):
        msg = f"Resource '{resource_id}' not found"
        super().__init__(msg, resource_id=resource_id)
        self.error_code = "RESOURCE_NOT_FOUND"


class InvalidActionError(ActionExecutionError):
    """Raised when action parameters are invalid.
    
    Examples:
        - Unknown command
        - Missing required parameters
        - Invalid parameter values for this action
    """
    
    def __init__(self, message: str, command: str = ""):
        super().__init__(message, action_command=command)
        self.error_code = "INVALID_ACTION"


class APIError(FinOpsException):
    """Raised when HTTP API communication fails.
    
    Examples:
        - Network connection errors
        - HTTP status codes (4xx, 5xx)
        - Timeout on API call
        - JSON parsing errors
    """
    
    def __init__(self, message: str, status_code: int = 0, endpoint: str = ""):
        self.status_code = status_code
        self.endpoint = endpoint
        super().__init__(message, "API_ERROR")


class LLMInferenceError(FinOpsException):
    """Raised when LLM inference fails.
    
    Examples:
        - LLM API call failed
        - Invalid response format
        - JSON parsing of LLM output failed
        - Token limit exceeded
    """
    
    def __init__(self, message: str, recoverable: bool = True):
        self.recoverable = recoverable  # Can fallback to heuristic
        super().__init__(message, "LLM_ERROR")


class ValidationError(FinOpsException):
    """Raised when data validation fails.
    
    Examples:
        - Pydantic model validation error
        - Type mismatch
        - Out-of-range values
    """
    
    def __init__(self, message: str, field: str = ""):
        self.field = field
        super().__init__(message, "VALIDATION_ERROR")
