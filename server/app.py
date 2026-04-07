"""ASGI application entry module expected by OpenEnv validators."""

import uvicorn

from main import app

__all__ = ["app", "main"]


def main() -> None:
	"""Console entrypoint used by OpenEnv packaging checks."""
	uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
	main()
