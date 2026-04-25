"""FastAPI app exposing WardRoundEnvironment via OpenEnv HTTP server."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies first."
    ) from e

try:
    from ..models import Action, Observation
    from .environment import WardRoundEnvironment
except (ModuleNotFoundError, ImportError):
    from models import Action, Observation
    from server.environment import WardRoundEnvironment


import os

app = create_app(
    WardRoundEnvironment,
    Action,
    Observation,
    env_name="wardround-env",
    max_concurrent_envs=4,
)

# Hugging Face Spaces often require a root_path for /docs to work
if os.environ.get("SPACE_ID"):
    title = os.environ.get("SPACE_ID").split("/")[-1]
    app.root_path = f"/embed/{os.environ.get('SPACE_ID')}/"

# Explicitly enable documentation for Hugging Face Spaces
app.docs_url = "/docs"
app.redoc_url = "/redoc"
app.openapi_url = "/openapi.json"


@app.get("/")
def root() -> dict[str, str]:
    """Human-friendly landing endpoint for local testing."""
    return {
        "message": "WardRound-Env server is running.",
        "health": "/health",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server for local development."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
