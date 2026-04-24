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
except ModuleNotFoundError:
    from models import Action, Observation
    from server.environment import WardRoundEnvironment


app = create_app(
    WardRoundEnvironment,
    Action,
    Observation,
    env_name="wardround-env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server for local development."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
