from fastapi import FastAPI

from apps.api.dependencies import get_settings

app = FastAPI(title="Agentic ML Engineer API")


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
