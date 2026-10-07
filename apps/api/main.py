from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.dependencies import get_settings
from apps.api.routes.agent import router as agent_router

app = FastAPI(title="Agentic ML Engineer API")
app.include_router(agent_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "environment": settings.environment}
