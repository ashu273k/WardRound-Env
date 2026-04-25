import os
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, field_validator

try:
    from .environment import WardRoundEnvironment
    from ..models import Action, Observation
except (ModuleNotFoundError, ImportError):
    from server.environment import WardRoundEnvironment
    from models import Action, Observation

# --- REQUIRED MODELS (FLAT) ---

class StepRequest(BaseModel):
    action_type: str = Field(..., description="Clinical action to perform (e.g., 'ask_nurse')", example="request_test")
    patient_id: str = Field(..., description="ID of the target patient", example="P001")

    @field_validator("action_type")
    @classmethod
    def validate_action(cls, v: str) -> str:
        valid_actions = [
            "ask_nurse", "request_test", "ask_consultant", 
            "decide_treatment", "reassure_patient", "present_case", "escalate"
        ]
        if v not in valid_actions:
            raise ValueError(f"Invalid action_type: {v}. Must be one of {valid_actions}")
        return v

# --- APP INITIALIZATION ---

app = FastAPI(
    title="WardRound-Env",
    description="Multi-agent hospital RL environment for Meta OpenEnv Hackathon.",
    version="1.1.0"
)

# Shared environment instance for the session
env = WardRoundEnvironment()

# Hugging Face Spaces root_path handling
if os.environ.get("SPACE_ID"):
    app.root_path = f"/embed/{os.environ.get('SPACE_ID')}/"

app.docs_url = "/docs"
app.openapi_url = "/openapi.json"

@app.get("/")
def root():
    return {
        "message": "WardRound-Env (Flat API) is running.",
        "health": "/health",
        "docs": "/docs",
        "step_schema": "POST /step { 'action_type': '...', 'patient_id': '...' }"
    }

@app.get("/health")
def health():
    return {"status": "ok", "trust_score": env._state.trust_score}

@app.post("/reset")
def reset(seed: int = 42):
    """Resets the ward state and deteriorates patients based on seed."""
    obs = env.reset(seed=seed)
    return obs

@app.post("/step")
def step(req: StepRequest):
    """
    Perform a clinical action. 
    Accepts a FLAT request body for ease of use in Swagger UI.
    """
    # 1. Validate environment state
    if not env._state.patients:
        raise HTTPException(status_code=400, detail="Environment not reset. Please call /reset first.")

    # 2. Validate patient_id
    if req.patient_id not in [p.id for p in env._state.patients]:
        raise HTTPException(status_code=422, detail=f"Patient ID '{req.patient_id}' not found in current ward.")

    # 3. Convert flat request to internal Action model
    action = Action(
        action_type=req.action_type,
        patient_id=req.patient_id
    )

    # 4. Apply step logic
    try:
        obs = env.step(action)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step execution error: {str(e)}")

@app.get("/schema")
def get_openapi_schema():
    return app.openapi()

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
