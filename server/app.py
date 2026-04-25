"""Clean FastAPI server for WardRound-Env.

NO OpenEnv wrapper - pure FastAPI with flat request format.
This ensures consistent behavior across local and deployed environments.
"""

import os
import sys
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure proper imports work in all contexts
try:
    from .environment import WardRoundEnvironment
except ImportError:
    from server.environment import WardRoundEnvironment

try:
    from ..models import Action
except ImportError:
    from models import Action


# ═══════════════════════════════════════════════════════════════════════════
# Request/Response Models (FLAT format - no nesting)
# ═══════════════════════════════════════════════════════════════════════════

class ResetRequest(BaseModel):
    seed: int = Field(default=42, description="Random seed for reproducibility")
    task_id: str = Field(default="easy", description="Task difficulty: easy, medium, hard")


class StepRequest(BaseModel):
    """Flat action request - NO nested 'action' field."""
    action_type: str = Field(
        ..., 
        description="Clinical action to perform",
        examples=["present_case", "ask_nurse", "request_test", "ask_consultant", "decide_treatment", "reassure_patient"]
    )
    patient_id: str = Field(
        ..., 
        description="ID of the target patient",
        examples=["P001", "P002"]
    )
    content: str = Field(default="", description="Optional action content/message")
    reason: str = Field(default=None, description="Optional reason for the action")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        valid_actions = [
            "present_case",
            "answer_question",
            "ask_nurse",
            "ask_consultant",
            "decide_treatment",
            "reassure_patient",
            "request_test",
            "escalate",
        ]
        if v not in valid_actions:
            raise ValueError(f"Invalid action_type: '{v}'. Must be one of {valid_actions}")
        return v


class StepResponse(BaseModel):
    """Response from step endpoint."""
    observation: Dict[str, Any]
    reward: float
    done: bool


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="WardRound-Env",
    description="Multi-agent hospital ward-round RL environment for OpenEnv Hackathon.",
    version="2.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Global environment instance
env = WardRoundEnvironment()

# Hugging Face Spaces compatibility
if os.environ.get("SPACE_ID"):
    app.root_path = f"/embed/{os.environ.get('SPACE_ID')}/"


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Landing page with API info."""
    return {
        "message": "WardRound-Env API is running.",
        "version": "2.0.0",
        "endpoints": {
            "health": "GET /health",
            "reset": "POST /reset",
            "step": "POST /step {action_type, patient_id}",
            "state": "GET /state",
            "schema": "GET /schema",
            "docs": "GET /docs",
        }
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/state")
def get_state():
    """Get current environment state."""
    return {
        "episode_id": env._state.episode_id,
        "step_count": env._state.step_count,
        "task_id": env._state.task_id,
        "time_remaining": env._state.time_remaining,
        "started": env._state.started,
    }


@app.get("/schema")
def get_schema():
    """Get action and observation schemas."""
    return {
        "action": {
            "type": "object",
            "required": ["action_type", "patient_id"],
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["present_case", "answer_question", "ask_nurse", "ask_consultant", 
                             "decide_treatment", "reassure_patient", "request_test", "escalate"]
                },
                "patient_id": {"type": "string"},
                "content": {"type": "string", "default": ""},
                "reason": {"type": "string", "default": None},
            }
        },
        "step_request_format": "FLAT: {action_type: str, patient_id: str}",
    }


@app.post("/reset")
def reset(req: ResetRequest = None):
    """Reset the environment to start a new episode."""
    if req is None:
        req = ResetRequest()
    
    print(f"[RESET] seed={req.seed}, task_id={req.task_id}")
    
    try:
        obs = env.reset(seed=req.seed, task_id=req.task_id)
        
        # Convert observation to dict
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else dict(obs)
        
        return {
            "observation": obs_dict,
            "reward": 0.0,
            "done": False,
        }
    except Exception as e:
        print(f"[RESET ERROR] {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Reset failed: {str(e)}"}
        )


@app.post("/step")
def step(req: StepRequest):
    """
    Execute a clinical action in the ward round.
    
    Accepts FLAT request body:
    {
        "action_type": "ask_nurse",
        "patient_id": "P001"
    }
    """
    # Debug logging
    print(f"[STEP] Received: action_type={req.action_type}, patient_id={req.patient_id}")
    print(f"[STEP] State before: started={env._state.started}, step_count={env._state.step_count}")
    
    # Validate environment is started
    if not env._state.started:
        raise HTTPException(
            status_code=400, 
            detail="Environment not started. Call POST /reset first."
        )
    
    # Validate patient_id exists
    valid_patient_ids = [p.id for p in env._state.patients]
    if req.patient_id not in valid_patient_ids:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid patient_id: '{req.patient_id}'. Valid IDs: {valid_patient_ids}"
        )
    
    try:
        # Create Action object from flat request
        action = Action(
            action_type=req.action_type,
            patient_id=req.patient_id,
            content=req.content or "",
            reason=req.reason,
        )
        
        print(f"[STEP] Created Action: {action}")
        
        # Execute step
        obs = env.step(action)
        
        # Extract values
        obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else dict(obs)
        reward = float(obs.reward) if obs.reward is not None else 0.0
        done = bool(obs.done) if hasattr(obs, 'done') else False
        
        print(f"[STEP] Result: reward={reward}, done={done}")
        
        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STEP ERROR] {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Step execution failed: {str(e)}", "traceback": traceback.format_exc()}
        )


@app.get("/metadata")
def get_metadata():
    """Environment metadata."""
    return {
        "name": "WardRound-Env",
        "description": "Multi-agent hospital ward-round RL environment",
        "version": "2.0.0",
        "author": "Team WardRound",
        "theme": "Theme #1 - Multi-Agent Interactions",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
