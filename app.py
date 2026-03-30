from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from env import InboxPilotEnv
from models import EnvironmentState, Observation, Action, StepReward

app = FastAPI(title="InboxPilot OpenEnv API")

# Simple in-memory store for active environments
envs: Dict[str, InboxPilotEnv] = {}

class ResetRequest(BaseModel):
    task_id: str = "easy"
    instance_id: str = "default"

class StepRequest(BaseModel):
    instance_id: str = "default"
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: StepReward
    is_done: bool
    info: Dict[str, Any]

@app.get("/")
def root():
    return {"status": "ok", "env": "InboxPilot"}

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest):
    try:
        env = InboxPilotEnv(task_id=req.task_id)
        envs[req.instance_id] = env
        return env.reset()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step_env(req: StepRequest):
    if req.instance_id not in envs:
        raise HTTPException(status_code=404, detail="Environment instance not found. Call /reset first.")
    
    env = envs[req.instance_id]
    obs, reward, is_done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, is_done=is_done, info=info)

@app.get("/state", response_model=EnvironmentState)
def get_state(instance_id: str = "default"):
    if instance_id not in envs:
        raise HTTPException(status_code=404, detail="Environment instance not found.")
    return envs[instance_id].state()
