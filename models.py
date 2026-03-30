from pydantic import BaseModel, model_validator
from typing import List, Optional, Literal

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str

class EnvironmentState(BaseModel):
    current_email: Optional[Email] = None
    inbox_summary: str
    previous_actions: List[str]
    task_name: str
    step_count: int

class Observation(BaseModel):
    current_email: Optional[Email] = None
    inbox_summary: str
    previous_actions: List[str]
    task_name: str
    step_count: int

class Action(BaseModel):
    action_type: Literal['classify', 'prioritize', 'reply', 'finish']
    email_id: Optional[str] = None
    label: Optional[str] = None
    priority: Optional[str] = None
    reply_text: Optional[str] = None

    @model_validator(mode='after')
    def check_conditional_fields(self) -> 'Action':
        if self.action_type != 'finish' and not self.email_id:
            raise ValueError(f"email_id is required for action_type '{self.action_type}'")
            
        if self.action_type == 'classify' and not self.label:
            raise ValueError("label is required for action_type 'classify'")
            
        if self.action_type == 'prioritize' and not self.priority:
            raise ValueError("priority is required for action_type 'prioritize'")
            
        if self.action_type == 'reply' and not self.reply_text:
            raise ValueError("reply_text is required for action_type 'reply'")
            
        return self

class StepReward(BaseModel):
    reward: float
    progress_score: float
    explanation: str

class TaskMetadata(BaseModel):
    task_id: str
    complexity: str
    description: str
