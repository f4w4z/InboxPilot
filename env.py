import copy
from typing import Tuple, Dict, Any
from models import EnvironmentState, Observation, Action, StepReward
from tasks import TASKS

class InboxPilotEnv:
    def __init__(self, task_id: str = "easy"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}")
        self.task_id = task_id
        self.task_data = TASKS[task_id]
        self.reset()

    def reset(self) -> Observation:
        self.inbox = copy.deepcopy(self.task_data["inbox"])
        self.expected_actions = copy.deepcopy(self.task_data["expected_actions"])
        self.previous_actions = []
        self.step_count = 0
        self.is_done = False
        
        return self._get_observation()

    def state(self) -> EnvironmentState:
        current_email = self.inbox[0] if self.inbox else None
        inbox_summary = f"{len(self.inbox)} emails remaining."
        
        return EnvironmentState(
            current_email=current_email,
            inbox_summary=inbox_summary,
            previous_actions=self.previous_actions.copy(),
            task_name=self.task_data["metadata"].task_id,
            step_count=self.step_count
        )

    def _get_observation(self) -> Observation:
        s = self.state()
        return Observation(
            current_email=s.current_email,
            inbox_summary=s.inbox_summary,
            previous_actions=s.previous_actions,
            task_name=s.task_name,
            step_count=s.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, StepReward, bool, Dict[str, Any]]:
        self.step_count += 1
        
        if self.is_done:
            return self._get_observation(), StepReward(reward=0.0, progress_score=1.0, explanation="Environment is already done."), True, {}

        self.previous_actions.append(f"{action.action_type} on {action.email_id}")

        total_expected = len(self.task_data["expected_actions"])
        progress = 1.0 - (len(self.expected_actions) / total_expected) if total_expected > 0 else 0.0

        if action.action_type == "finish":
            self.is_done = True
            if not self.expected_actions:
                return self._get_observation(), StepReward(reward=1.0, progress_score=1.0, explanation="Task completed successfully."), True, {}
            else:
                return self._get_observation(), StepReward(reward=-0.5, progress_score=progress, explanation="Finished prematurely with pending required actions."), True, {}

        # Block non-finish actions if inbox is empty / all expected actions are complete
        if not self.expected_actions or not self.inbox:
            return self._get_observation(), StepReward(
                reward=-0.5,
                progress_score=1.0,
                explanation="No pending emails remain in the inbox. You must use the 'finish' action to complete the task."
            ), False, {}
            
        # Check for actions on already-processed or non-existent emails
        email_in_inbox = any(e.id == action.email_id for e in self.inbox)
        if not email_in_inbox:
            return self._get_observation(), StepReward(
                reward=-0.5,
                progress_score=progress,
                explanation=f"Email '{action.email_id}' is no longer in the inbox or does not exist."
            ), False, {}

        # Matching logic
        matched_idx = -1
        for i, expected in enumerate(self.expected_actions):
            if expected["email_id"] == action.email_id and expected["action_type"] == action.action_type:
                # Check specifics
                match = True
                if expected.get("label"):
                    if isinstance(expected["label"], list):
                        if action.label not in expected["label"]: match = False
                    else:
                        if action.label != expected["label"]: match = False
                if expected.get("priority") and action.priority != expected["priority"]: match = False
                if expected.get("reply_keywords"):
                    text = (action.reply_text or "").lower()
                    if not any(kw.lower() in text for kw in expected["reply_keywords"]):
                        match = False
                
                if match:
                    matched_idx = i
                    break

        if matched_idx >= 0:
            self.expected_actions.pop(matched_idx)
            
            # If the email has no more expected actions, remove it from inbox
            remaining_for_email = any(e["email_id"] == action.email_id for e in self.expected_actions)
            if not remaining_for_email:
                self.inbox = [e for e in self.inbox if e.id != action.email_id]

            new_progress = 1.0 - (len(self.expected_actions) / total_expected) if total_expected > 0 else 0.0
            
            if not self.expected_actions:
                self.is_done = True
                explanation = "Task completed successfully."
            else:
                explanation = "Correct action performed."
            
            reward = StepReward(
                reward=1.0, 
                progress_score=new_progress, 
                explanation=explanation
            )
            return self._get_observation(), reward, self.is_done, {}
        else:
            reward = StepReward(
                reward=-0.1, 
                progress_score=progress, 
                explanation="Action did not match any required pending actions or was incorrect."
            )
            return self._get_observation(), reward, False, {}
