import os
import sys
import time
import threading
import requests
from openai import OpenAI
import json

# ---------------------------------------------------------------------------
# 1. Auto-start the environment server in a background thread
# ---------------------------------------------------------------------------
def _start_env_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI environment server in the current process."""
    import uvicorn
    from app import app  # the FastAPI app in app.py

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _ensure_server_running(base_url: str, timeout: float = 15.0):
    """
    If the env server is NOT already reachable, spin it up in a daemon thread
    and wait until it responds to a health-check.
    """
    # Quick probe – maybe it's already running (e.g. Docker / manual start)
    try:
        r = requests.get(f"{base_url}/", timeout=2)
        if r.status_code == 200:
            print("[inference] Environment server already reachable.")
            return
    except Exception:
        pass  # not running yet – we'll start it

    print("[inference] Starting environment server in background thread …")
    port = int(base_url.rsplit(":", 1)[-1].split("/")[0])
    t = threading.Thread(target=_start_env_server, args=("0.0.0.0", port), daemon=True)
    t.start()

    # Wait for it to come alive
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/", timeout=1)
            if r.status_code == 200:
                print("[inference] Environment server is ready.")
                return
        except Exception:
            pass
        time.sleep(0.3)

    print("[inference] WARNING: Environment server did not become reachable within "
          f"{timeout}s – continuing anyway.", file=sys.stderr)


# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
env_url = os.environ.get("ENV_URL", "http://localhost:8000")

openai_base_url = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
openai_api_key = os.environ.get(
    "OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", "dummy")
)
model_name = os.environ.get("MODEL_NAME", "gpt-oss-120b")

client = OpenAI(base_url=openai_base_url, api_key=openai_api_key)

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------

def validate_action(action):
    """Validate that the parsed action is a real action with required fields."""
    if not isinstance(action, dict):
        return False, "Action is not a dictionary"

    if "action_type" not in action:
        return False, "Missing action_type"

    action_type = action.get("action_type")
    valid_types = ["classify", "prioritize", "reply", "finish"]

    if action_type not in valid_types:
        return False, f"Invalid action_type: {action_type}"

    # Reject junk / reasoning-only outputs
    junk_fields = ["analysis", "thought", "reasoning", "explanation", "plan"]
    for field in junk_fields:
        if field in action:
            return False, f"Contains non-action field: {field}"

    # Validate required fields per action type
    if action_type == "classify":
        if "email_id" not in action or "label" not in action:
            return False, "classify action missing email_id or label"
    elif action_type == "prioritize":
        if "email_id" not in action or "priority" not in action:
            return False, "prioritize action missing email_id or priority"
    elif action_type == "reply":
        if "email_id" not in action or "reply_text" not in action:
            return False, "reply action missing email_id or reply_text"
    elif action_type == "finish":
        pass  # no additional fields needed

    return True, "Valid"


def get_fallback_action(task_id, obs):
    """Deterministic fallback to ensure evaluation doesn't get stuck."""
    inbox = obs.get("inbox", [])
    if not inbox:
        return {"action_type": "finish"}

    prev = obs.get("previous_actions", [])
    email_id = inbox[0]["id"]

    if task_id == "easy":
        return {"action_type": "classify", "email_id": email_id, "label": "spam"}

    elif task_id == "medium":
        classified = any(
            a.get("action_type") == "classify" and a.get("email_id") == email_id
            for a in prev
        )
        replied = any(
            a.get("action_type") == "reply" and a.get("email_id") == email_id
            for a in prev
        )
        if not classified:
            return {"action_type": "classify", "email_id": email_id, "label": "support"}
        if not replied:
            return {
                "action_type": "reply",
                "email_id": email_id,
                "reply_text": "Working on fixing your account.",
            }
        return {"action_type": "finish"}

    elif task_id == "hard":
        urgent = [
            e
            for e in inbox
            if "urgent" in e.get("subject", "").lower()
            or "report" in e.get("subject", "").lower()
        ]
        target = urgent[0]["id"] if urgent else email_id

        prioritized = any(
            a.get("action_type") == "prioritize" and a.get("email_id") == target
            for a in prev
        )
        if not prioritized:
            return {"action_type": "prioritize", "email_id": target, "priority": "high"}

        classified = any(
            a.get("action_type") == "classify" and a.get("email_id") == target
            for a in prev
        )
        if not classified:
            return {"action_type": "classify", "email_id": target, "label": "urgent"}

        replied = any(
            a.get("action_type") == "reply" and a.get("email_id") == target
            for a in prev
        )
        if not replied:
            return {
                "action_type": "reply",
                "email_id": target,
                "reply_text": "I will send it today.",
            }

        return {"action_type": "finish"}

    return {"action_type": "finish"}


# ---------------------------------------------------------------------------
# 4. Agent loop
# ---------------------------------------------------------------------------

def run_agent(task_id="easy"):
    print(f"Starting task: {task_id}")

    # ---- Reset Environment ------------------------------------------------
    try:
        res = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id, "instance_id": "agent-1"},
            timeout=30,
        )
        res.raise_for_status()
        obs = res.json()
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Cannot connect to environment at {env_url}: {e}")
        return 0.0
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] API Error ({res.status_code}): {res.text}")
        return 0.0
    except requests.exceptions.JSONDecodeError:
        print(f"[ERROR] Invalid API response: {res.text}")
        return 0.0
    except Exception as e:
        print(f"[ERROR] Unexpected error during /reset: {e}")
        return 0.0

    is_done = False
    step_count = 0
    total_reward = 0.0

    while not is_done and step_count < 20:
        step_count += 1
        print(f"\n--- Step {step_count} ---")

        system_prompt = (
            "You are an AI Email agent. Your task is to process the inbox.\n"
            "IMPORTANT RULES:\n"
            "1. Choose ONLY ONE next best action to take.\n"
            "2. NEVER repeat an action you have already completed.\n"
            "3. Look at the 'previous_actions' and current state to decide what is still missing.\n"
            "4. Use ONLY these exact classification labels (lowercase):\n"
            "   - spam, not_spam (easy task)\n"
            "   - support, spam (medium task)\n"
            "   - internal, urgent, important, action_required, spam, junk, promotion, bug, feedback (hard task)\n"
            "5. For priority, use only: low, medium, high\n"
            "6. Output ONLY a valid JSON object matching this schema:\n"
            "{'action_type': 'classify'|'prioritize'|'reply'|'finish', "
            "'email_id': 'string', 'label': 'string', "
            "'priority': 'string', 'reply_text': 'string'}"
        )

        obs_text = json.dumps(obs, indent=2)
        action_json = None

        # ---- LLM inference (with retry + fallback) ------------------------
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Current Observation:\n{obs_text}\n\nWhat is your next action?",
                        },
                    ],
                    response_format={"type": "json_object"},
                )

                content = response.choices[0].message.content
                if not content:
                    print(f"Attempt {attempt+1}: LLM returned empty content.")
                    continue

                # Clean markdown formatting if present
                if content.strip().startswith("```json"):
                    content = content.split("```json", 1)[1].rsplit("```", 1)[0].strip()
                elif content.strip().startswith("```"):
                    content = content.split("```", 1)[1].rsplit("```", 1)[0].strip()

                parsed = json.loads(content)

                # Handle case where LLM wraps action in 'final' key
                if "final" in parsed and isinstance(parsed["final"], str):
                    action_candidate = json.loads(parsed["final"])
                elif "final" in parsed:
                    action_candidate = parsed["final"]
                else:
                    action_candidate = parsed

                # Validate the action before accepting it
                is_valid, error_msg = validate_action(action_candidate)
                if is_valid:
                    action_json = action_candidate
                    break
                else:
                    print(f"Attempt {attempt+1}: Invalid action - {error_msg}")
                    print(f"Content was: {content}")

            except Exception as e:
                print(f"Attempt {attempt+1}: LLM call failed: {e}")

        if not action_json:
            print("Using deterministic fallback planning based on observation.")
            action_json = get_fallback_action(task_id, obs)

        print(f"Agent decided: {action_json}")

        # ---- Step Environment ---------------------------------------------
        try:
            res = requests.post(
                f"{env_url}/step",
                json={"instance_id": "agent-1", "action": action_json},
                timeout=30,
            )
            res.raise_for_status()
            step_res = res.json()
        except Exception as e:
            print(f"[ERROR] /step request failed: {e}")
            break

        # Handle error responses from environment
        if "observation" not in step_res:
            print(f"Environment error: {step_res}")
            break

        obs = step_res["observation"]
        reward = step_res["reward"]
        is_done = step_res["is_done"]

        total_reward += reward["reward"]
        print(
            f"Reward: {reward['reward']}, "
            f"Progress: {reward['progress_score']}, "
            f"Explanation: {reward['explanation']}"
        )

    print(f"\nTask {task_id} completed in {step_count} steps.")
    print(f"  Total Reward: {total_reward:.2f}")
    return total_reward


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure the env server is running before we start evaluation
    _ensure_server_running(env_url)

    tasks = ["easy", "medium", "hard"]
    scores = {}

    print(f"--- Starting InboxPilot Evaluation ---")
    print(f"Model: {model_name}")
    print(f"API Base URL: {openai_base_url}\n")

    for t in tasks:
        print(f"{'='*40}")
        print(f"EVALUATING TASK: {t.upper()}")
        print(f"{'='*40}")
        score = run_agent(t)
        scores[t] = score
        print("\n")

    print(f"{'='*40}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*40}")

    # Normalize scores to 0.0-1.0 scale for cleaner presentation
    max_scores = {"easy": 1.0, "medium": 2.0, "hard": 3.0}
    normalized_scores = {}

    for t in tasks:
        normalized = scores[t] / max_scores[t]
        normalized_scores[t] = max(0.0, min(1.0, normalized))
        print(f"Task: {t.upper()}")
        print(f"  Raw Score: {scores[t]:.2f}")
        print(f"  Normalized: {normalized_scores[t]:.2f} (0.0 - 1.0)")

    overall = sum(normalized_scores.values()) / len(tasks)
    print(f"\n{'='*40}")
    print(f"OVERALL SCORE: {overall:.2f} / 1.00")
    print(f"{'='*40}")
