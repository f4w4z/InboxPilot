---
title: InboxPilot
emoji: 📬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# InboxPilot

InboxPilot is a real-world OpenEnv-compatible AI environment for email triage. The environment simulates a human inbox workflow where an AI agent must classify incoming emails, prioritize them, draft safe and useful replies, and avoid incorrect actions.

## Overview

- **Lightweight**: Uses a clean Python-only implementation with a FastAPI wrapper.
- **Tasks**:
  1. *Easy*: spam / non-spam classification
  2. *Medium*: classify + draft reply
  3. *Hard*: full multi-step inbox handling with prioritization, categorization, and reply generation
- **Scoring**: Deterministic grader returning a StepReward containing `reward`, `progress_score`, and `explanation`.

## Project Structure

- `app.py`: FastAPI server wrapper exposing OpenEnv-style generic endpoints.
- `env.py`: Contains `InboxPilotEnv` logic.
- `models.py`: Strongly-typed Pydantic schemas defining `Observation`, `Action`, `StepReward`, `EnvironmentState`.
- `tasks.py`: Definitions for Easy, Medium, and Hard task episodes.
- `inference.py`: Baseline agent execution loop using OpenAI's client.

## API Endpoints

- `GET /` - Health check.
- `POST /reset` - Resets the env for a given task ID. Returns `Observation`.
- `POST /step` - Passes an `Action` to the env. Returns `Observation`, `StepReward`, `is_done`.
- `GET /state` - Access internal `EnvironmentState`.

## Running the API

You can run the environment natively or via Docker.

### Native
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Docker
```bash
docker build -t inboxpilot .
docker run -p 8000:8000 inboxpilot
```

## Running the Agent

Provide your OpenAI API key and start the API, then run the inference loop.
```bash
export OPENAI_API_KEY="your-api-key"
# export API_BASE_URL="http://localhost:8000" (default)
python inference.py
```
