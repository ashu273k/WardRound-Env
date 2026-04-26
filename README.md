---
title: WardRound Env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---
# WardRound-Env

WardRound-Env is an OpenEnv-compatible hospital ward-round simulator for **Theme #1: Multi-Agent Interactions**.
The learning policy plays a Junior Doctor who must coordinate with scripted Consultant, Nurse, and Patient/Family agents.

- **Core OpenEnv environment** implemented (`reset`, `step`, `state`)
- **Advanced Multi-Agent Reasoning**: Integrated hidden Consultant personalities (Conservative, Aggressive, Risk-Averse) and Family emotional states. 
- **Unified Trust System**: Added a `trust_score` variable that bridges social interaction and clinical approval, creating complex decision trade-offs.
- **Scenario-driven tasks**: `easy`, `medium`, and `hard` scenarios with dynamic deterioration.
- **Grader & Validation**: Deterministic reward system and environment validation passed.
- **FastAPI / OpenEnv Server**: Ready for deployment on Hugging Face Spaces.

## Repository layout

```text
.
├─ openenv.yaml
├─ pyproject.toml
├─ models.py
├─ agents.py
├─ client.py
├─ inference.py
├─ scenarios/
│  ├─ easy.json
│  ├─ medium.json
│  └─ hard.json
├─ server/
│  ├─ __init__.py
│  ├─ app.py
│  └─ environment.py
├─ docs/
│  ├─ ROADMAP.md
│  └─ mini-blog.md
└─ Dockerfile
```

## Team roles

- **Ram (Lead):** OpenEnv skeleton, interfaces, deployment, integration
- **Ashu:** reward shaping, deterministic grader, training scripts
- **Abhijeet:** scripted agents and scenario authoring

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run baseline local inference:

```bash
python inference.py --task-id easy --seed 42
```

Run the OpenEnv server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate environment:

```bash
openenv validate .
```

## Task definitions

- **easy:** stable patient, supportive consultant
- **medium:** conflicting clinical pressure, moderate urgency
- **hard:** ethical conflict and stronger time pressure

## Determinism

- Default seed is `42`
- `reset(seed=...)` controls deterministic episode initialization
- Scenario data is loaded from fixed JSON files in `scenarios/`

## Deployment notes

- `openenv.yaml` points to `server.app:app`
- `Dockerfile` starts `uvicorn server.app:app` on port `8000`
- Target platform: Hugging Face Spaces (Docker mode)

## Medical disclaimer

This project is a simulation for research and education only.
It is not medical advice and must not be used for real clinical decisions.

---
Hugging Face Link: [hf](https://huggingface.co/spaces/Abhijeet-2005/WardRound-Env)
