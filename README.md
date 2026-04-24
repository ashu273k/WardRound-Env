# WardRound-Env

A multi-agent hospital ward-round simulation environment where an AI **Junior Doctor** agent learns to lead effective morning ward rounds by interacting with other agents (Senior Consultant, Nurse, Patient/Family).

> **Status:** This repository currently contains the project design in [ROADMAP.md](ROADMAP.md). The OpenEnv environment implementation, training scripts, and Hugging Face Space will be added next.

## Why this exists
Ward rounds are a high-stakes, time-constrained coordination problem: you must communicate clearly, negotiate disagreements, prioritize tasks, and make decisions under pressure. WardRound-Env is designed to make those skills measurable and trainable.

This project is built for **Theme #1 — Multi-Agent Interactions**.

## What you will build
WardRound-Env will provide:

- An **OpenEnv-compliant** environment (`reset/step/state`, typed models, `openenv.yaml`)
- A scripted multi-agent cast with distinct incentives and personalities
- **Three difficulty tiers** (easy/medium/hard) with increasing conflict and time pressure
- A **shaped reward function** (dense, partial progress signals)
- A **deterministic grader** that outputs a final score in **[0.0, 1.0]**
- A minimal **training script** (Unsloth or Hugging Face TRL) demonstrating reward improvement
- A **Hugging Face Space** (Docker) + short demo video + mini-blog

## Planned repository structure

This is the intended layout once implementation lands:

```
.
├─ openenv.yaml
├─ README.md
├─ ROADMAP.md
├─ wardround_env/
│  ├─ __init__.py
│  ├─ environment.py        # OpenEnv: reset/step/state
│  ├─ models.py             # Typed Action/Observation/State/Reward
│  ├─ agents/
│  │  ├─ consultant.py      # Scripted Senior Consultant
│  │  ├─ nurse.py           # Scripted Nurse
│  │  └─ patient_family.py  # Scripted Patient/Family
│  ├─ scenarios/
│  │  ├─ easy/
│  │  ├─ medium/
│  │  └─ hard/
│  ├─ grader/
│  │  ├─ rubric.py          # Deterministic rubric + subscores
│  │  └─ grader.py          # score in [0, 1]
│  └─ demo/
│     └─ run_episode.py
├─ train/
│  ├─ train_trl.py           # TRL/Unsloth training entry
│  └─ eval.py                # Batch evaluation + plots
├─ app/
│  └─ app.py                 # HF Space UI (planned)
├─ Dockerfile
└─ requirements.txt
```

If you prefer a different layout (package name, script locations), we can adjust it before coding starts.

## Roles (agents) in the simulation

### 1) Junior Doctor (learning agent)
The policy you train. Responsibilities:

- Present patient cases clearly and concisely
- Answer questions from the Senior Consultant
- Coordinate with the Nurse for immediate tasks (e.g., labs, IV access, medications)
- Respond to patient/family concerns with empathy and clarity
- Make treatment decisions while managing disagreement and time pressure

### 2) Senior Consultant (scripted)
Rule-based agent that:

- Probes for missing information (red flags, differentials, safety checks)
- Challenges plans and asks follow-up questions
- In harder settings, may strongly disagree to force negotiation

### 3) Nurse (scripted)
Rule-based agent that:

- Requests actionable orders and clarifications
- Flags practical constraints (time, staffing, feasibility)
- Tracks whether tasks were actually ordered (not just mentioned)

### 4) Patient / Family (scripted)
Rule-based agent that:

- Asks questions, expresses concerns, and may refuse parts of the plan
- In hard settings, may create an ethical conflict (capacity, consent, risk trade-offs)

## Difficulty levels

- **Easy:** stable patients, cooperative team, minimal interruptions
- **Medium:** one conflicting clinical opinion + mild time pressure
- **Hard:** ethical dilemma + strong disagreement (consultant vs family) + tight time constraints

## Environment design (OpenEnv)

WardRound-Env is intended to be an **interactive, turn-based** environment.

### Episode structure (high level)
Each episode simulates a single ward round consisting of multiple “round events”, e.g.

1. Junior Doctor presents the current patient
2. Consultant questions / challenges
3. Nurse requests actionable items
4. Patient/family raises concerns
5. Junior Doctor decides and issues orders
6. Time advances; tasks either get done or are delayed

The episode ends when:

- The ward round is completed (all patients addressed), or
- Time runs out, or
- A safety-critical failure condition is triggered (scenario-dependent)

### OpenEnv deliverables (checklist)

The OpenEnv submission will include:

- `openenv.yaml` describing the environment, schemas, and entrypoints
- Typed models for `Action`, `Observation`, `State`, and `Reward`
- `reset()` returning the initial observation + state
- `step(action)` returning next observation + reward + done + info/metadata
- `state` serialization (for debugging and grader determinism)

### Determinism and reproducibility
The environment and grader are designed to be deterministic given:

- `seed`
- scenario id
- difficulty

This is important for stable training curves and for a robust leaderboard-style evaluation.

## Observations, actions, and state (proposed contract)

> These are the **intended** interfaces. Exact schemas will be finalized in `models.py` when implementation begins.

### Observation (what the agent sees)
A structured observation will include:

- Current patient summary (age/sex, diagnosis, vitals, key labs/imaging)
- Current open issues / tasks outstanding
- Conversation context / last utterances from consultant, nurse, and patient/family
- Time remaining / time pressure indicators
- Any scenario-specific constraints (e.g., consent requirements)

The environment may also provide a compact “checklist view” (what has been covered vs missing).

### Action (what the agent does)
The action should be a structured command with two channels:

1. **Communication**: what the Junior Doctor says (presentation + answers)
2. **Orders**: discrete orders (labs, meds, consults, imaging) and task assignments

This separation helps the grader distinguish “good talk” from “real orders”.

### State (internal)
Internal state tracks:

- Patient trajectory (improving/stable/worsening)
- Pending tasks and their completion times
- Team alignment (agreement level)
- Safety flags / guideline constraints
- Whether key info was presented or omitted

## Reward design (shaped)
The reward will provide **partial credit** throughout the episode.

Examples of shaped reward components:

- **Presentation quality**: completeness, structure, brevity
- **Clinical reasoning**: appropriate differential/plan given scenario facts
- **Coordination**: timely actionable tasks for nursing staff
- **Empathy & communication**: addressing patient/family concerns
- **Safety**: red-flag recognition, contraindication avoidance
- **Time efficiency**: finishing within time limits
- **Conflict handling**: de-escalation, negotiation, shared decision-making

> The final evaluation will also include a deterministic grader score; reward shaping is for training signal.

## Deterministic grader (final score 0.0–1.0)
A deterministic grader will map the trajectory of an episode to a single score in **[0, 1]**.

Proposed scoring axes (weighted):

- **Patient outcome**: safety + trajectory (e.g., avoided deterioration)
- **Plan correctness**: scenario-specific required actions completed
- **Team agreement**: reduced friction / resolved conflicts
- **Time efficiency**: completed round within constraints
- **Communication quality**: key concerns addressed

The grader must be:

- deterministic (same inputs → same score)
- transparent (returns sub-scores / rubric breakdown)
- strict about actionable vs non-actionable behavior

## Scenarios (tasks)
Each difficulty tier will include scenario templates (patient cases + scripted agent behavior).

- **Easy scenarios** emphasize clear presentation and routine tasking.
- **Medium scenarios** introduce one conflicting opinion and require negotiation.
- **Hard scenarios** introduce an ethical dilemma and strong disagreement.

Scenarios should be authored as data (YAML/JSON) so the environment can load them consistently.

## Training (Unsloth / Hugging Face TRL)
The training deliverable is a minimal script that demonstrates:

- baseline policy score distribution
- trained policy score distribution
- reward curves / grader improvement over time

The training loop will:

- sample scenarios
- run rollouts in the environment
- compute shaped reward + final grader score
- optimize the policy using TRL (or Unsloth-backed training)

## Running the project (planned)

> Commands will be updated once code lands.

### Local setup
- Create a virtual environment: `python -m venv .venv`
- Activate it (bash/zsh): `source .venv/bin/activate`
- Install dependencies (planned): `pip install -r requirements.txt`

### Run a single episode (planned)
You will be able to run something like:

- `python -m wardround_env.demo.run_episode --difficulty easy --seed 0`

### Train (planned)
You will be able to run something like:

- `python -m train.train_trl --difficulty medium --steps 20000`

## Deployment (Hugging Face Spaces + Docker)
The project will be packaged with Docker for a Hugging Face Space.

Planned deliverables:

- `Dockerfile`
- `openenv.yaml`
- App entry (Gradio or similar) that runs an episode and shows:
  - transcript
  - orders
  - rubric breakdown
  - final score

## Definition of done (Round 2)

This is the minimum “done” bar we’re targeting:

- **Environment**: OpenEnv-compliant environment runs locally with at least one scenario per difficulty
- **Agents**: Consultant, Nurse, and Patient/Family scripted policies produce realistic, consistent behavior
- **Reward**: shaped reward emits partial-credit signals throughout the episode
- **Grader**: deterministic scorer returns score in [0, 1] + rubric breakdown
- **Training**: reproducible training run shows score improvement vs baseline
- **Deployment**: Hugging Face Space builds from Docker and runs the demo end-to-end
- **Demo**: short video + mini-blog with results and failure cases

## Evaluation & demo deliverables

- Demo video (< 2 minutes)
- Mini-blog post describing:
  - the multi-agent design
  - the grader rubric
  - training results
  - failure cases and how the trained agent improves

## Team workflow

- **Ram**: environment skeleton, OpenEnv compliance, deployment
- **Ashu**: reward shaping, deterministic grader, training + plots
- **Abhijeet**: scripted multi-agent behaviors, scenario authoring (easy/medium/hard)

Suggested collaboration:

- Keep scenario files small and testable
- Add a deterministic “golden episode” regression test for the grader
- Track all interface changes in a single place (models + `openenv.yaml`)

## Contributing

This repo is being built fast (48-hour sprint). To keep integration smooth:

- **One source of truth for interfaces:** `models.py` + `openenv.yaml`
- **Scenario changes are data-first:** add/modify scenario files rather than hard-coding in logic
- **Grader changes must stay deterministic:** avoid external calls, randomness, or time-based logic
- **Add a tiny regression test whenever possible:** one “golden episode” with fixed seed and expected rubric breakdown

Ownership (initial):

- Environment skeleton + OpenEnv compliance + Docker/Space: Ram
- Reward shaping + deterministic grader + training runs: Ashu
- Scripted agents + scenario authoring (easy/medium/hard): Abhijeet

## Medical disclaimer
This is a simulated environment for research/education in multi-agent interaction and reinforcement learning. It is **not** medical advice and must not be used for real clinical decision-making.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for the project plan, roles, and 48-hour timeline.
