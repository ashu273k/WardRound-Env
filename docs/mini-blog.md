# WardRound-Env Mini-Blog Draft

## What we built

WardRound-Env is a multi-agent hospital ward-round simulator built for OpenEnv Hackathon Round 2 (Theme #1).
The learning agent is a Junior Doctor policy that must communicate and coordinate with three scripted agents:
Senior Consultant, Nurse, and Patient/Family.

## Why this is a good multi-agent benchmark

- Multiple stakeholders with different incentives
- Conflicting opinions under time pressure
- Mix of communication quality and actionable decisions
- Deterministic episodes for reproducible training/evaluation

## Environment design

- OpenEnv-compliant `reset`, `step`, and `state`
- Typed Pydantic models for actions, observations, and state
- Three task levels:
  - easy: cooperative baseline
  - medium: conflicting consultant signals
  - hard: ethical pressure and urgency

## Training and evaluation plan

- Use shaped rewards for dense feedback
- Add deterministic grader score in `[0.0, 1.0]`
- Compare baseline inference against trained policy curves

## What we learned

- Strong interfaces (`models.py` + `openenv.yaml`) reduce integration issues
- Deterministic scenario files simplify debugging and regression testing
- Explicit multi-agent responses improve interpretability of policy behavior
