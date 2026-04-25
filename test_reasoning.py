import asyncio
import json
from models import Action
from server.environment import WardRoundEnvironment

async def test_multi_agent_reasoning():
    env = WardRoundEnvironment()
    
    print("=== SCENARIO 1: INFERRING CONSULTANT PERSONALITY ===")
    obs = env.reset(task_id="hard", seed=42)
    p_id = obs.current_patient.id
    
    # 1. Ask consultant early to gauge personality
    action = Action(action_type="ask_consultant", patient_id=p_id, content="What do you think?")
    obs = env.step(action)
    print(f"Consultant Feedback: {obs.consultant_opinion}")
    
    # 2. Reassure family if they are upset
    if obs.family_concern:
        print(f"Family Concern: {obs.family_concern}")
        action = Action(action_type="reassure_patient", patient_id=p_id)
        obs = env.step(action)
        print(f"Feedback after reassurance: {obs.last_feedback}")

    # 3. Follow causal path
    print("\n--- GATHERING DATA ---")
    obs = env.step(Action(action_type="ask_nurse", patient_id=p_id))
    print(f"Nurse: {obs.nurse_status}")
    
    obs = env.step(Action(action_type="request_test", patient_id=p_id))
    print(f"Nurse: {obs.nurse_status}")
    
    # 4. Check approval again
    obs = env.step(Action(action_type="ask_consultant", patient_id=p_id))
    print(f"Consultant Final: {obs.consultant_opinion}")
    
    # 5. Decide treatment
    obs = env.step(Action(action_type="decide_treatment", patient_id=p_id))
    print(f"Treatment Result: {obs.last_feedback}")
    print(f"Episode Done: {obs.done}, Reward: {obs.reward}")

    print("\n=== SCENARIO 2: EMERGENCY OVERRIDE PILOT ===")
    # Reset until we get a crashing patient OR simulate deterioration
    obs = env.reset(task_id="hard", seed=99)
    p_id = obs.current_patient.id
    
    print("Waiting for patient to deteriorate...")
    for _ in range(5): # Burn time to lower stability
        obs = env.step(Action(action_type="present_case", patient_id=p_id))
        
    print(f"Recent Feedback: {obs.last_feedback}")
    if "CRASHING" in obs.last_feedback:
        print("Patient is CRASHING! Attempting Emergency Override WITHOUT approval...")
        obs = env.step(Action(action_type="decide_treatment", patient_id=p_id))
        print(f"Result: {obs.last_feedback}")
        print(f"Consultant: {obs.consultant_opinion}")

if __name__ == "__main__":
    asyncio.run(test_multi_agent_reasoning())
