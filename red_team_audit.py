import asyncio
from models import Action
from server.environment import WardRoundEnvironment

async def red_team_judge_audit():
    env = WardRoundEnvironment()
    
    print("="*60)
    print("RED TEAM AUDIT: JUDGE PERSPECTIVE")
    print("="*60)

    # TEST: Can I 'Brute Force' the Checklist?
    # Sequence: ask_nurse -> request_test -> present_case -> ask_consultant
    # This sequence succeeds 100% against Conservative consultants.
    # This means the "Multi-Agent" aspect is just a 4-step unlocking puzzle.
    
    # HYPOTHESIS: If I can save the patient with the SAME sequence every time 
    # regardless of family state or patient age, it's not "Reasoning".
    
    success_count = 0
    seeds = [10, 20, 30] # Different seeds -> different families/personalities
    for s in seeds:
        obs = env.reset(seed=s)
        p_id = obs.current_patient.id
        
        # Fixed "Optimal" checklist
        env.step(Action(action_type="ask_nurse", patient_id=p_id))
        env.step(Action(action_type="request_test", patient_id=p_id))
        env.step(Action(action_type="present_case", patient_id=p_id))
        obs = env.step(Action(action_type="ask_consultant", patient_id=p_id))
        
        if obs.metadata.get('approval_granted', False) or "authorize" in (obs.consultant_opinion or "").lower():
            print(f"Seed {s}: FIXED CHECKLIST WORKED. (Suspected Mechanical Behavior)")
            success_count += 1
        else:
            print(f"Seed {s}: FIXED CHECKLIST FAILED. (Good! Dynamic requirements?)")
            print(f"  Reason: {obs.consultant_opinion}")

    print("\n[VERDICT PREVIEW] If Success Rate is 100%, the environment is engineered, not reasoning-based.")

if __name__ == "__main__":
    asyncio.run(red_team_judge_audit())
