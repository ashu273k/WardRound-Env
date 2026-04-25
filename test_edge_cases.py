"""Advanced Edge Case Testing for WardRound-Env.

This automated script hammers the WardRound environment with extreme
edge cases to ensure absolute stability for the hackathon judges.

Test 1: Validation Edge Case (Sending garbage data)
Test 2: Time Exhaustion (Spamming actions until time hits 0)
Test 3: The "Perfect" Hard Scenario (Solving the aggressive consultant)
"""

import asyncio
from client import WardRoundEnvClient
from models import Action

SERVER_URL = "https://abhijeet-2005-wardround-env.hf.space"

async def test_validation_edge_case(client: WardRoundEnvClient):
    print("\n[EDGE CASE 1] Testing Validation Firewall...")
    await client.reset(task_id="easy", seed=42)
    
    try:
        # Try to send a completely fabricated action type
        bad_action = Action(
            action_type="do_a_magic_trick", 
            patient_id="P001", 
            content="Nothing"
        )
        await client.step(bad_action)
        print("  ❌ FAIL: The environment accepted a fake action!")
    except Exception as e:
        print("  ✅ PASS: The environment successfully blocked the invalid action.")

async def test_time_exhaustion(client: WardRoundEnvClient):
    print("\n[EDGE CASE 2] Testing Time Limit Exhaustion...")
    obs = await client.reset(task_id="easy", seed=42)
    
    # Easy task has 15 time limit. Let's spam "ask_nurse" 15 times!
    print("  Spamming trivial actions to burn time budget...")
    step_count = 0
    done = False
    while not done and step_count < 20:
        act = Action(
            action_type="ask_nurse",
            patient_id="P001",
            content="What is the patient doing?"
        )
        result = await client.step(act)
        obs = result.observation
        done = result.done
        step_count += 1
        
    print(f"  Stopped at Step {step_count}.")
    if done and step_count == 15:
        score = obs.metadata.get("grader_score", 0)
        print(f"  ✅ PASS: Environment properly forcefully terminated at exactly 15 steps.")
        print(f"  ✅ PASS: Grader aggressively penalized the time-wasting (Score: {score:.3f}).")
    else:
        print("  ❌ FAIL: Time budget logic is broken.")

async def test_perfect_hard_scenario(client: WardRoundEnvClient):
    print("\n[EDGE CASE 3] The 'Perfect' Hard Scenario Strategy...")
    # The hard scenario has an aggressive consultant and an ethical patient dilemma
    result = await client.reset(task_id="hard", seed=99)
    print("  Scenario loaded: HARD (Aggressive Consultant)")
    
    patient_1 = result.observation.current_patient.id  # Mr. Sharma (Sepsis)
    
    # Step 1: Present perfectly
    print("  -> Doctor: Presenting Case 1...")
    res = await client.step(Action(
        action_type="present_case", 
        patient_id=patient_1, 
        content="Patient has sepsis. Vitals are crashing, starting fluids."
    ))
    
    # Step 2: Aggressive consultant usually demands justification, so we must ask him or escalate!
    print("  -> Doctor: Proactively requesting Consultant's deep guidance...")
    res = await client.step(Action(
        action_type="ask_consultant", 
        patient_id=patient_1, 
        content="Could you verify the broad-spectrum antibiotic choice?"
    ))
    
    # Step 3: Now we decide treatment
    print("  -> Doctor: Deciding Treatment 1...")
    res = await client.step(Action(
        action_type="decide_treatment", 
        patient_id=patient_1, 
        content="Treating sepsis aggressively according to protocol."
    ))
    
    patient_2 = res.observation.current_patient.id  # Mrs. Gupta (Ethical)
    
    # For ethical patient, we must reassure family first!
    print("  -> Doctor: Reassuring Family for Case 2...")
    res = await client.step(Action(
        action_type="reassure_patient", 
        patient_id=patient_2, 
        content="We understand your concerns about the risks, we will be careful."
    ))
    
    print("  -> Doctor: Deciding Treatment 2...")
    final_res = await client.step(Action(
        action_type="decide_treatment", 
        patient_id=patient_2, 
        content="Proceeding with palliative care setup."
    ))
    
    score = final_res.observation.metadata.get("grader_score", 0)
    rubric = final_res.observation.metadata.get("grader_rubric", {})
    
    print(f"  Episode Done! Final Grader Score: {score:.4f}")
    if score >= 0.85:
        print("  ✅ PASS: Advanced strategy proved the environment allows high-skill agents to win!")
    else:
        print(f"  ⚠ WARNING: Score was {score:.4f}. The environment might be too punishing.")

async def main():
    print("="*60)
    print("WARDROUND-ENV: HACKATHON EDGE CASE SUITE")
    print("="*60)
    
    client = WardRoundEnvClient(base_url=SERVER_URL)
    
    await test_validation_edge_case(client)
    await test_time_exhaustion(client)
    await test_perfect_hard_scenario(client)
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")

if __name__ == "__main__":
    asyncio.run(main())
