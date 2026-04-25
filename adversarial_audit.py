import random
import json
import asyncio
from models import Action
from server.environment import WardRoundEnvironment

async def run_audit():
    env = WardRoundEnvironment()
    
    print("="*60)
    print("ADVERSARIAL AUDIT: BRUTAL SYSTEMS TESTING")
    print("="*60)

    # --- PHASE 1: CHEATING STRATEGIES ---
    print("\n[TEST 1] Fixed Policy: (vitals -> labs -> treat) vs ALL Consultants")
    # A fixed policy that doesn't adapt to consultant types
    survival_fixed = 0
    total_eps = 50
    for i in range(total_eps):
        obs = env.reset(seed=i)
        p_id = obs.current_patient.id
        # Step 1: vitals
        obs = env.step(Action(action_type="ask_nurse", patient_id=p_id))
        # Step 2: labs
        obs = env.step(Action(action_type="request_test", patient_id=p_id))
        # Step 3: ask (to get approval)
        obs = env.step(Action(action_type="ask_consultant", patient_id=p_id))
        # Step 4: treat
        obs = env.step(Action(action_type="decide_treatment", patient_id=p_id))
        if obs.metadata.get('grader_score', 0) > 0.5: # Simple success check
            survival_fixed += 1
    print(f"  Fixed Policy Success Rate: {survival_fixed/total_eps*100:.1f}%")
    print("  Verdict: If high, system is too mechanical. (Expect < 100% since Risk-Averse might reject)")

    print("\n[TEST 2] Ignore Agents: Blind Treatment at Step 1")
    survival_ignore = 0
    for i in range(total_eps):
        obs = env.reset(seed=i+100)
        obs = env.step(Action(action_type="decide_treatment", patient_id=obs.current_patient.id))
        if not obs.metadata.get('grader_score', 0) < 0.1: # If it's not a failure
             survival_ignore += 1
    print(f"  Blind Tactic Success Rate: {survival_ignore/total_eps*100:.1f}%")
    print("  Verdict: Should be ~0% because of Approval Requirement.")

    # --- PHASE 2 & 5: EDGE CASE ATTACKS ---
    print("\n[TEST 3] Spamming 'reassure_patient' (Old Exploit Check)")
    obs = env.reset(seed=42)
    p_id = obs.current_patient.id
    total_r = 0
    for _ in range(10):
        obs = env.step(Action(action_type="reassure_patient", patient_id=p_id))
        total_r += (obs.reward or 0)
    print(f"  Total Reward for 10x reassurance: {total_r:+.2f}")
    print("  Verdict: Should be negative due to -0.01 step penalty outweighing gains or lack of gain.")

    print("\n[TEST 4] Acting on Dead Patients")
    obs = env.reset(seed=7)
    p_id = obs.current_patient.id
    # Kill the patient via unauthorized treatment
    obs = env.step(Action(action_type="decide_treatment", patient_id=p_id))
    print(f"  Status after kill: {obs.last_feedback}")
    # Try to treat again
    obs_after = env.step(Action(action_type="decide_treatment", patient_id=p_id))
    print(f"  Reward for acting on dead: {obs_after.reward:+.4f}")
    print("  Verdict: Should be penalty and no state change.")

    # --- PHASE 4: RANDOM POLICY ---
    print("\n[TEST 5] Random Policy Benchmark (100 episodes)")
    random_survival = 0
    random_returns = []
    for i in range(100):
        obs = env.reset(seed=i+500)
        done = False
        ep_ret = 0
        while not done:
            p_id = obs.current_patient.id
            act_type = random.choice(['ask_nurse', 'request_test', 'ask_consultant', 'decide_treatment', 'reassure_patient'])
            obs = env.step(Action(action_type=act_type, patient_id=p_id))
            ep_ret += (obs.reward or 0)
            done = obs.done
        
        score = obs.metadata.get('grader_score', 0)
        if score > 0.5: random_survival += 1
        random_returns.append(ep_ret)
    
    print(f"  Random Survival Rate: {random_survival}%")
    print(f"  Avg Random Return: {sum(random_returns)/100:.4f}")
    print("  Verdict: Target 1-5%.")

if __name__ == "__main__":
    asyncio.run(run_audit())
