import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class PatientHiddenState:
    """Hidden state for true POMDP dynamics."""
    id: str
    name: str
    true_diagnosis: str
    vital_stability: float = 1.0  # Decays over time
    decay_rate: float = 0.05
    known_symptoms: List[str] = field(default_factory=list)
    treated: bool = False
    is_dead: bool = False

    def get_clinical_appearance(self) -> str:
        if self.is_dead:
            return "Patient is unresponsive (Deceased)."
        if self.vital_stability > 0.8:
            return "Patient appears stable and comfortable."
        elif self.vital_stability > 0.4:
            return "Patient is sweating, distressed, and tachycardic."
        else:
            return "Patient is cyanotic, gasping, and crashing!"

class TrueWardRoundEnv:
    """A mathematically rigorous Gym-style RL environment for LLM agents."""

    def __init__(self):
        self.max_time = 20
        self.time_remaining = self.max_time
        self.patients: Dict[str, PatientHiddenState] = {}
        self.step_count = 0

    def reset(self) -> str:
        self.time_remaining = self.max_time
        self.step_count = 0
        
        # Initialize hidden state patients
        self.patients = {
            "P001": PatientHiddenState(
                id="P001", name="Ramesh", true_diagnosis="viral", 
                vital_stability=1.0, decay_rate=0.01,
            ),
            "P002": PatientHiddenState(
                id="P002", name="Gupta", true_diagnosis="sepsis", 
                vital_stability=0.6, decay_rate=0.15, # Fast decay!
            )
        }
        return self._build_observation()

    def _build_observation(self) -> str:
        obs = {
            "time_remaining": self.time_remaining,
            "patients": []
        }
        for p in self.patients.values():
            status = {
                "id": p.id,
                "name": p.name,
                "appearance": p.get_clinical_appearance(),
                "known_symptoms": p.known_symptoms,
                "status": "Treated" if p.treated else "Pending"
            }
            obs["patients"].append(status)
        return json.dumps(obs, indent=2)

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.step_count += 1
        self.time_remaining -= 1
        
        # 1. Base Penalties (Bleed points)
        reward = -0.01  # Step penalty
        
        action_type = action.get("action_type")
        patient_id = action.get("patient_id")
        
        if action_type == "ask_consultant":
            reward -= 0.05
        
        # 2. Action Logic
        if patient_id in self.patients and not self.patients[patient_id].is_dead:
            p = self.patients[patient_id]
            
            if action_type == "request_vitals":
                if p.true_diagnosis == "sepsis":
                    p.known_symptoms.append("BP 80/50, HR 130")
                else:
                    p.known_symptoms.append("Vitals stable")
                    
            elif action_type == "decide_treatment":
                treatment = action.get("content", "").lower()
                p.treated = True
                if p.true_diagnosis == "sepsis" and "antibiotics" not in treatment:
                    p.vital_stability -= 0.5  # Wrong treatment accelerates death
                elif p.true_diagnosis == "sepsis" and "antibiotics" in treatment:
                    p.vital_stability = 1.0  # Saved
                    p.decay_rate = 0.0

        # 3. Cascading Deterioration (Cross-patient dependency)
        for p in self.patients.values():
            if not p.treated and not p.is_dead:
                p.vital_stability -= p.decay_rate
                if p.vital_stability <= 0.0:
                    p.is_dead = True
                    reward -= 1.0  # Severe penalty for patient death

        # 4. Check Termination
        all_treated_or_dead = all(p.treated or p.is_dead for p in self.patients.values())
        done = self.time_remaining <= 0 or all_treated_or_dead

        # 5. Terminal Reward Setup
        if done:
            survived = sum(1 for p in self.patients.values() if not p.is_dead)
            total = len(self.patients)
            success_bonus = (survived / total) * 0.5
            time_bonus = (self.time_remaining / self.max_time) * 0.5
            reward += success_bonus + time_bonus

        return self._build_observation(), reward, done, {"survived": sum(1 for p in self.patients.values() if not p.is_dead)}


# ═══════════════════════════════════════════════════════════════════════════
#  MINIMAL TRAINING LOOP & SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

class MockPolicy:
    """A mock policy representing an LLM or RL agent exploring actions."""
    def act(self, obs_json: str, epsilon: float) -> Dict[str, Any]:
        obs = json.loads(obs_json)
        pending = [p["id"] for p in obs["patients"] if p["status"] == "Pending" and "Deceased" not in p["appearance"]]
        
        if not pending:
            return {"action_type": "wait", "patient_id": None}
            
        target = pending[0]
        # Epsilon-greedy: Randomly guess or do correct triage
        if random.random() < epsilon:
            actions = ["request_vitals", "ask_consultant", "decide_treatment"]
            return {"action_type": random.choice(actions), "patient_id": target, "content": "Random guess"}
        else:
            # "Learned" optimal policy: Notice crashing patient, treat with antibiotics immediately
            target = "P002" if "P002" in pending else "P001"
            return {"action_type": "decide_treatment", "patient_id": target, "content": "Administer IV antibiotics immediately"}


def run_training_simulation(episodes: int = 1000):
    env = TrueWardRoundEnv()
    policy = MockPolicy()
    
    returns = []
    success_rates = []
    
    print(f"Starting Training Simulation ({episodes} episodes)...")
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        
        # Epsilon decays from 1.0 -> 0.0
        epsilon = max(0.01, 1.0 - (ep / (episodes * 0.8)))
        
        while not done:
            action = policy.act(obs, epsilon)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        returns.append(total_reward)
        success_rates.append(1 if info["survived"] == 2 else 0)
        
        if ep % 200 == 0 or ep == episodes - 1:
            avg_ret = sum(returns[-100:]) / min(100, len(returns))
            avg_surv = sum(success_rates[-100:]) / min(100, len(success_rates))
            print(f"Episode {ep:04d} | Epsilon: {epsilon:.2f} | Avg Return: {avg_ret:.2f} | Survival Rate: {avg_surv*100:.0f}%")

if __name__ == "__main__":
    run_training_simulation(1000)
