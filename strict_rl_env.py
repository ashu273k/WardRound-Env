import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class StrictPatientState:
    id: str
    name: str
    true_diagnosis: str
    golden_drug_id: int
    
    # Strict causal trackers
    vitals_checked: bool = False
    labs_checked: bool = False
    
    vital_stability: float = 1.0
    decay_rate: float = 0.2  # Very aggressive decay
    treated: bool = False
    is_dead: bool = False

    def get_clinical_appearance(self) -> str:
        if self.is_dead:
            return "Deceased"
        if self.vital_stability > 0.8:
            return "Stable"
        elif self.vital_stability > 0.4:
            return "Tachycardic & Distressed"
        else:
            return "Crashing"

class StrictWardRoundEnv:
    def __init__(self):
        # Optimal path: vitals(1) -> labs(1) -> drug(1) = 3 steps per patient.
        # Total optimal for 2 patients = 6 steps. Max time = 7.
        self.max_time = 7  
        self.time_remaining = self.max_time
        self.patients: Dict[str, StrictPatientState] = {}
        self.step_count = 0

    def reset(self) -> str:
        self.time_remaining = self.max_time
        self.step_count = 0
        
        self.patients = {
            "P001": StrictPatientState(
                id="P001", name="Ramesh", true_diagnosis="viral", golden_drug_id=1, decay_rate=0.1
            ),
            "P002": StrictPatientState(
                id="P002", name="Gupta", true_diagnosis="sepsis", golden_drug_id=2, decay_rate=0.25
            )
        }
        return self._build_observation()

    def _build_observation(self) -> str:
        obs = {
            "time_remaining": self.time_remaining,
            "patients": []
        }
        for p in self.patients.values():
            obs["patients"].append({
                "id": p.id,
                "appearance": p.get_clinical_appearance(),
                "vitals_known": p.vitals_checked,
                "labs_known": p.labs_checked,
                "status": "Treated" if p.treated else "Pending"
            })
        return json.dumps(obs, indent=2)

    def step(self, action: Dict[str, Any]) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.step_count += 1
        self.time_remaining -= 1
        
        reward = -0.01  # Universal step penalty
        
        action_type = action.get("action_type")
        patient_id = action.get("patient_id")
        
        if patient_id in self.patients and not self.patients[patient_id].is_dead:
            p = self.patients[patient_id]
            
            # Causal Processing
            if action_type == "request_vitals":
                p.vitals_checked = True
                reward += 0.05  # Minor sub-goal reward to guide learning
                
            elif action_type == "request_labs":
                if not p.vitals_checked:
                    # Invalid causal chain!
                    reward -= 0.5
                    p.vital_stability -= 0.5  # Critical delay
                else:
                    p.labs_checked = True
                    reward += 0.05
                    
            elif action_type == "administer_drug":
                drug_id = action.get("drug_id", -1)
                
                # Strict Precondition Masking Failure
                if not p.vitals_checked or not p.labs_checked:
                    p.is_dead = True
                    p.vital_stability = 0.0
                    reward -= 1.0  # Instant death for premature treatment
                    
                # Wrong treatment Failure
                elif drug_id != p.golden_drug_id:
                    p.is_dead = True
                    p.vital_stability = 0.0
                    reward -= 1.0  # Fatal adverse reaction
                    
                # Correct Sequence & Drug
                else:
                    p.treated = True
                    p.vital_stability = 1.0
                    p.decay_rate = 0.0
                    reward += 0.2

        # Cross-patient continuous deterioration
        for p in self.patients.values():
            if not p.treated and not p.is_dead:
                p.vital_stability -= p.decay_rate
                if p.vital_stability <= 0.0:
                    p.is_dead = True
                    reward -= 1.0

        all_treated_or_dead = all(p.treated or p.is_dead for p in self.patients.values())
        done = self.time_remaining <= 0 or all_treated_or_dead

        # Terminal Scoring
        if done:
            survived = sum(1 for p in self.patients.values() if not p.is_dead)
            reward += (survived / len(self.patients)) * 1.0
            
            if survived < len(self.patients):
                reward -= 1.0  # Episode failed if anyone died

        return self._build_observation(), reward, done, {"survived": sum(1 for p in self.patients.values() if not p.is_dead)}


class StrictRandomPolicy:
    def act(self, obs_json: str) -> Dict[str, Any]:
        obs = json.loads(obs_json)
        pending = [p["id"] for p in obs["patients"] if p["status"] == "Pending" and p["appearance"] != "Deceased"]
        
        if not pending:
            return {"action_type": "wait"}
            
        target = random.choice(pending)
        action_type = random.choice(["request_vitals", "request_labs", "administer_drug"])
        
        payload = {"action_type": action_type, "patient_id": target}
        
        if action_type == "administer_drug":
            payload["drug_id"] = random.choice([1, 2, 3])  # Blindly guess drug
            
        return payload

def run_causal_audit(episodes=1000):
    env = StrictWardRoundEnv()
    policy = StrictRandomPolicy()
    
    returns = []
    survivals = []
    
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            
        returns.append(ep_reward)
        survivals.append(1 if info["survived"] == len(env.patients) else 0)
        
    avg_ret = sum(returns) / episodes
    surv_rate = sum(survivals) / episodes * 100
    
    print("="*50)
    print("STRICT CAUSAL ENVIRONMENT - RANDOM POLICY AUDIT")
    print(f"Total Episodes Run: {episodes}")
    print(f"Average Episodic Return: {avg_ret:.4f}")
    print(f"Survival Rate (Both patients saved): {surv_rate:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_causal_audit(1000)
