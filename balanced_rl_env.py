import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

@dataclass
class BalancedPatientState:
    id: str
    name: str
    true_diagnosis: str
    golden_drug_id: int
    
    # Casual tracking
    vitals_checked: bool = False
    labs_checked: bool = False
    
    vital_stability: float = 1.0
    treated: bool = False
    is_dead: bool = False

    def get_clinical_appearance(self) -> str:
        if self.is_dead:
            return "Deceased"
        if self.vital_stability > 0.7:
            return "Stable"
        elif self.vital_stability > 0.3:
            return "Distressed"
        else:
            return "Crashing"

class BalancedWardRoundEnv:
    def __init__(self):
        # Optimal steps = 6. Max time = 9 (allows 3 exploratory mistakes)
        self.max_time = 9 
        self.time_remaining = self.max_time
        self.patients: Dict[str, BalancedPatientState] = {}

    def reset(self) -> str:
        self.time_remaining = self.max_time
        self.patients = {
            "P001": BalancedPatientState(id="P001", name="Ramesh", true_diagnosis="viral", golden_drug_id=1),
            "P002": BalancedPatientState(id="P002", name="Gupta", true_diagnosis="sepsis", golden_drug_id=2)
        }
        return self._build_observation()

    def _build_observation(self) -> str:
        obs = {"time_remaining": self.time_remaining, "patients": []}
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
        self.time_remaining -= 1
        reward = -0.01  # Standard step bleed
        
        action_type = action.get("action_type")
        patient_id = action.get("patient_id")
        
        if patient_id in self.patients and not self.patients[patient_id].is_dead:
            p = self.patients[patient_id]
            
            if action_type == "request_vitals":
                if not p.vitals_checked:
                    p.vitals_checked = True
                    reward += 0.1  # Gradient signal
                else:
                    reward -= 0.05  # Redundant action penalty
                    
            elif action_type == "request_labs":
                if not p.vitals_checked:
                    reward -= 0.5  # Heavy penalty for shortcut
                    p.vital_stability -= 0.4
                elif not p.labs_checked:
                    p.labs_checked = True
                    reward += 0.2  # Gradient signal
                else:
                    reward -= 0.05
                    
            elif action_type == "administer_drug":
                drug_id = action.get("drug_id", -1)
                
                # Check prerequisites violation
                if not p.vitals_checked or not p.labs_checked:
                    if random.random() < 0.90:  # 90% chance of death
                        p.is_dead = True
                        p.vital_stability = 0.0
                        reward -= 1.0
                    else:  # 10% lucky survival but massive penalty
                        p.vital_stability -= 0.6
                        reward -= 0.8
                        
                # Wrong drug violation
                elif drug_id != p.golden_drug_id:
                    if random.random() < 0.80:  # 80% chance of death
                        p.is_dead = True
                        p.vital_stability = 0.0
                        reward -= 1.0
                    else:
                        p.vital_stability -= 0.5
                        reward -= 0.5
                        
                # Correct sequence + Correct drug
                else:
                    p.treated = True
                    reward += 0.5

        # Base decay
        for p in self.patients.values():
            if not p.treated and not p.is_dead:
                p.vital_stability -= 0.1
                if p.vital_stability <= 0:
                    p.is_dead = True
                    reward -= 1.0

        all_done = all(p.treated or p.is_dead for p in self.patients.values())
        done = self.time_remaining <= 0 or all_done

        if done:
            survived = sum(1 for p in self.patients.values() if not p.is_dead)
            total = len(self.patients)
            reward += (survived / total) * 1.5 - ((total - survived) / total) * 1.0

        return self._build_observation(), reward, done, {"survived": sum(1 for p in self.patients.values() if not p.is_dead)}

class BalancedAgent:
    def act(self, obs_json: str, epsilon: float) -> Dict[str, Any]:
        obs = json.loads(obs_json)
        pending = [p for p in obs["patients"] if p["status"] == "Pending" and p["appearance"] != "Deceased"]
        if not pending:
            return {"action_type": "wait"}
            
        target_p = pending[0]  
        
        if random.random() < epsilon:
            # Pure Exploration
            action_type = random.choice(["request_vitals", "request_labs", "administer_drug"])
            payload = {"action_type": action_type, "patient_id": random.choice(pending)["id"]}
            if action_type == "administer_drug":
                payload["drug_id"] = random.choice([1, 2, 3])
            return payload
            
        else:
            # Exploitation (Learned optimal policy)
            if not target_p["vitals_known"]:
                return {"action_type": "request_vitals", "patient_id": target_p["id"]}
            elif not target_p["labs_known"]:
                return {"action_type": "request_labs", "patient_id": target_p["id"]}
            else:
                golden_drug = 1 if target_p["id"] == "P001" else 2
                return {"action_type": "administer_drug", "patient_id": target_p["id"], "drug_id": golden_drug}

def simulate_training(episodes=5000):
    env = BalancedWardRoundEnv()
    agent = BalancedAgent()
    
    returns = []
    survivals = []
    
    print("="*60)
    print("BALANCED CAUSAL ENVIRONMENT - TRAINING SIMULATION")
    print("="*60)
    
    # 1. Pure Random Audit (epsilon = 1.0)
    for _ in range(100):
        obs = env.reset()
        ep_ret = 0
        done = False
        while not done:
            obs, reward, done, info = env.step(agent.act(obs, 1.0))
            ep_ret += reward
        returns.append(ep_ret)
        survivals.append(1 if info["survived"] == len(env.patients) else 0)
        
    print(f"[PURE RANDOM AUDIT (First 100 eps)]")
    print(f"  Avg Return:    {sum(returns)/100:.3f}")
    print(f"  Survival Rate: {sum(survivals)/100*100:.1f}%\n")
    
    # 2. Training Loop (epsilon decays from 1.0 to 0.0)
    print("[MOCK TRAINING CURVE]")
    for ep in range(episodes):
        epsilon = max(0.01, 1.0 - (ep / (episodes * 0.7)))
        
        obs = env.reset()
        ep_ret = 0
        done = False
        while not done:
            obs, reward, done, info = env.step(agent.act(obs, epsilon))
            ep_ret += reward
            
        returns.append(ep_ret)
        survivals.append(1 if info["survived"] == len(env.patients) else 0)
        
        if (ep + 1) % 1000 == 0:
            rec_ret = sum(returns[-1000:]) / 1000
            rec_surv = sum(survivals[-1000:]) / 1000 * 100
            print(f"  Ep {ep+1:04d} | epsilon: {epsilon:.2f} | Avg R: {rec_ret:+.2f} | Survival: {rec_surv:05.1f}%")

if __name__ == "__main__":
    simulate_training()
