import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from balanced_rl_env import BalancedPatientState, BalancedWardRoundEnv

# Re-use the Balanced Env from before
# Implement a REAL Q-Learning Agent

class QLearningAgent:
    def __init__(self, action_space: List[Dict[str, Any]], alpha=0.2, gamma=0.95):
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma

    def get_state_key(self, obs_json: str) -> Tuple:
        """Flatten JSON observation into a hashable state tuple."""
        obs = json.loads(obs_json)
        state_list = []
        for p in obs['patients']:
            state_list.append((
                p['id'], 
                p['vitals_known'], 
                p['labs_known'], 
                p['status'], 
                p['appearance']
            ))
        return (obs['time_remaining'], tuple(state_list))

    def get_q(self, state_key, action_idx):
        return self.q_table.get((state_key, action_idx), 0.0)

    def act(self, obs_json: str, epsilon: float) -> int:
        state_key = self.get_state_key(obs_json)
        if random.random() < epsilon:
            return random.randint(0, len(self.action_space) - 1)
        
        q_values = [self.get_q(state_key, a) for a in range(len(self.action_space))]
        max_q = max(q_values)
        best_actions = [a for a in range(len(self.action_space)) if q_values[a] == max_q]
        return random.choice(best_actions)

    def update(self, obs_json: str, action_idx: int, reward: float, next_obs_json: str, done: bool):
        state_key = self.get_state_key(obs_json)
        next_state_key = self.get_state_key(next_obs_json)
        
        old_q = self.get_q(state_key, action_idx)
        if done:
            target_q = reward
        else:
            future_qs = [self.get_q(next_state_key, a) for a in range(len(self.action_space))]
            target_q = reward + self.gamma * max(future_qs)
            
        self.q_table[(state_key, action_idx)] = old_q + self.alpha * (target_q - old_q)


def generate_action_space() -> List[Dict[str, Any]]:
    actions = []
    for p in ["P001", "P002"]:
        actions.append({"action_type": "request_vitals", "patient_id": p})
        actions.append({"action_type": "request_labs", "patient_id": p})
        for d in [1, 2, 3]:
            actions.append({"action_type": "administer_drug", "patient_id": p, "drug_id": d})
    return actions

def run_real_training(episodes=3000):
    env = BalancedWardRoundEnv()
    action_space = generate_action_space()
    agent = QLearningAgent(action_space, alpha=0.3, gamma=0.99)
    
    returns = []
    survivals = []
    
    print("="*60)
    print("STARTING REAL Q-LEARNING TRAINING (3000 Eps)")
    print("="*60)
    
    early_trajectory = []
    late_trajectory = []

    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - (ep / (episodes * 0.7)))
        
        obs = env.reset()
        done = False
        ep_ret = 0
        
        traj = []
        
        while not done:
            action_idx = agent.act(obs, epsilon)
            action = action_space[action_idx]
            
            next_obs, reward, done, info = env.step(action)
            
            # Record trajectory
            traj.append((action, reward))
            
            # Learn
            agent.update(obs, action_idx, reward, next_obs, done)
            
            obs = next_obs
            ep_ret += reward
            
        returns.append(ep_ret)
        survivals.append(1 if info["survived"] == len(env.patients) else 0)
        
        if ep == 10:
            early_trajectory = traj
        elif ep == episodes - 1:
            late_trajectory = traj
            
        if (ep + 1) % 500 == 0:
            rec_ret = sum(returns[-500:]) / 500
            rec_surv = sum(survivals[-500:]) / 500 * 100
            print(f"  Ep {ep+1:04d} | e: {epsilon:.2f} | Avg Return: {rec_ret:+.3f} | Survival: {rec_surv:05.1f}% | States: {len(agent.q_table)}")

    print("\n" + "="*60)
    print(f"TRAINING COMPLETE. TOTAL UNIQUE Q-STATES EXPLORED: {len(agent.q_table)}")
    print("="*60)
    
    print("\nSAMPLE TRAJECTORY: EARLY (Episode 10, Random Guessing)")
    for t in early_trajectory:
        act = t[0].get('action_type')
        target = t[0].get('patient_id')
        print(f"  -> {act} on {target} | Reward: {t[1]:+.2f}")
        
    print("\nSAMPLE TRAJECTORY: LATE (Episode 3000, Exploiting Learned Policy)")
    for t in late_trajectory:
        act = t[0].get('action_type')
        target = t[0].get('patient_id')
        print(f"  -> {act} on {target} | Reward: {t[1]:+.2f}")

if __name__ == "__main__":
    run_real_training(5000)
