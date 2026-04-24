
### **Project Description for Round 2: WardRound-Env**

**Project Name:** WardRound-Env  
**Theme:** Theme #1 - Multi-Agent Interactions  
**Goal:** Build a realistic multi-agent hospital ward round simulator where an AI agent (Junior Doctor) learns to lead effective morning ward rounds by interacting with other agents.

#### Project Overview
WardRound-Env simulates a hospital ward round where the **main agent (Junior Doctor)** must:
- Present patient cases clearly
- Answer questions from the Senior Consultant
- Coordinate with the Nurse for immediate tasks
- Handle patient/family concerns
- Make treatment decisions while managing conflicting opinions and time pressure

Other roles (Senior Consultant, Nurse, Patient/Family) will be **rule-based scripted agents** with different personalities, incentives, and knowledge levels. This creates natural negotiation, theory-of-mind, and strategic behavior.

The environment will support **3 difficulty levels**:
- **Easy**: Stable patients, cooperative team
- **Medium**: One conflicting opinion + minor time pressure
- **Hard**: Ethical dilemma, strong disagreement between consultant and family, tight time constraints

#### Key Requirements
- Full OpenEnv compliance (typed models, step/reset/state, openenv.yaml)
- Hosted on Hugging Face Spaces with Docker
- Meaningful shaped reward function with partial progress signals
- Deterministic grader that outputs score between 0.0 – 1.0
- Training script using Unsloth or HF TRL (to show reward improvement)
- Short demo video (<2 min) + mini-blog on Hugging Face

#### Why This Project is Strong
- High Innovation (Multi-agent medical decision making with real conflicts)
- Excellent Storytelling potential (dramatic ward round scenarios)
- Clear observable training progress (better patient outcomes, fewer conflicts, faster rounds)
- Real-world relevance (hospital workflow simulation)

---

### Team Division (Using Cursor)

Here’s how we’ll split the work:

**Ram (You)** - Project Lead & Core Environment
- Create the project skeleton using `openenv init`
- Design `models.py` (Action, Observation, Reward)
- Implement `reset()` and core `step()` logic
- Handle `openenv.yaml`, deployment to HF Spaces, and Docker
- Final integration and testing

**Ashu** - Reward System & Training
- Design the reward function (partial rewards for good presentation, coordination, decision quality)
- Implement the deterministic grader (patient outcome, team agreement, time efficiency)
- Build the training script using Unsloth / HF TRL
- Run training and generate reward curves / before-after comparisons

**Abhijeet** - Multi-Agent Simulation & Tasks
- Create rule-based behaviors for Senior Consultant, Nurse, and Patient/Family agents
- Define the 3 tasks (easy, medium, hard) with different scenarios
- Implement patient cases and dynamic responses during ward round
- Help with prompt engineering for agent interactions

#### Timeline (48 hours)
- **Day 1 Morning**: Project setup + models + basic environment (Ram)
- **Day 1 Afternoon**: Multi-agent logic + tasks (Abhijeet) + Reward design (Ashu)
- **Day 1 Evening**: Integration + basic testing
- **Day 2 Morning**: Training script + initial training runs
- **Day 2 Afternoon**: Polish, video recording, mini-blog, final deployment

We will use **Cursor** heavily for code generation and debugging.

---

Would you like me to now create:
1. The **full starter code** (openenv.yaml + models.py + basic environment.py skeleton) so you can start immediately?
2. Or a more detailed task list with exact files each person should work on?

Just say “Give me the starter code” and I’ll provide everything ready to copy-paste. 

Ready when you are! Let’s build this.