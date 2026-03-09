# RL Loop Agent — Loop Distribution via Reinforcement Learning

> **Component 2 of the Loop Parallelization Pipeline**
> Inspired by *"Reinforcement Learning Assisted Loop Distribution for Locality and Vectorization"* — IITH Compilers Lab, Dr. Ramakrishna Upadrasta

---

## What This Project Does

Once the **SCC Dependence Graph (SDG)** is built (Component 1 — SDG Builder), the next problem is: **in what order should we distribute the loop nodes, and should each node be split into a parallel loop or kept sequential?**

This project trains a **PPO (Proximal Policy Optimization) reinforcement learning agent** to make that decision. The agent walks the SDG node by node, observes each node's properties (size, parallelizability, loop-carried count), and decides: **SPLIT** (emit a parallel loop) or **KEEP** (emit a sequential loop).

The trained agent is benchmarked against two baselines — a no-optimization agent (always KEEP) and a greedy agent (always SPLIT) — to show the RL agent learns a smarter distribution strategy.

---

## How It Connects to Component 1

This project **directly consumes the JSON output** of the SDG Builder:

```
sdg-loop-distribution/output/graphs/
    ├── example_loop_sdg.json       ← input to this project
    ├── loop_parallel_sdg.json
    ├── loop_mixed_sdg.json
    ├── loop_carried_sdg.json
    └── loop_complex_sdg.json
```

Each JSON file contains the SDG nodes with their features — `size`, `parallelizable`, `loop_carried_count` — which become the **observation vector** for the RL agent.

---

## The Full Pipeline

```
SDG JSON (from Component 1)
        │
        ▼
[ LoopDistributionEnv ]         ← Gymnasium environment
  Reads SDG nodes one by one
  Observation = [size, parallelizable, loop_carried_count, progress]
  Action Space = {0: KEEP, 1: SPLIT}
        │
        ▼
[ PPO Agent ]                   ← stable-baselines3
  MlpPolicy, lr=0.0003
  200 episodes × 50 timesteps
        │
        ▼
[ Reward Function ]
  Rewards correct decisions, penalizes wrong ones
  e.g. SPLIT on parallelizable node → +5.0
       KEEP  on parallelizable node → -2.0
        │
        ▼
[ Evaluation ]
  Compare RL vs No-Opt vs Always-Split
        │
        ▼
[ Output ]
  Trained model (.zip)
  Distribution plan (JSON)
  Reward curve (PNG)
  Comparison chart (PNG)
```

---

## Project Structure

```
rl-loop-agent/
│
├── env/
│   └── loop_env.py             # Gymnasium environment — reads SDG JSON, steps node by node
│
├── rewards/
│   └── reward_function.py      # Reward logic based on node properties and agent action
│
├── train/
│   └── trainer.py              # PPO training loop, evaluation, baseline comparison, plots
│
├── output/
│   ├── models/                 # Saved PPO models (one per loop)
│   │   ├── example_loop_ppo.zip
│   │   ├── loop_parallel_ppo.zip
│   │   ├── loop_carried_ppo.zip
│   │   ├── loop_mixed_ppo.zip
│   │   └── loop_complex_ppo.zip
│   └── plots/                  # Training reward curves + strategy comparison charts
│       └── comparison_chart.png
│
└── notebooks/
    └── RL_Loop_Agent.ipynb     # Full Colab notebook — setup to training to evaluation
```

---

## Environment Design

### Observation Space
A 4-dimensional float vector representing the current SDG node:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `size` | Number of statements in this SCC |
| 1 | `parallelizable` | 1.0 if safe to parallelize, 0.0 if not |
| 2 | `loop_carried_count` | Number of loop-carried edges within this SCC |
| 3 | `progress` | `index / total_nodes` — how far through the SDG |

### Action Space
Discrete with 2 actions:

| Action | Label | Meaning |
|--------|-------|---------|
| `0` | KEEP | Keep this SCC in a sequential loop |
| `1` | SPLIT | Distribute this SCC into its own parallel loop |

### Episode
One episode = one complete topological walk through all SDG nodes. The episode ends when all nodes have been visited (`done = True`).

---

## Reward Function

The reward is computed per node based on what the agent decided versus what was correct:

```python
instruction_cost    = size * 2.0
cache_miss_penalty  = loop_carried_count * 3.0

if action == SPLIT:
    if parallelizable:   reward = +5.0 − (instruction_cost × 0.1)   # correct split
    else:                reward = +2.0 − (cache_miss_penalty × 0.2)  # risky split

if action == KEEP:
    if parallelizable:   reward = −2.0                                # missed opportunity
    else:                reward = +1.0 − (instruction_cost × 0.05)   # correct keep
```

**Key design decisions:**
- Splitting a parallelizable node earns the highest reward (+5.0 base)
- Keeping a parallelizable node is penalized (-2.0) — this is a missed parallelization opportunity
- Splitting a non-parallelizable node has a moderate reward reduced by cache penalty — it may still be worth distributing for locality even if not fully parallel
- Keeping a non-parallelizable node earns a small positive reward, reduced by instruction cost

---

## Baseline Agents

The trained PPO agent is compared against two rule-based baselines:

| Agent | Strategy | Purpose |
|-------|----------|---------|
| **No-Opt** | Always action=0 (KEEP) | Represents doing nothing — no distribution |
| **Always-Split** | Always action=1 (SPLIT) | Represents aggressive greedy distribution |
| **RL Agent** | Learned policy (PPO) | Adaptive decision per node |

The comparison shows the RL agent learns to outperform both baselines by making context-aware decisions — splitting when safe, keeping when loop-carried dependences make splitting costly.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | MlpPolicy (2-layer MLP) |
| Learning rate | 0.0003 |
| Episodes | 200 |
| Timesteps per episode | 50 |
| Framework | stable-baselines3 |

---

## Setup & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Run Training on All Loops

```python
import glob
import sys

sys.path.append("/path/to/rl-loop-agent")

from train.trainer import train_agent

SDG_PATH   = "/path/to/sdg-loop-distribution/output/graphs"
OUTPUT_DIR = "/path/to/rl-loop-agent/output"

sdg_files = glob.glob(f"{SDG_PATH}/*_sdg.json")

for f in sdg_files:
    print("Training on:", f)
    train_agent(f, OUTPUT_DIR)
```

### Load and Use a Trained Model

```python
from stable_baselines3 import PPO
from env.loop_env import LoopDistributionEnv

env   = LoopDistributionEnv("path/to/loop_mixed_sdg.json")
model = PPO.load("output/models/loop_mixed_ppo")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print("Node:", env.node_labels[env.index - 1],
          "→", "SPLIT" if action == 1 else "KEEP",
          "| Reward:", reward)
    if done:
        break
```

### Google Colab

Open `notebooks/RL_Loop_Agent.ipynb`. Set `ROOT` to your Drive path and `SDG_PATH` to the output folder from the SDG Builder project. Run all cells top to bottom.

---

## Output Files

### Distribution Plan (JSON)
Saved to `output/results/{loop_name}_results.json`:

```json
{
    "loop": "loop_mixed",
    "no_opt_reward": 1.5,
    "split_reward": 3.2,
    "rl_reward": 4.8,
    "distribution_plan": [
        { "node": "SCC1", "action": "SPLIT" },
        { "node": "SCC2", "action": "KEEP"  }
    ]
}
```

### Trained Model
Saved as `output/models/{loop_name}_ppo.zip` — loadable with `PPO.load()` from stable-baselines3.

### Plots
- `{loop_name}_reward.png` — episode reward curve over 200 training episodes
- `{loop_name}_comparison.png` — bar chart comparing No-Opt vs Always-Split vs RL Agent

---

## Test Cases

### loop_parallel — All Independent
3 statements, zero dependencies. All nodes are parallelizable. The agent should learn to SPLIT all nodes for maximum reward.

**Expected RL behavior:** SPLIT → SPLIT → SPLIT

### loop_carried — Loop-Carried Self-Dependence
S1 depends on its own previous iteration (`A[i-1]`). Node is not parallelizable. The agent should learn to KEEP.

**Expected RL behavior:** KEEP → SPLIT (S2 is independent)

### loop_mixed — One Cyclic, One Independent
SCC1 = {S3} parallelizable, SCC2 = {S1, S2} cyclic. The agent must learn to discriminate.

**Expected RL behavior:** SPLIT → KEEP

### example_loop / loop_complex — Full Cross-Dependency
All statements in one SCC. One node, not parallelizable. Single-step episode.

**Expected RL behavior:** KEEP

---

## Connection to Dr. Upadrasta's Research

This project implements **Component 2** of the system described in:

> *"Reinforcement Learning Assisted Loop Distribution for Locality and Vectorization"*
> S. VenkataKeerthy et al., IITH Compilers Lab, LLVM-HPC 2022

| This Project | Paper's System |
|---|---|
| `LoopDistributionEnv` | RL environment over SDG nodes |
| Observation vector (4 features) | Node feature representation |
| SPLIT / KEEP actions | Distribution decision per SCC |
| PPO agent | RL model learning distribution policy |
| Reward function | Optimization objective (parallelism + locality) |
| Baseline comparison | Evaluation against heuristic agents |

Component 1 (SDG Builder) feeds directly into this project. Together, they form the complete **SDG construction → RL-based distribution** pipeline described in Dr. Upadrasta's paper.

---

## Limitations & Future Work

- **Reward function is hand-crafted** — the paper uses actual runtime measurements (cache miss rates, vectorization speedup) as reward signals. This project approximates that with a cost model.
- **4-feature observation** — the paper uses richer IR2Vec embeddings from LLVM IR as node features. Extending the observation space with IR2Vec would significantly improve the agent.
- **1D loops only** — nested loop support requires 2D iteration vectors and a more complex environment.
- **200 episodes is lightweight** — production training would use thousands of real loop programs from benchmark suites (PolyBench, SPEC).
- **No actual compilation** — the distributed loop output is pseudo-C. A full system would feed into LLVM/Clang and measure real speedup.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `gymnasium` | ≥ 0.26 | RL environment base class |
| `stable-baselines3` | ≥ 2.0 | PPO implementation |
| `numpy` | ≥ 1.23 | Observation array handling |
| `matplotlib` | ≥ 3.5 | Reward curves and comparison plots |

---

## Author

Built as **Component 2 — RL Loop Distribution Agent** of the Loop Parallelization System.
Targeting internship application to the **Scalable Compilers for Heterogeneous Architectures Lab**,
Dept. of Computer Science & Engineering, IIT Hyderabad.
Lab: [compilers.cse.iith.ac.in](https://compilers.cse.iith.ac.in)