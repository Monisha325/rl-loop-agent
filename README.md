# RL Loop Agent — Loop Distribution via Reinforcement Learning
> Component 3 of the Compiler Analysis Pipeline  
> Inspired by *"Reinforcement Learning Assisted Loop Distribution for Locality and Vectorization"* — IITH Compilers Lab, Dr. Ramakrishna Upadrasta

---

## What I Built

A PPO agent that learns to decide, node by node over an SCC Dependence Graph (SDG), whether to **SPLIT** a loop into a parallel distribution or **KEEP** it sequential — balancing parallelism gain against loop-carried dependence cost.

The agent reads an SDG JSON (output from the SDG Builder), steps through each node as a separate decision, accumulates reward based on a cost model that penalizes wrong decisions, and outputs a per-node distribution plan. After training, it is evaluated against two rule-based baselines: No-Opt (always KEEP) and Always-Split.

---

## Pipeline Diagram

```
SDG JSON
    │
    ▼
[LoopDistributionEnv]   obs = [size, parallelizable, loop_carried_count, progress]
  Gymnasium env          action ∈ {0:KEEP, 1:SPLIT} — one decision per SDG node
    │
    ▼
[RewardFunction]        SPLIT + parallel    → +5.0 − (size × 0.1)
  cost model            SPLIT + carried     → +2.0 − (loop_carried × 0.6)
                        KEEP  + parallel    → −2.0
                        KEEP  + sequential  → +1.0 − (size × 0.05)
    │
    ▼
[PPO Agent]             MlpPolicy, lr=0.0003, 200 episodes × 50 timesteps/episode
  stable-baselines3      saved per loop as output/models/{loop}_ppo.zip
    │
    ▼
[Evaluation]            RL vs No-Opt vs Always-Split → results JSON + comparison chart
```

---

## Project Structure

```
rl-loop-agent/
├── env/
│   └── loop_env.py             # Gymnasium env — reads SDG JSON, steps node by node
├── rewards/
│   └── reward_function.py      # per-node reward based on action × node properties
├── train/
│   └── trainer.py              # PPO training + baseline agents + plotting + result saving
├── output/
│   ├── models/                 # saved PPO .zip for each of 5 trained loops
│   └── plots/                  # reward curves + strategy comparison bar charts
└── notebooks/
    └── RL_Loop_Agent.ipynb     # full Colab walkthrough
```

---

## How Each Component Works

### Environment — `loop_env.py`

`LoopDistributionEnv` is a standard Gymnasium env. On `reset()`, it loads the SDG JSON and sets the node index to 0. Each `step(action)` moves to the next node — the episode ends when all nodes have been decided. The observation at each step is a 4-float vector: `[size, parallelizable, loop_carried_count, progress]`, where `progress = index / total_nodes` gives the agent positional context within the loop.

### Reward Function — `reward_function.py`

`compute_reward(node, action)` implements a cost model with four cases. SPLITting a parallelizable node yields a base reward of +5.0 scaled down by instruction cost (`size × 0.1`). SPLITting a loop-carried node yields +2.0 but is penalized by dependence cost (`loop_carried × 0.6`). KEEPing a parallelizable node is always penalized at −2.0 (missed opportunity). KEEPing a sequential node yields +1.0 minus a small instruction cost. This asymmetry teaches the agent that the cost of missing parallelism is higher than the cost of a bad split.

### Training & Evaluation — `trainer.py`

`train_agent()` wraps the full lifecycle: instantiates the env, creates a PPO model with `MlpPolicy`, runs 200 episodes of 50 timesteps each, saves the model, then runs three evaluation passes — the trained RL agent, `no_opt_agent` (always action=0), and `always_split_agent` (always action=1). Results are saved as JSON and as a bar chart comparing total reward across strategies.

---

## Results

| Loop | No-Opt (always KEEP) | Always-Split | RL Agent |
|---|---|---|---|
| `loop_parallel` (3 independent nodes) | −6.00 | +13.80 | +13.80 |
| `loop_carried` (1 dependent, 1 free) | −1.30 | +5.40 | +5.40 |
| `loop_mixed` (1 parallel, 1 cyclic SCC) | −1.40 | +4.80 | **+5.20** |

`loop_mixed` is the critical case. Always-Split blindly splits the cyclic SCC and absorbs the loop-carried penalty, landing at +4.80. The RL agent correctly SPLITs the parallelizable node and KEEPs the cyclic one, scoring +5.20 — the only loop where RL outperforms both baselines by learning the asymmetric cost structure.

---

## Connection to Dr. Upadrasta's Research

| This Project | Paper's System |
|---|---|
| `LoopDistributionEnv` stepping over SDG nodes | RL environment over LLVM loop distribution pass |
| 4-feature observation vector (`_get_state()`) | IR2Vec embeddings as richer node features |
| SPLIT / KEEP action space | Loop distribution decisions per SCC |
| PPO with MlpPolicy (`trainer.py`) | Trained RL policy from LLVM-HPC 2022 |
| Hand-crafted cost model reward | Measured runtime speedup (cache miss rate, vectorization throughput) |
| No-Opt and Always-Split baselines | Rule-based distribution heuristics used for comparison |

---

## How to Run

```bash
pip install gymnasium stable-baselines3 matplotlib numpy

# Train on a specific loop SDG:
python -c "from train.trainer import train_agent; train_agent('path/to/loop_mixed_sdg.json', 'output')"

# Or open notebooks/RL_Loop_Agent.ipynb in Colab and run all cells
```

---

Built for internship application to the **Scalable Compilers for Heterogeneous Architectures Lab**,  
Dept. of Computer Science & Engineering, IIT Hyderabad — Dr. Ramakrishna Upadrasta  
[compilers.cse.iith.ac.in](https://compilers.cse.iith.ac.in)
