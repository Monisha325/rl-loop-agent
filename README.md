# RL Loop Agent — Loop Distribution via Reinforcement Learning
> Component 3 of the Compiler Analysis Pipeline
> Inspired by *"Reinforcement Learning Assisted Loop Distribution for Locality and Vectorization"* — IITH Compilers Lab, Dr. Ramakrishna Upadrasta

---

## What It Does

Trains a PPO agent to decide, node by node over an SCC Dependence Graph (SDG), whether to SPLIT a loop into a parallel distribution or KEEP it sequential — balancing parallelism gain against loop-carried dependence cost. It consumes the SDG JSON output from the SDG Builder (Component 2) and produces a per-node distribution plan. This mirrors the RL-based loop distribution policy in Dr. Upadrasta's LLVM-HPC 2022 paper, where the agent replaces hand-written heuristics for distribution ordering.

---

## Pipeline Diagram

```
SDG JSON (from Component 2)
        │
        ▼
[LoopDistributionEnv]   obs = [size, parallelizable, loop_carried_count, progress]
  Gymnasium env          action ∈ {0:KEEP, 1:SPLIT}, one step per SDG node
        │
        ▼
[RewardFunction]        SPLIT+parallel→+5.0, KEEP+parallel→-2.0
  cost model            SPLIT+carried→2.0−penalty, KEEP+sequential→1.0−cost
        │
        ▼
[PPO Agent]             MlpPolicy, lr=0.0003, 200 episodes × 50 timesteps
  stable-baselines3
        │
        ▼
[Evaluation]            RL vs No-Opt (always KEEP) vs Always-Split
        │
        ▼
Distribution plan (JSON) + reward curve (PNG) + comparison chart (PNG)
```

---

## Project Structure

```
rl-loop-agent/
├── env/
│   └── loop_env.py             # Gymnasium env — reads SDG JSON, steps node by node
├── rewards/
│   └── reward_function.py      # reward per node based on action + node properties
├── train/
│   └── trainer.py              # PPO training, baseline evaluation, result saving
├── output/
│   ├── models/                 # saved PPO .zip per loop (5 loops trained)
│   └── plots/                  # reward curves + strategy comparison bar charts
└── notebooks/
    └── RL_Loop_Agent.ipynb     # full Colab walkthrough
```

---

## Results

Rewards computed from `reward_function.py` across 3 representative test loops:

| Loop | No-Opt (always KEEP) | Always-Split | RL Agent (optimal) |
|---|---|---|---|
| `loop_parallel` (3 independent nodes) | -6.00 | +13.80 | +13.80 |
| `loop_carried` (1 dependent, 1 free) | -1.30 | +5.40 | +5.40 |
| `loop_mixed` (1 parallel, 1 cyclic SCC) | -1.40 | +4.80 | **+5.20** |

> `loop_mixed` is the only case where RL outperforms both baselines — it correctly SPLITs the parallelizable SCC and KEEPs the cyclic one, while Always-Split takes the penalty on the loop-carried node.

---

## Connection to Dr. Upadrasta's Research

| This Project | Paper's System | Your Component | Their Equivalent |
|---|---|---|---|
| `LoopDistributionEnv` | RL env over SDG nodes | `loop_env.py` | LLVM loop distribution environment |
| 4-feature observation vector | Node feature representation | `_get_state()` in env | IR2Vec embeddings as node features |
| SPLIT / KEEP actions | Distribution decision per SCC | `action_space` | Loop distribution pass actions |
| PPO (MlpPolicy) | RL policy over distribution space | `trainer.py` | Trained RL model (LLVM-HPC 2022) |
| Hand-crafted cost model reward | Runtime measurements (cache, vectorization) | `reward_function.py` | Measured speedup as reward signal |
| Baseline comparison | Heuristic agent evaluation | `no_opt_agent`, `always_split_agent` | Rule-based distribution baselines |

---

## How to Run

```bash
pip install gymnasium stable-baselines3 matplotlib numpy
python -c "from train.trainer import train_agent; train_agent('path/to/loop_mixed_sdg.json', 'output')"
# or open notebooks/RL_Loop_Agent.ipynb in Colab and run all cells
```

---

## Limitations

- Reward is a hand-crafted cost model — the paper uses actual runtime measurements (cache miss rates, vectorization throughput) from compiled LLVM IR as the reward signal
- Observation uses 4 scalar features — the paper uses IR2Vec embeddings from LLVM IR as richer node representations; integrating IR2Vec from Component 1 is the direct next step
- 200 episodes × 50 timesteps is lightweight — production training uses thousands of real loops from PolyBench and SPEC benchmark suites
- No actual compilation — distribution plan is output as JSON pseudo-decisions; a full system feeds into the LLVM loop distribution pass and measures real speedup

---

Built for internship application to the **Scalable Compilers for Heterogeneous Architectures Lab**,
Dept. of Computer Science & Engineering, IIT Hyderabad — Dr. Ramakrishna Upadrasta
[compilers.cse.iith.ac.in](https://compilers.cse.iith.ac.in)