# PhGPO: Pheromone-Guided Policy Optimization for Long-Horizon Tool Planning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **PhGPO** is a novel reinforcement learning framework that enables explicit reuse of historically successful tool-use trajectories for long-horizon tool planning. Inspired by Ant Colony Optimization (ACO), PhGPO maintains a pheromone-based explicit transition prior that guides policy optimization toward historically successful tool transitions.

<img width="2812" height="1260" alt="image" src="https://github.com/user-attachments/assets/a14d4fc8-49b7-4a84-93ac-e7cadd32090d" />



## 🎯 Overview

Long-horizon multi-step tool planning is challenging because the exploration space suffers from a combinatorial explosion. Even when a correct tool-use path is found, it is typically absorbed implicitly into policy parameters, making successful transition patterns hard to reuse in subsequent training.

**PhGPO addresses this challenge by:**

1. **Explicit Transition Prior**: Distilling successful trajectories into an explicit pheromone-based transition prior
2. **Tool-Transition Graph**: Building an MCP-grounded graph that captures tool transitions and argument invocations
3. **Progressive Training**: Using a progressive pipeline that accumulates and exploits pheromone over long tool-use trajectories

## ✨ Key Features

### 🐜 Pheromone Mechanism

- **Task-Agnostic Pheromones**: Cross-task transition patterns accumulated from all successful trajectories
- **Task-Dependent Pheromones**: Task-specific patterns retrieved via similarity-based memory banks
- **Dynamic Updates**: Pheromone deposition and evaporation keep patterns reusable and up-to-date

### 🔄 Progressive Training Pipeline

1. **Supervised Warm-up**: Next-tool prediction objective for stable initialization
2. **Pheromone-Guided RL**: Progressive schedule with decaying oracle guidance
3. **Full Pheromone Optimization**: Autonomous rollouts under complete pheromone guidance

### 🛠️ Tool-Transition Graph

- Built on Model Context Protocol (MCP)
- Captures tool-to-tool dependencies
- Maintains argument-invocation patterns
- Provides structured space for modeling multi-step trajectories


### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30.0
- numpy >= 1.20.0
- tqdm, wandb (optional)


## 📊 Benchmarks

PhGPO is evaluated on three long-horizon tool-use benchmarks:

### Toolathlon
- **Tools**: 604 tools across 32 applications
- **Tasks**: 108 manually sourced tasks
- **Trajectory Length**: ~20 tool calls

### TRAJECT-Bench
- **Tools**: 1,228 production-style APIs
- **Queries**: 5,870 queries
- **Trajectory Length**: 3-10+ tool calls

### TOUCAN (Multi-step)
- **Tools**: 2,000+ tools from 495 MCP servers
- **Trajectories**: 1,646,546 trajectories
- **Trajectory Length**: 2-25+ tool calls

## 📈 Results

### Main Results

| Method | Toolathlon (Match R.) | TRAJECT-Bench (Match R.) | TOUCAN (Match R.) |
|--------|----------------------|-------------------------|-------------------|
| ReAct | 5.31 | 15.32 | 11.64 |
| Gorilla | 6.60 | 29.04 | 22.32 |
| TreeRL | 21.26 | 35.45 | 30.65 |
| GTool | 20.60 | 32.50 | 27.56 |
| **PhGPO (Ours)** | **25.25** | **41.62** | **35.17** |

*Results on Qwen2.5-7B backbone. See paper for complete results.*

### Key Improvements

- ✅ **+18.8% vs TreeRL** on Toolathlon
- ✅ **+17.4% vs GTool** on TRAJECT-Bench
- ✅ **+14.7% vs TreeRL** on TOUCAN
- ✅ Consistent gains across all benchmarks and backbones

## 🔬 Method Details

### Pheromone-Guided Sampling

At each decision step, PhGPO samples tools according to:

```math
p(a_t | s_t, a_{t-1}) \propto \exp(\log \pi_\theta(a_t | s_t) + \beta \log \tau(a_{t-1}, a_t | x))
```

where:
- `π_θ(a_t | s_t)` is the neural policy
- `τ(a_{t-1}, a_t | x)` is the fused pheromone
- `β` controls pheromone influence

### Pheromone Update

Pheromones are updated via ACO-style deposition and evaporation:

```math
\tau^{agn} \leftarrow \text{clip}((1 - \rho)\tau^{agn} + \alpha q(\xi), [\tau_{min}, \tau_{max}])
```

where:
- `ρ ∈ (0,1)` is the evaporation rate
- `α > 0` is the deposition rate
- `q(ξ) ∈ [0,1]` is the trajectory quality score

### Progressive Training Schedule

1. **Supervised Warm-up**: Pre-train with next-tool prediction
2. **Mixed Curriculum**: Gradually decay oracle probability `p_tf` while increasing horizon length
3. **Full Pheromone Rollouts**: Autonomous trajectory generation under pheromone guidance



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Ant Colony Optimization (ACO) principles
- Built on the Model Context Protocol (MCP)
- Benchmarks: Toolathlon, TRAJECT-Bench, TOUCAN



**Note**: This is a preliminary work under review. Code and additional resources will be released upon publication.
