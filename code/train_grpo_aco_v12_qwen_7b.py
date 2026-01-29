import json
import os
os.environ["NCCL_TIMEOUT"] = "1800"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import datetime
import gc
import random
import math
import hashlib
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import contextlib
from llm_tool_simulator2 import create_simulator_from_config, LLMSimulatorConfig


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("[WARNING] sentence-transformers not installed.")
    print("         Run: pip install sentence-transformers")
    print("         Context-aware pheromone will be disabled.")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP



def setup_distributed():
    import datetime
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=60)
        )
        print(f"[DDP] Initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_value(value, average=True):
    if not dist.is_initialized():
        return value

    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=torch.cuda.current_device())

    dist.all_reduce(value, op=dist.ReduceOp.SUM)

    if average:
        value = value / get_world_size()

    return value.item() if value.numel() == 1 else value



def log(msg):
    if is_main_process():
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}", flush=True)


def gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0



class FileNaming:

    def __init__(self, experiment_name: str, project_root: Path):
        self.experiment_name = experiment_name
        self.project_root = project_root
        self.output_dir = project_root / "GRPO-ACO" / f"checkpoints_{experiment_name}"


    @property
    def sl_best(self) -> Path:
        return self.output_dir / "sl_best.pt"

    @property
    def mixed_best(self) -> Path:
        return self.output_dir / "mixed_best.pt"

    @property
    def rl_best(self) -> Path:
        return self.output_dir / "rl_best.pt"

    @property
    def policy_final(self) -> Path:
        return self.output_dir / "policy_final.pt"

    def rl_epoch(self, epoch: int) -> Path:
        return self.output_dir / f"rl_epoch_{epoch}.pt"


    @property
    def pheromone_mixed_best(self) -> Path:
        return self.output_dir / "pheromone_mixed_best.json"

    @property
    def pheromone_rl_best(self) -> Path:
        return self.output_dir / "pheromone_rl_best.json"

    @property
    def pheromone_final(self) -> Path:
        return self.output_dir / "pheromone_final.json"


    @property
    def predictions_detailed(self) -> Path:
        return self.output_dir / "predictions_detailed.json"

    @property
    def results_json(self) -> Path:
        return self.output_dir / "results.json"


    @property
    def training_log_json(self) -> Path:
        return self.output_dir / "training_step_log.json"

    @property
    def training_log_csv(self) -> Path:
        return self.output_dir / "training_step_log.csv"

    @property
    def pheromone_evolution(self) -> Path:
        return self.output_dir / "pheromone_evolution.json"

    @property
    def loss_curve(self) -> Path:
        return self.output_dir / "loss_curve.json"

    @property
    def accuracy_curve(self) -> Path:
        return self.output_dir / "accuracy_curve.json"



@dataclass
class StepRecord:
    global_step: int
    epoch: int
    episode_idx: int

    loss: float
    pg_loss: float
    entropy: float

    match_ratio: float
    model_match_ratio: float
    success: bool

    pheromone_tool_mean: float
    pheromone_tool_max: float
    pheromone_tool_min: float
    pheromone_num_edges: int
    pheromone_context_memories: int
    pheromone_hybrid_rate: float

    beta: float
    teacher_forcing: float
    context_weight: float
    learning_rate: float
    avg_reward: float = 0.0
    avg_return: float = 0.0
    trajectory_bonus: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    step_acc_pos_all: List[float] = field(default_factory=list)
    step_acc_pos_model: List[float] = field(default_factory=list)

    step_acc_mean_all: float = 0.0
    step_acc_mean_model: float = 0.0
    step_acc_correct_all: int = 0
    step_acc_total_all: int = 0
    step_acc_correct_model: int = 0
    step_acc_total_model: int = 0

    timestamp: float = field(default_factory=time.time)


class TrainingLogger:

    def __init__(self, file_naming: FileNaming):
        self.file_naming = file_naming
        self.records: List[StepRecord] = []
        self.global_step = 0

        self.pheromone_snapshots: List[Dict] = []
        self.snapshot_interval = 20

        self.window_size = 50

    def log_step(self, record: StepRecord):
        record.global_step = self.global_step
        self.records.append(record)
        self.global_step += 1

    def log_pheromone_snapshot(self, pheromone, step: int):
        if step % self.snapshot_interval != 0:
            return

        stats = pheromone.get_statistics()
        scalar_stats = stats.get('scalar', {})

        snapshot = {
            "step": step,
            "timestamp": time.time(),
            "scalar": {
                "tool_mean": scalar_stats.get('tool', {}).get('mean', 0),
                "tool_max": scalar_stats.get('tool', {}).get('max', 0),
                "tool_min": scalar_stats.get('tool', {}).get('min', 0),
                "num_edges": scalar_stats.get('tool', {}).get('num_edges', 0),
            },
            "context": {
                "total_memories": stats.get('context', {}).get('total_memories', 0),
                "num_edges_with_memory": stats.get('context', {}).get('num_edges_with_memory', 0),
                "avg_memory_score": stats.get('context', {}).get('avg_memory_score', 0),
            },
            "hybrid": {
                "context_weight": stats.get('context_weight', 0),
                "hybrid_rate": stats.get('hybrid_rate', 0),
            },
        }
        self.pheromone_snapshots.append(snapshot)

    def get_smoothed_values(self, key: str, window: int = None) -> List[float]:
        if window is None:
            window = self.window_size

        values = [getattr(r, key, 0) for r in self.records]

        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(float(np.mean(values[start:end])))

        return smoothed

    def save(self):
        if not is_main_process():
            return

        records_dict = []
        for r in self.records:
            records_dict.append({
                "global_step": r.global_step,
                "epoch": r.epoch,
                "episode_idx": r.episode_idx,
                "loss": r.loss,
                "pg_loss": r.pg_loss,
                "entropy": r.entropy,
                "match_ratio": r.match_ratio,
                "model_match_ratio": r.model_match_ratio,
                "success": r.success,
                "pheromone_tool_mean": r.pheromone_tool_mean,
                "pheromone_tool_max": r.pheromone_tool_max,
                "pheromone_tool_min": r.pheromone_tool_min,
                "pheromone_num_edges": r.pheromone_num_edges,
                "pheromone_context_memories": r.pheromone_context_memories,
                "pheromone_hybrid_rate": r.pheromone_hybrid_rate,
                "beta": r.beta,
                "teacher_forcing": r.teacher_forcing,
                "context_weight": r.context_weight,
                "learning_rate": r.learning_rate,
                "step_acc_pos_all": getattr(r, 'step_acc_pos_all', []),
                "step_acc_pos_model": getattr(r, 'step_acc_pos_model', []),
                "step_acc_mean_all": getattr(r, "step_acc_mean_all", 0.0),
                "step_acc_mean_model": getattr(r, "step_acc_mean_model", 0.0),
                "step_acc_correct_all": getattr(r, "step_acc_correct_all", 0),
                "step_acc_total_all": getattr(r, "step_acc_total_all", 0),
                "step_acc_correct_model": getattr(r, "step_acc_correct_model", 0),
                "step_acc_total_model": getattr(r, "step_acc_total_model", 0),
                "timestamp": r.timestamp,
                "avg_reward": getattr(r, 'avg_reward', 0.0),
                "avg_return": getattr(r, 'avg_return', 0.0),
                "trajectory_bonus": getattr(r, 'trajectory_bonus', 0.0),
                "min_reward": getattr(r, 'min_reward', 0.0),
                "max_reward": getattr(r, 'max_reward', 0.0),
            })

        with open(self.file_naming.training_log_json, 'w', encoding='utf-8') as f:
            json.dump({
                "records": records_dict,
                "total_steps": len(records_dict),
                "summary": self.get_summary(),
            }, f, indent=2, ensure_ascii=False)

        if records_dict:
            with open(self.file_naming.training_log_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=records_dict[0].keys())
                writer.writeheader()
                writer.writerows(records_dict)

        with open(self.file_naming.pheromone_evolution, 'w', encoding='utf-8') as f:
            json.dump({
                "snapshots": self.pheromone_snapshots,
                "snapshot_interval": self.snapshot_interval,
                "total_snapshots": len(self.pheromone_snapshots),
            }, f, indent=2, ensure_ascii=False)

        self._save_curve_data()

        log(f"[TrainingLogger] Saved {len(self.records)} step records")
        log(f"[TrainingLogger] Saved {len(self.pheromone_snapshots)} pheromone snapshots")

    def _save_curve_data(self):
        if not self.records:
            return

        steps = [r.global_step for r in self.records]

        loss_data = {
            "steps": steps,
            "loss": [r.loss for r in self.records],
            "loss_smoothed": self.get_smoothed_values("loss"),
            "pg_loss": [r.pg_loss for r in self.records],
            "entropy": [r.entropy for r in self.records],
        }
        with open(self.file_naming.loss_curve, 'w', encoding='utf-8') as f:
            json.dump(loss_data, f)

        acc_data = {
            "steps": steps,
            "match_ratio": [r.match_ratio for r in self.records],
            "match_ratio_smoothed": self.get_smoothed_values("match_ratio"),
            "model_match_ratio": [r.model_match_ratio for r in self.records],
            "success_rate": [1 if r.success else 0 for r in self.records],
        }
        with open(self.file_naming.accuracy_curve, 'w', encoding='utf-8') as f:
            json.dump(acc_data, f)

        reward_data = {
            "steps": steps,
            "avg_reward": [getattr(r, 'avg_reward', 0) for r in self.records],
            "avg_return": [getattr(r, 'avg_return', 0) for r in self.records],
            "avg_return_smoothed": self.get_smoothed_values("avg_return") if hasattr(self.records[0], 'avg_return') else [],
            "trajectory_bonus": [getattr(r, 'trajectory_bonus', 0) for r in self.records],
        }
        with open(self.file_naming.output_dir / "reward_curve.json", 'w', encoding='utf-8') as f:
            json.dump(reward_data, f)

    def get_summary(self) -> Dict:
        if not self.records:
            return {}

        losses = [r.loss for r in self.records]
        match_ratios = [r.match_ratio for r in self.records]
        pheromone_means = [r.pheromone_tool_mean for r in self.records]

        n = len(self.records)
        n_10 = max(1, n // 10)

        return {
            "total_steps": n,
            "loss": {
                "mean": float(np.mean(losses)),
                "std": float(np.std(losses)),
                "min": float(np.min(losses)),
                "max": float(np.max(losses)),
                "final": float(np.mean(losses[-n_10:])),
                "initial": float(np.mean(losses[:n_10])),
            },
            "match_ratio": {
                "mean": float(np.mean(match_ratios)),
                "std": float(np.std(match_ratios)),
                "final": float(np.mean(match_ratios[-n_10:])),
                "initial": float(np.mean(match_ratios[:n_10])),
                "improvement": float(np.mean(match_ratios[-n_10:]) - np.mean(match_ratios[:n_10])),
            },
            "pheromone": {
                "mean_initial": float(np.mean(pheromone_means[:n_10])),
                "mean_final": float(np.mean(pheromone_means[-n_10:])),
                "growth": float(np.mean(pheromone_means[-n_10:]) - np.mean(pheromone_means[:n_10])),
            },
            "stepwise_accuracy": {
                "mean_model": float(sum(getattr(r, 'step_acc_correct_model', 0) for r in self.records) /
                                   max(1, sum(getattr(r, 'step_acc_total_model', 0) for r in self.records))),
                "mean_all": float(sum(getattr(r, 'step_acc_correct_all', 0) for r in self.records) /
                                 max(1, sum(getattr(r, 'step_acc_total_all', 0) for r in self.records))),
                "total_model_predictions": int(sum(getattr(r, 'step_acc_total_model', 0) for r in self.records)),
                "total_predictions": int(sum(getattr(r, 'step_acc_total_all', 0) for r in self.records)),
            },
        }



class PassAtKEvaluator:

    @staticmethod
    def evaluate_pass_at_k(
        policy,
        pheromone,
        rollout_manager,
        episodes: List[Dict],
        cfg,
        k_values: List[int] = [1, 3],
        n_episodes: int = 100,
        n_runs_per_episode: int = 5,
        beta_override: float = None,
        success_key: str = None,

    ) -> Dict:
        if not is_main_process():
            return {}

        policy.eval()

        old_eps = cfg.epsilon_greedy
        cfg.epsilon_greedy = 0.0

        results_per_episode = []

        success_key = success_key or getattr(cfg, 'pass_at_k_success_key', 'full_success')

        log(f"[Pass@K] Evaluating {n_episodes} episodes, {n_runs_per_episode} runs each... (success_key={success_key})")

        for ep_idx, episode in enumerate(episodes[:n_episodes]):
            episode_runs = []

            for run_idx in range(n_runs_per_episode):
                random.seed(cfg.random_seed + ep_idx * 1000 + run_idx)
                torch.manual_seed(cfg.random_seed + ep_idx * 1000 + run_idx)

                traj = rollout_manager.rollout_episode(
                    policy, pheromone, episode, cfg,
                    teacher_forcing_prob=0.0,
                    beta_override=beta_override,
                )

                run_lenient_success = bool(traj.get('success', False))
                run_full_success = bool(traj.get('full_success', False))
                run_success = bool(traj.get(success_key, run_full_success if success_key == 'full_success' else run_lenient_success))

                episode_runs.append({
                    'success': run_success,
                    'success_lenient': run_lenient_success,
                    'success_full': run_full_success,
                    'match_ratio': traj.get('match_ratio', 0.0),
                    'prefix_match_ratio': traj.get('prefix_match_ratio', traj.get('match_ratio', 0.0)),
                    'full_match_ratio': traj.get('full_match_ratio', 0.0),
                    'n_steps': len(traj.get('actions', [])),
                })

            results_per_episode.append({
                'episode_idx': ep_idx,
                'task_name': episode.get('task_name', ''),
                'runs': episode_runs,
            })

            if (ep_idx + 1) % 20 == 0:
                log(f"[Pass@K] Progress: {ep_idx + 1}/{n_episodes}")

        cfg.epsilon_greedy = old_eps
        policy.train()

        def _pass_at_k(n: int, c: int, k: int) -> float:
            if k <= 0:
                return 0.0
            if c <= 0:
                return 0.0
            if k > n:
                return 0.0
            if (n - c) < k:
                return 1.0
            return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

        pass_at_k_results = {}
        for k in k_values:
            if k > n_runs_per_episode:
                log(f"[Warning] k={k} > n_runs={n_runs_per_episode}, skipping")
                continue

            per_ep = []
            for ep_result in results_per_episode:
                runs = ep_result['runs']
                c_success = sum(1 for r in runs if r['success'])
                per_ep.append(_pass_at_k(n_runs_per_episode, c_success, k))

            pass_at_k_results[f'pass@{k}'] = float(np.mean(per_ep)) if per_ep else 0.0

        all_steps = []
        successful_steps = []
        all_match_ratios = []

        for ep_result in results_per_episode:
            for run in ep_result['runs']:
                all_steps.append(run['n_steps'])
                all_match_ratios.append(run['match_ratio'])
                if run['success']:
                    successful_steps.append(run['n_steps'])

        return {
            **pass_at_k_results,
            'avg_steps_all': float(np.mean(all_steps)) if all_steps else 0,
            'avg_steps_success': float(np.mean(successful_steps)) if successful_steps else 0,
            'std_steps_success': float(np.std(successful_steps)) if len(successful_steps) > 1 else 0,
            'avg_match_ratio': float(np.mean(all_match_ratios)) if all_match_ratios else 0,
            'n_episodes_evaluated': len(results_per_episode),
            'n_runs_per_episode': n_runs_per_episode,
            'n_successful_runs': len(successful_steps),
            'total_runs': len(all_steps),
            'success_rate': len(successful_steps) / len(all_steps) if all_steps else 0,
            'success_key': success_key,
        }



class Config:

    project_root = Path("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use")

    EXPERIMENT_NAME = "v12_qwen_7b_0109"

    pass_at_k_values = [1, 3]
    pass_at_k_runs = 5
    pass_at_k_episodes = 100
    pass_at_k_success_key = "success"

    USE_7B = True

    model_path = "/seu_share2/home/fenglei/sharedata/Qwen2.5-7B-Instruct" if USE_7B else \
                 "/seu_share2/home/fenglei/230250004/Qwen2.5-1.5B-Instruct"

    original_tools_path = project_root / "json_file" / "all_tools.json"
    extended_tools_path = project_root / "json_file" / "mcp_rl_graph_v2.json"
    rl_dataset_path = project_root / "GRPO-ACO" / "data" / "rl_dataset_llm_v3.json"
    simulator_database_path = project_root / "GRPO-ACO" / "data" / "tool_simulator_database.json"

    context_pheromone_enabled = True
    context_pheromone_weight = 0.4
    context_weight_initial = 0.2
    context_weight_final = 0.5
    context_weight_warmup_epochs = 20

    max_memory_per_edge = 100
    similarity_threshold = 0.25
    memory_time_decay = 0.998
    memory_min_score = 0.3

    embedding_model = "/seu_share2/home/fenglei/230250004/all-MiniLM-L6-v2"
    embedding_cache_size = 1000

    context_confidence_threshold = 0.25
    context_min_activated = 2
    context_min_similarity = 0.35
    context_min_avg_score = 0.4

    sl_epochs = 30
    sl_batch_size = 1 if USE_7B else 16
    sl_lr = 5e-5
    sl_target_acc = 0.25
    sl_min_epochs = 15

    mixed_epochs = 35
    mixed_episodes_per_epoch = 800
    mixed_lr = 1e-6

    tf_initial = 0.4
    tf_final = 0.15
    tf_decay_epochs = 20

    sl_weight_initial = 0.7
    sl_weight_final = 0.3

    rl_epochs = 25
    rl_lr = 1e-7
    rl_min_tf = 0.05
    rl_tf_initial = 0.15
    rl_tf_decay_epochs = 10

    success_match_threshold = 0.7

    first_step_use_pheromone = False
    param_pheromone_enabled = True
    param_beta = 0.3
    param_tau0 = 1.0
    param_alpha = 0.3

    reward_layer1_intent = 0.5
    reward_layer2_execution = 0.5
    reward_execution_error = -0.5

    lookahead_enabled = True
    lookahead_candidates = 5
    lookahead_in_eval = True
    lookahead_in_train = False

    smart_param_match = True
    param_semantic_weight = 0.3
    param_history_weight = 0.2

    golden_path_min_count = 3
    golden_path_min_success = 0.5
    golden_path_boost = 10.0

    pheromone_warmup_rounds = 3
    pheromone_warmup_mult = 8.0

    rl_beta_initial = 0.5
    rl_beta_final = 0.7
    rl_beta_decay_epochs = 15

    pheromone_inject_every = 3
    pheromone_inject_mult = 5.0

    protected_edge_threshold = 5
    protected_edge_min_tau = 5.0

    gradient_accumulation = 16 if USE_7B else 1
    min_lr = 1e-7

    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2
    random_seed = 42

    lora_r = 64 if USE_7B else 64
    lora_alpha = 128 if USE_7B else 128

    max_seq_length = 4096 if USE_7B else 4096
    max_steps_per_episode = 20
    max_history = 10

    curriculum_enabled = True
    curriculum_stages = [
        {"max_steps": 3,  "epochs": 6, "tf_start": 0.4,  "tf_end": 0.25},
        {"max_steps": 5,  "epochs": 6, "tf_start": 0.35, "tf_end": 0.2},
        {"max_steps": 7,  "epochs": 6, "tf_start": 0.3,  "tf_end": 0.15},
        {"max_steps": 10, "epochs": 7, "tf_start": 0.25, "tf_end": 0.1},
    ]

    top_k = 20
    num_rollouts = 5 if USE_7B else 5
    temperature = 0.7
    epsilon_greedy = 0.05

    clip_ratio = 0.2
    max_grad_norm = 0.5

    beta_pheromone = 0.7
    beta_pheromone_max = 0.8

    reward_exact_match = 5.0
    reward_same_category = 0.3
    reward_wrong = -0.3
    reward_recovery = 0.5

    reward_trajectory_complete = 15.0
    reward_high_match = 10.0
    reward_medium_match = 6.0
    reward_partial_match = 3.0
    reward_first_steps_match = 2.0

    elite_buffer_size = 500
    elite_replay_ratio = 0.2
    elite_match_threshold = 0.3
    elite_pheromone_multiplier = 5.0
    elite_min_tf_to_collect = 0.4
    elite_min_model_match = 0.3

    tau0 = 1.0
    rho = 0.01
    alpha_global = 0.5
    alpha_local = 0.3
    beta_penalty = 0.03
    min_tau = 0.5
    max_tau = 20.0

    alpha_entropy = 0.02
    gamma_discount = 0.99

    eval_every_epochs = 2
    log_every = 20
    save_every_epochs = 5

    use_amp = True
    use_grad_scaler = False
    find_unused_parameters = True
    verbose_eval = True
    save_predictions = True
    max_verbose_episodes = 50

    use_llm_simulator = True
    use_hybrid_simulator = True

    llm_simulator_config = {
        "api_key": "sk-PovCrGTefqSW0POpIed5jFWF3HN6Cc95PXoWa70Zx1MKLNg4",
        "base_url": "http://172.22.2.242:3010/v1",
        "model": "deepseek-v3.1",
        "max_retries": 2,
        "timeout": 20.0,
        "cache_size": 5000,
        "enable_cache": True,
        "verbose": False,
    }





class TwoLayerToolSystem:

    def __init__(self, original_tools_path: str, extended_tools_path: str):
        with open(original_tools_path, 'r', encoding='utf-8') as f:
            original_tools = json.load(f)

        with open(extended_tools_path, 'r', encoding='utf-8') as f:
            extended_data = json.load(f)
        extended_tools = extended_data.get("tools", extended_data)

        self.original_tools = {}
        self.original_id_to_name = {}
        self.original_name_to_id = {}

        for idx, tool in enumerate(original_tools):
            name = tool.get("name")
            if name:
                self.original_tools[name] = {
                    "id": idx,
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
                self.original_id_to_name[idx] = name
                self.original_name_to_id[name] = idx

        self.n_original_tools = len(self.original_tools)

        self.extended_tools = {}
        self.extended_id_to_name = {}
        self.extended_name_to_id = {}
        self.original_to_variants: Dict[str, List[str]] = defaultdict(list)

        for tool in extended_tools:
            name = tool.get("name")
            original_name = tool.get("original_name", name)

            if name:
                self.extended_tools[name] = {
                    "id": tool.get("id"),
                    "name": name,
                    "original_name": original_name,
                    "actual_keys": tool.get("actual_keys", []),
                    "key_hash": tool.get("key_hash", ""),
                    "call_count": tool.get("call_count", 0),
                    "next_tools": tool.get("next_tools", {}),
                }
                self.extended_id_to_name[tool.get("id")] = name
                self.extended_name_to_id[name] = tool.get("id")
                self.original_to_variants[original_name].append(name)

        self.n_extended_tools = len(self.extended_tools)

        for original_name in self.original_to_variants:
            self.original_to_variants[original_name].sort(
                key=lambda v: self.extended_tools.get(v, {}).get("call_count", 0),
                reverse=True
            )

        log(f"[TwoLayerToolSystem] Initialized:")
        log(f"    Original tools: {self.n_original_tools}")
        log(f"    Extended tools: {self.n_extended_tools}")
        log(f"    Tools with variants: {len(self.original_to_variants)}")

    def get_variants(self, original_name: str) -> List[str]:
        return self.original_to_variants.get(original_name, [original_name])

    def get_best_variant(self, original_name: str, context_keys: Set[str] = None) -> str:
        variants = self.get_variants(original_name)

        if not variants:
            return original_name

        if context_keys is None:
            return variants[0]

        best_match = None
        best_score = -1

        for variant in variants:
            actual_keys = set(self.extended_tools.get(variant, {}).get("actual_keys", []))
            if not actual_keys:
                continue

            intersection = len(context_keys & actual_keys)
            union = len(context_keys | actual_keys)
            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_match = variant

        return best_match if best_match else variants[0]

    def original_name_from_extended(self, extended_name: str) -> str:
        return self.extended_tools.get(extended_name, {}).get("original_name", extended_name)

    def original_id_from_extended_id(self, extended_id: int) -> Optional[int]:
        ext_name = self.extended_id_to_name.get(extended_id, "")
        original_name = self.original_name_from_extended(ext_name)
        return self.original_name_to_id.get(original_name)



@dataclass
class TaskMemory:
    embedding: np.ndarray
    success_score: float
    timestamp: float
    task_hash: str


@dataclass
class ContextQueryResult:
    tau: float
    confidence: float
    n_activated: int
    max_similarity: float
    avg_score: float

    def is_confident(self, min_confidence: float = 0.3) -> bool:
        return self.confidence >= min_confidence


class ContextAwarePheromoneSystem:

    def __init__(self, cfg, device=None):
        self.cfg = cfg
        self.device = device

        if not cfg.context_pheromone_enabled or not SENTENCE_TRANSFORMER_AVAILABLE:
            self.enabled = False
            log("[ContextAwarePheromone] DISABLED")
            if not SENTENCE_TRANSFORMER_AVAILABLE:
                log("    Reason: sentence-transformers not available")
            return

        self.enabled = True

        log(f"[ContextAwarePheromone] Loading embedding model...")
        log(f"    Model: {cfg.embedding_model}")
        self.encoder = SentenceTransformer(cfg.embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        log(f"    Embedding dim: {self.embedding_dim}")

        self.edge_memory: Dict[Tuple[str, str], List[TaskMemory]] = defaultdict(list)

        self.max_memory_per_edge = cfg.max_memory_per_edge
        self.similarity_threshold = cfg.similarity_threshold
        self.time_decay = cfg.memory_time_decay
        self.base_tau = cfg.tau0
        self.min_score = cfg.memory_min_score

        self.confidence_threshold = getattr(cfg, 'context_confidence_threshold', 0.25)
        self.min_activated_memories = getattr(cfg, 'context_min_activated', 2)
        self.min_similarity_for_confidence = getattr(cfg, 'context_min_similarity', 0.35)
        self.min_avg_score_for_confidence = getattr(cfg, 'context_min_avg_score', 0.4)

        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_size = cfg.embedding_cache_size

        self.total_queries = 0
        self.cache_hits = 0
        self.total_updates = 0
        self.confident_queries = 0
        self.fallback_queries = 0

        log(f"[ContextAwarePheromone] Initialized:")
        log(f"    Max memory per edge: {self.max_memory_per_edge}")
        log(f"    Similarity threshold: {self.similarity_threshold}")
        log(f"    Confidence threshold: {self.confidence_threshold}")

    def encode_task(self, task_text: str) -> np.ndarray:
        if not self.enabled:
            return np.zeros(384)

        task_text = task_text[:512]

        task_hash = hashlib.md5(task_text.encode()).hexdigest()[:16]

        self.total_queries += 1

        if task_hash in self._embedding_cache:
            self.cache_hits += 1
            return self._embedding_cache[task_hash]

        with torch.no_grad():
            embedding = self.encoder.encode(task_text, convert_to_numpy=True)

        self._embedding_cache[task_hash] = embedding
        self._cache_order.append(task_hash)

        if len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            self._embedding_cache.pop(oldest, None)

        return embedding

    def get_context_tau_with_confidence(
        self,
        prev_tool: str,
        next_tool: str,
        task_embedding: np.ndarray
    ) -> ContextQueryResult:
        no_confidence = ContextQueryResult(
            tau=0.0,
            confidence=0.0,
            n_activated=0,
            max_similarity=0.0,
            avg_score=0.0
        )

        if not self.enabled:
            return no_confidence

        memories = self.edge_memory.get((prev_tool, next_tool), [])
        if not memories:
            return no_confidence

        memory_embeddings = np.stack([m.embedding for m in memories])
        memory_scores = np.array([m.success_score for m in memories])

        task_norm = np.linalg.norm(task_embedding) + 1e-8
        memory_norms = np.linalg.norm(memory_embeddings, axis=1) + 1e-8
        similarities = np.dot(memory_embeddings, task_embedding) / (memory_norms * task_norm)

        mask = similarities >= self.similarity_threshold
        if not np.any(mask):
            return no_confidence

        valid_sims = similarities[mask]
        valid_scores = memory_scores[mask]

        n_activated = len(valid_sims)
        max_sim = float(np.max(valid_sims))
        avg_score = float(np.mean(valid_scores))

        weighted_score = np.sum(valid_sims * valid_scores) / (np.sum(valid_sims) + 1e-8)
        context_tau = min(
            self.cfg.max_tau,
            self.base_tau + weighted_score * (self.cfg.max_tau - self.base_tau) * 0.8
        )

        confidence = min(1.0, n_activated / max(1, self.min_activated_memories))
        confidence *= max_sim
        confidence *= avg_score

        if max_sim < self.min_similarity_for_confidence:
            confidence *= 0.5
        if avg_score < self.min_avg_score_for_confidence:
            confidence *= 0.5
        if n_activated < self.min_activated_memories:
            confidence *= 0.7

        return ContextQueryResult(
            tau=context_tau,
            confidence=confidence,
            n_activated=n_activated,
            max_similarity=max_sim,
            avg_score=avg_score
        )

    def get_context_tau(
        self,
        prev_tool: str,
        next_tool: str,
        task_embedding: np.ndarray
    ) -> float:
        result = self.get_context_tau_with_confidence(prev_tool, next_tool, task_embedding)
        return result.tau if result.confidence > 0 else self.base_tau

    def update_memory(
        self,
        prev_tool: str,
        next_tool: str,
        task_text: str,
        success_score: float
    ):
        if not self.enabled:
            return

        if success_score < self.min_score:
            return

        self.total_updates += 1
        edge_key = (prev_tool, next_tool)

        task_embedding = self.encode_task(task_text)
        task_hash = hashlib.md5(task_text.encode()).hexdigest()[:16]

        for mem in self.edge_memory[edge_key]:
            if mem.task_hash == task_hash:
                mem.success_score = max(mem.success_score, success_score)
                mem.timestamp = time.time()
                return

        self.edge_memory[edge_key].append(TaskMemory(
            embedding=task_embedding.copy(),
            success_score=success_score,
            timestamp=time.time(),
            task_hash=task_hash
        ))

        if len(self.edge_memory[edge_key]) > self.max_memory_per_edge:
            self.edge_memory[edge_key].sort(key=lambda m: m.success_score, reverse=True)
            self.edge_memory[edge_key] = self.edge_memory[edge_key][:self.max_memory_per_edge]

    def update_trajectory_memory(
        self,
        transitions: List[Tuple[str, str]],
        task_text: str,
        success_score: float
    ):
        if not self.enabled:
            return

        if success_score < self.min_score:
            return

        for prev_tool, next_tool in transitions:
            self.update_memory(prev_tool, next_tool, task_text, success_score)

    def decay_memories(self):
        if not self.enabled:
            return

        for edge_key in list(self.edge_memory.keys()):
            for mem in self.edge_memory[edge_key]:
                mem.success_score *= self.time_decay

            self.edge_memory[edge_key] = [
                m for m in self.edge_memory[edge_key]
                if m.success_score >= 0.1
            ]

            if not self.edge_memory[edge_key]:
                del self.edge_memory[edge_key]

    def get_statistics(self) -> Dict:
        if not self.enabled:
            return {"enabled": False}

        total_memories = sum(len(v) for v in self.edge_memory.values())
        num_edges = len(self.edge_memory)

        all_scores = []
        for memories in self.edge_memory.values():
            for m in memories:
                all_scores.append(m.success_score)

        return {
            "enabled": True,
            "num_edges_with_memory": num_edges,
            "total_memories": total_memories,
            "avg_memories_per_edge": total_memories / max(1, num_edges),
            "avg_memory_score": float(np.mean(all_scores)) if all_scores else 0,
            "cache_hit_rate": self.cache_hits / max(1, self.total_queries),
            "total_updates": self.total_updates,
            "confident_queries": self.confident_queries,
            "fallback_queries": self.fallback_queries,
            "confidence_rate": self.confident_queries / max(1, self.confident_queries + self.fallback_queries),
        }

    def save(self, path: str):
        if not self.enabled:
            return

        data = {
            "edge_memories": {},
            "statistics": self.get_statistics(),
        }

        for edge_key, memories in self.edge_memory.items():
            key_str = f"{edge_key[0]}|||{edge_key[1]}"
            data["edge_memories"][key_str] = [
                {
                    "embedding": mem.embedding.tolist(),
                    "success_score": mem.success_score,
                    "timestamp": mem.timestamp,
                    "task_hash": mem.task_hash,
                }
                for mem in memories
            ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        if not self.enabled:
            return

        if not Path(path).exists():
            return

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.edge_memory.clear()

        for edge_str, memories in data.get("edge_memories", {}).items():
            parts = edge_str.split("|||")
            if len(parts) == 2:
                for mem_data in memories:
                    self.edge_memory[(parts[0], parts[1])].append(TaskMemory(
                        embedding=np.array(mem_data["embedding"]),
                        success_score=mem_data["success_score"],
                        timestamp=mem_data["timestamp"],
                        task_hash=mem_data["task_hash"],
                    ))





class ScalarPheromoneSystem:

    def __init__(self, tool_system: TwoLayerToolSystem, cfg):
        self.tool_system = tool_system
        self.cfg = cfg

        self.tau_tool: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: cfg.tau0)
        )

        self.tau_param: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: cfg.param_tau0)
        )

        self.tool_visit_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.tool_success_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.param_visit_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.param_success_count: Dict[Tuple[str, str], int] = defaultdict(int)

        self.tau0 = cfg.tau0
        self.rho = cfg.rho
        self.alpha_global = cfg.alpha_global
        self.alpha_local = cfg.alpha_local
        self.min_tau = cfg.min_tau
        self.max_tau = cfg.max_tau
        self.param_tau0 = cfg.param_tau0
        self.param_alpha = cfg.param_alpha

        self.protected_edges: Set[Tuple[str, str]] = set()

    def get_tool_tau(self, prev_tool: str, next_tool: str) -> float:
        return self.tau_tool[prev_tool][next_tool]

    def get_param_tau(self, tool: str, variant: str) -> float:
        return self.tau_param[tool][variant]

    def update_tool_pheromone(
        self,
        prev_tool: str,
        next_tool: str,
        success: bool,
        match_ratio: float,
        multiplier: float = 1.0
    ):
        self.tool_visit_count[(prev_tool, next_tool)] += 1

        if success:
            self.tool_success_count[(prev_tool, next_tool)] += 1
            delta = self.alpha_global * match_ratio * multiplier
            self.tau_tool[prev_tool][next_tool] = min(
                self.max_tau,
                self.tau_tool[prev_tool][next_tool] + delta
            )
        elif match_ratio > 0.3:
            delta = self.alpha_local * match_ratio * 0.5
            self.tau_tool[prev_tool][next_tool] = min(
                self.max_tau,
                self.tau_tool[prev_tool][next_tool] + delta
            )

    def update_param_pheromone(
        self,
        tool: str,
        variant: str,
        success: bool,
        match_ratio: float
    ):
        self.param_visit_count[(tool, variant)] += 1

        if success:
            self.param_success_count[(tool, variant)] += 1
            delta = self.param_alpha * match_ratio
            self.tau_param[tool][variant] = min(
                self.max_tau,
                self.tau_param[tool][variant] + delta
            )

    def evaporate(self):
        for prev in list(self.tau_tool.keys()):
            for next_tool in list(self.tau_tool[prev].keys()):
                self.tau_tool[prev][next_tool] = max(
                    self.min_tau,
                    self.tau_tool[prev][next_tool] * (1 - self.rho)
                )

        for tool in list(self.tau_param.keys()):
            for variant in list(self.tau_param[tool].keys()):
                self.tau_param[tool][variant] = max(
                    self.min_tau,
                    self.tau_param[tool][variant] * (1 - self.rho * 0.5)
                )

    def protected_evaporate(self):
        self.evaporate()

        for (prev, next_tool) in self.protected_edges:
            if self.tau_tool[prev][next_tool] < self.cfg.protected_edge_min_tau:
                self.tau_tool[prev][next_tool] = self.cfg.protected_edge_min_tau

    def get_statistics(self) -> Dict:
        tool_taus = [
            self.tau_tool[p][n]
            for p in self.tau_tool
            for n in self.tau_tool[p]
        ]
        param_taus = [
            self.tau_param[t][v]
            for t in self.tau_param
            for v in self.tau_param[t]
        ]

        return {
            "tool": {
                "num_edges": len(tool_taus),
                "mean": sum(tool_taus) / max(1, len(tool_taus)),
                "max": max(tool_taus) if tool_taus else 0,
                "min": min(tool_taus) if tool_taus else 0,
            },
            "param": {
                "num_edges": len(param_taus),
                "mean": sum(param_taus) / max(1, len(param_taus)),
                "max": max(param_taus) if param_taus else 0,
            },
            "visits": {
                "tool_total": sum(self.tool_visit_count.values()),
                "tool_success": sum(self.tool_success_count.values()),
            },
            "protected_edges": len(self.protected_edges),
        }

    def save(self, path: str):
        data = {
            "tau_tool": {k: dict(v) for k, v in self.tau_tool.items()},
            "tau_param": {k: dict(v) for k, v in self.tau_param.items()},
            "protected_edges": [list(e) for e in self.protected_edges],
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)



class HybridPheromoneSystem:

    def __init__(self, tool_system: TwoLayerToolSystem, cfg, device=None):
        self.tool_system = tool_system
        self.cfg = cfg
        self.device = device

        self.scalar = ScalarPheromoneSystem(tool_system, cfg)

        self.context = ContextAwarePheromoneSystem(cfg, device)

        self.context_weight = cfg.context_weight_initial

        self.confidence_threshold = getattr(cfg, 'context_confidence_threshold', 0.25)

        self._current_task_embedding: Optional[np.ndarray] = None
        self._current_task_text: Optional[str] = None

        self.hybrid_used = 0
        self.scalar_fallback = 0

        log(f"[HybridPheromone] Initialized:")
        log(f"    Context weight: {self.context_weight}")
        log(f"    Context enabled: {self.context.enabled}")
        log(f"    Confidence threshold: {self.confidence_threshold}")

    def set_current_task(self, task_text: str):
        if task_text == self._current_task_text:
            return

        self._current_task_text = task_text
        if self.context.enabled:
            self._current_task_embedding = self.context.encode_task(task_text)
        else:
            self._current_task_embedding = None

    def update_context_weight(self, global_epoch: int):
        if global_epoch >= self.cfg.context_weight_warmup_epochs:
            self.context_weight = self.cfg.context_weight_final
        else:
            progress = global_epoch / self.cfg.context_weight_warmup_epochs
            self.context_weight = (
                self.cfg.context_weight_initial +
                (self.cfg.context_weight_final - self.cfg.context_weight_initial) * progress
            )

    def get_tool_tau(
        self,
        prev_tool: str,
        next_tool: str,
        task_text: str = None
    ) -> float:
        scalar_tau = self.scalar.get_tool_tau(prev_tool, next_tool)

        if not self.context.enabled or task_text is None:
            return scalar_tau

        if task_text == self._current_task_text and self._current_task_embedding is not None:
            task_embedding = self._current_task_embedding
        else:
            task_embedding = self.context.encode_task(task_text)

        context_result = self.context.get_context_tau_with_confidence(
            prev_tool, next_tool, task_embedding
        )

        if not context_result.is_confident(self.confidence_threshold):
            self.scalar_fallback += 1
            self.context.fallback_queries += 1
            return scalar_tau

        self.hybrid_used += 1
        self.context.confident_queries += 1

        effective_weight = self.context_weight * min(1.0, context_result.confidence * 1.5)

        return (1 - effective_weight) * scalar_tau + effective_weight * context_result.tau

    def get_batch_tool_tau(
        self,
        prev_tool: str,
        candidate_tools: List[str],
        task_text: str = None
    ) -> torch.Tensor:
        device = self.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        taus = [
            self.get_tool_tau(prev_tool, t, task_text)
            for t in candidate_tools
        ]
        return torch.tensor(taus, device=device)

    def get_param_tau(self, tool: str, variant: str) -> float:
        return self.scalar.get_param_tau(tool, variant)

    def update_tool_pheromone(
        self,
        prev_tool: str,
        next_tool: str,
        success: bool,
        match_ratio: float,
        task_text: str = None,
        multiplier: float = 1.0
    ):
        self.scalar.update_tool_pheromone(
            prev_tool, next_tool, success, match_ratio, multiplier
        )

        if task_text and ((success and match_ratio >= 0.5) or match_ratio >= 0.7):
            self.context.update_memory(prev_tool, next_tool, task_text, match_ratio)

    def update_param_pheromone(
        self,
        tool: str,
        variant: str,
        success: bool,
        match_ratio: float
    ):
        self.scalar.update_param_pheromone(tool, variant, success, match_ratio)

    def update_trajectory(
        self,
        transitions: List[Tuple[str, str]],
        success: bool,
        match_ratio: float,
        task_text: str = None,
        multiplier: float = 1.0
    ):
        for prev_tool, next_tool in transitions:
            self.update_tool_pheromone(
                prev_tool, next_tool, success, match_ratio, task_text, multiplier
            )

    def evaporate(self):
        self.scalar.evaporate()
        if self.context.enabled:
            self.context.decay_memories()

    def protected_evaporate(self):
        self.scalar.protected_evaporate()
        if self.context.enabled:
            self.context.decay_memories()

    def get_statistics(self) -> Dict:
        total_queries = self.hybrid_used + self.scalar_fallback

        return {
            "scalar": self.scalar.get_statistics(),
            "context": self.context.get_statistics(),
            "context_weight": self.context_weight,
            "confidence_threshold": self.confidence_threshold,
            "hybrid_used": self.hybrid_used,
            "scalar_fallback": self.scalar_fallback,
            "hybrid_rate": self.hybrid_used / max(1, total_queries),
        }

    def save(self, path: str):
        scalar_path = path.replace(".json", "_scalar.json")
        self.scalar.save(scalar_path)

        if self.context.enabled:
            context_path = path.replace(".json", "_context.json")
            self.context.save(context_path)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "context_weight": self.context_weight,
                "scalar_path": scalar_path,
                "context_path": path.replace(".json", "_context.json") if self.context.enabled else None,
                "statistics": self.get_statistics(),
            }, f, indent=2, ensure_ascii=False)





class SuccessTransitionMatrix:

    def __init__(self):
        self.transition_count = defaultdict(int)
        self.success_count = defaultdict(int)
        self.sequence_count = defaultdict(int)

        self.total_trajectories = 0
        self.success_trajectories = 0

    def add_trajectory(self, transitions: List[Tuple[str, str]], success: bool):
        self.total_trajectories += 1

        for (prev, next_tool) in transitions:
            self.transition_count[(prev, next_tool)] += 1
            if success:
                self.success_count[(prev, next_tool)] += 1

        if success:
            self.success_trajectories += 1
            sequence = tuple(t[1] for t in transitions)
            self.sequence_count[sequence] += 1

    def get_golden_edges(
        self,
        min_count: int = 3,
        min_success_rate: float = 0.5
    ) -> List[Tuple[Tuple[str, str], float]]:
        golden = []

        for edge, count in self.transition_count.items():
            if count >= min_count:
                success_rate = self.success_count[edge] / count
                if success_rate >= min_success_rate:
                    golden.append((edge, success_rate))

        return sorted(golden, key=lambda x: x[1], reverse=True)

    def get_statistics(self) -> Dict:
        return {
            "total_trajectories": self.total_trajectories,
            "success_trajectories": self.success_trajectories,
            "success_rate": self.success_trajectories / max(1, self.total_trajectories),
            "unique_edges": len(self.transition_count),
            "unique_sequences": len(self.sequence_count),
        }



class PheromoneEnhancer:

    def __init__(self, cfg):
        self.cfg = cfg
        self.transition_matrix = SuccessTransitionMatrix()

    def record_trajectory(self, transitions: List[Tuple[str, str]], success: bool):
        self.transition_matrix.add_trajectory(transitions, success)

    def extract_golden_paths(self, pheromone: HybridPheromoneSystem) -> Dict:
        golden_edges = self.transition_matrix.get_golden_edges(
            min_count=self.cfg.golden_path_min_count,
            min_success_rate=self.cfg.golden_path_min_success
        )

        pheromone.scalar.protected_edges = set()

        for edge, rate in golden_edges:
            if self.transition_matrix.success_count[edge] >= self.cfg.protected_edge_threshold:
                pheromone.scalar.protected_edges.add(edge)

        return {
            "golden_edges": golden_edges,
            "protected_edges": len(pheromone.scalar.protected_edges),
            "matrix_stats": self.transition_matrix.get_statistics(),
        }

    def warmup_pheromone(
        self,
        pheromone: HybridPheromoneSystem,
        elite_buffer,
        n_rounds: int = 3,
        multiplier: float = 8.0
    ):
        log(f"[Pheromone Warmup] Starting {n_rounds} rounds, multiplier={multiplier}")

        total_enhanced = 0

        for round_idx in range(n_rounds):
            golden_edges = self.transition_matrix.get_golden_edges(
                min_count=self.cfg.golden_path_min_count,
                min_success_rate=self.cfg.golden_path_min_success
            )

            for (prev, next_tool), success_rate in golden_edges:
                delta = pheromone.scalar.alpha_global * multiplier * success_rate
                pheromone.scalar.tau_tool[prev][next_tool] = min(
                    pheromone.scalar.max_tau,
                    pheromone.scalar.tau_tool[prev][next_tool] + delta
                )
                total_enhanced += 1

            for traj in elite_buffer.success_buffer:
                for (prev, next_tool) in traj.transitions:
                    delta = pheromone.scalar.alpha_global * multiplier * traj.model_match_ratio
                    pheromone.scalar.tau_tool[prev][next_tool] = min(
                        pheromone.scalar.max_tau,
                        pheromone.scalar.tau_tool[prev][next_tool] + delta
                    )

        log(f"[Pheromone Warmup] Enhanced {total_enhanced} edges over {n_rounds} rounds")
        return total_enhanced

    def inject_experience(
        self,
        pheromone: HybridPheromoneSystem,
        elite_buffer,
        multiplier: float = 5.0
    ) -> int:
        n_injected = 0

        for traj in elite_buffer.success_buffer:
            for (prev, next_tool) in traj.transitions:
                delta = pheromone.scalar.alpha_global * multiplier * traj.model_match_ratio
                pheromone.scalar.tau_tool[prev][next_tool] = min(
                    pheromone.scalar.max_tau,
                    pheromone.scalar.tau_tool[prev][next_tool] + delta
                )
                n_injected += 1

        golden_edges = self.transition_matrix.get_golden_edges(
            min_count=self.cfg.golden_path_min_count,
            min_success_rate=self.cfg.golden_path_min_success
        )[:100]

        for (prev, next_tool), success_rate in golden_edges:
            delta = pheromone.scalar.alpha_global * multiplier * success_rate * 0.5
            pheromone.scalar.tau_tool[prev][next_tool] = min(
                pheromone.scalar.max_tau,
                pheromone.scalar.tau_tool[prev][next_tool] + delta
            )
            n_injected += 1

        return n_injected



class ToolSimulator:

    def __init__(self, database_path: str, tool_system: TwoLayerToolSystem):
        self.tool_system = tool_system
        self.outputs = {}

        if Path(database_path).exists():
            with open(database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for tool_name, tool_data in data.items():
                if isinstance(tool_data, dict):
                    outputs = tool_data.get("outputs", [])
                    if outputs:
                        self.outputs[tool_name] = outputs

        log(f"[ToolSimulator] Loaded outputs for {len(self.outputs)} tools")

    def get_output(self, variant_name: str, params: dict = None) -> str:
        if variant_name in self.outputs:
            outputs = self.outputs[variant_name]
            if isinstance(outputs, list) and outputs:
                return random.choice(outputs)
            return str(outputs)

        original_name = self.tool_system.original_name_from_extended(variant_name)
        if original_name in self.outputs:
            outputs = self.outputs[original_name]
            if isinstance(outputs, list) and outputs:
                return random.choice(outputs)
            return str(outputs)

        return f"[{variant_name}] executed successfully"



class ToolSelectionPolicy(nn.Module):

    def __init__(
        self,
        model_path: str,
        num_original_tools: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device=None
    ):
        super().__init__()

        self.num_tools = num_original_tools
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        self.model = get_peft_model(self.model, lora_config)

        hidden_size = self.model.config.hidden_size
        self.tool_head = nn.Linear(hidden_size, num_original_tools, dtype=torch.bfloat16)

        self.to(self.device)

    def encode_states(self, texts, max_length=512):
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return self.tool_head(last_hidden)

    def sample_action(
        self,
        state_text: str,
        prev_tool: str,
        pheromone: HybridPheromoneSystem,
        tool_system: TwoLayerToolSystem,
        task_text: str = None,
        gt_action: str = None,
        teacher_forcing_prob: float = 0.0,
        top_k: int = 20,
        beta_pheromone: float = 0.5,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        max_length: int = 512,
        is_first_step: bool = False,
        use_pheromone_first_step: bool = False
    ):
        if gt_action is not None and random.random() < teacher_forcing_prob:
            input_ids, attn = self.encode_states([state_text], max_length)
            logits = self.forward(input_ids, attn).squeeze(0)
            gt_id = tool_system.original_name_to_id.get(gt_action)
            if gt_id is not None:
                log_prob = F.log_softmax(logits, dim=-1)[gt_id].item()
                return gt_action, log_prob, logits, True

        input_ids, attn = self.encode_states([state_text], max_length)
        logits = self.forward(input_ids, attn).squeeze(0)

        policy_probs = F.softmax(logits / temperature, dim=-1)

        top_k_probs, top_k_ids = torch.topk(
            policy_probs,
            k=min(top_k, self.num_tools)
        )

        if is_first_step and not use_pheromone_first_step:
            fused_probs = top_k_probs
        else:
            candidate_tools = [
                tool_system.original_id_to_name.get(tid.item(), "")
                for tid in top_k_ids
            ]
            tau_values = pheromone.get_batch_tool_tau(prev_tool, candidate_tools, task_text)

            fused_probs = F.softmax(
                torch.log(top_k_probs + 1e-10) + beta_pheromone * torch.log(tau_values + 1e-10),
                dim=-1
            )

        if random.random() < epsilon:
            idx = random.randint(0, len(top_k_ids) - 1)
        else:
            idx = torch.multinomial(fused_probs, 1).item()

        action_id = top_k_ids[idx].item()
        action_name = tool_system.original_id_to_name.get(action_id, f"tool_{action_id}")
        log_prob = torch.log(fused_probs[idx] + 1e-10).item()

        return action_name, log_prob, logits, False

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")





class SmartParamMatcher:

    def __init__(
        self,
        tool_system: TwoLayerToolSystem,
        pheromone: HybridPheromoneSystem,
        cfg
    ):
        self.tool_system = tool_system
        self.pheromone = pheromone
        self.cfg = cfg

        self.success_history: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self.variant_success_rate: Dict[str, Tuple[int, int]] = defaultdict(
            lambda: (0, 0)
        )

        self.param_aliases = {
            "query": ["q", "search", "keyword", "text", "input"],
            "path": ["file", "filepath", "filename", "dir", "directory"],
            "url": ["link", "uri", "address"],
            "content": ["text", "body", "data", "message"],
            "name": ["title", "label", "id"],
            "limit": ["max", "count", "num", "size", "top"],
        }

        self.alias_to_canonical = {}
        for canonical, aliases in self.param_aliases.items():
            self.alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self.alias_to_canonical[alias] = canonical

    def _compute_context_hash(self, context_keys: Set[str]) -> str:
        if not context_keys:
            return ""
        return hashlib.md5(str(sorted(context_keys)).encode()).hexdigest()[:8]

    def _normalize_param_name(self, name: str) -> str:
        name_lower = name.lower().strip()
        return self.alias_to_canonical.get(name_lower, name_lower)

    def _semantic_similarity(self, keys1: Set[str], keys2: Set[str]) -> float:
        if not keys1 or not keys2:
            return 0.5

        norm1 = {self._normalize_param_name(k) for k in keys1}
        norm2 = {self._normalize_param_name(k) for k in keys2}

        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)
        jaccard = intersection / union if union > 0 else 0

        substring_bonus = 0
        for k1 in keys1:
            for k2 in keys2:
                if k1.lower() in k2.lower() or k2.lower() in k1.lower():
                    substring_bonus += 0.1
        substring_bonus = min(0.3, substring_bonus)

        return min(1.0, jaccard + substring_bonus)

    def _get_history_score(self, tool_name: str, variant: str, context_hash: str) -> float:
        history = self.success_history.get((tool_name, context_hash))
        if not history or variant not in history:
            return 0.5

        total = sum(history.values())
        return history[variant] / total if total > 0 else 0.5

    def _get_variant_success_rate(self, variant: str) -> float:
        success, total = self.variant_success_rate.get(variant, (0, 0))
        if total < 3:
            return 0.5
        return success / total

    def match_variant(
        self,
        tool_name: str,
        context_keys: Set[str] = None,
        use_pheromone: bool = True
    ) -> Tuple[str, float]:
        variants = self.tool_system.get_variants(tool_name)

        if len(variants) <= 1:
            return (variants[0] if variants else tool_name, 1.0)

        context_hash = self._compute_context_hash(context_keys) if context_keys else ""

        scores = []
        for variant in variants:
            tool_info = self.tool_system.extended_tools.get(variant, {})
            actual_keys = set(tool_info.get("actual_keys", []))

            freq_score = min(1.0, tool_info.get("call_count", 0) / 1000.0)

            if context_keys and actual_keys and self.cfg.smart_param_match:
                semantic_score = self._semantic_similarity(context_keys, actual_keys)
            else:
                semantic_score = 0.5

            if use_pheromone and self.cfg.param_pheromone_enabled:
                pheromone_score = min(
                    1.0,
                    self.pheromone.get_param_tau(tool_name, variant) / self.cfg.max_tau
                )
            else:
                pheromone_score = 0.5

            history_score = self._get_history_score(tool_name, variant, context_hash)

            success_rate_score = self._get_variant_success_rate(variant)

            total_score = (
                0.15 * freq_score +
                0.25 * semantic_score +
                0.20 * pheromone_score +
                0.20 * history_score +
                0.20 * success_rate_score
            )

            scores.append((variant, total_score))

        best_variant, best_score = max(scores, key=lambda x: x[1])
        return best_variant, best_score

    def update_history(
        self,
        tool_name: str,
        variant: str,
        context_keys: Set[str],
        success: bool,
        execution_success: bool
    ):
        context_hash = self._compute_context_hash(context_keys)

        if success:
            self.success_history[(tool_name, context_hash)][variant] += 1

        old_success, old_total = self.variant_success_rate.get(variant, (0, 0))
        new_success = old_success + (1 if execution_success else 0)
        new_total = old_total + 1
        self.variant_success_rate[variant] = (new_success, new_total)

    def get_top_k_variants(
        self,
        tool_name: str,
        context_keys: Set[str] = None,
        k: int = 3
    ) -> List[Tuple[str, float]]:
        variants = self.tool_system.get_variants(tool_name)

        if len(variants) <= k:
            return [(v, 1.0 / len(variants)) for v in variants]

        all_scores = []
        for variant in variants:
            tool_info = self.tool_system.extended_tools.get(variant, {})
            freq_score = min(1.0, tool_info.get("call_count", 0) / 1000.0)
            pheromone_score = min(
                1.0,
                self.pheromone.get_param_tau(tool_name, variant) / self.cfg.max_tau
            )
            total_score = 0.4 * freq_score + 0.6 * pheromone_score
            all_scores.append((variant, total_score))

        return sorted(all_scores, key=lambda x: x[1], reverse=True)[:k]


ParamMatcher = SmartParamMatcher



@dataclass
class EliteTrajectory:
    episode: Dict
    actions: List[str]
    variants: List[str]
    gt_actions: List[str]
    match_ratio: float
    model_match_ratio: float
    success: bool
    transitions: List[Tuple[str, str]]
    param_selections: List[Tuple[str, str]]
    rewards: List[float]
    states: List[str]
    tf_mask: List[bool]
    task_text: str = ""
    timestamp: float = field(default_factory=time.time)


class EliteTrajectoryBuffer:

    def __init__(
        self,
        max_size: int = 500,
        match_threshold: float = 0.5,
        min_model_match: float = 0.3
    ):
        self.max_size = max_size
        self.match_threshold = match_threshold
        self.min_model_match = min_model_match

        self.buffer: List[EliteTrajectory] = []
        self.success_buffer: List[EliteTrajectory] = []

        self.total_added = 0
        self.success_added = 0
        self.rejected_low_model_match = 0

    def add(self, trajectory: EliteTrajectory) -> bool:
        if trajectory.model_match_ratio < self.min_model_match:
            self.rejected_low_model_match += 1
            return False

        if trajectory.success:
            self.success_buffer.append(trajectory)
            self.success_added += 1

            if len(self.success_buffer) > self.max_size // 2:
                self.success_buffer = sorted(
                    self.success_buffer,
                    key=lambda x: x.model_match_ratio,
                    reverse=True
                )[:self.max_size // 2]

        if trajectory.match_ratio >= self.match_threshold:
            self.buffer.append(trajectory)
            self.total_added += 1

            if len(self.buffer) > self.max_size:
                self.buffer = sorted(
                    self.buffer,
                    key=lambda x: (x.success, x.model_match_ratio),
                    reverse=True
                )[:self.max_size]

        return True

    def sample(self, n: int) -> List[EliteTrajectory]:
        result = []

        if self.success_buffer:
            n_success = min(n // 2, len(self.success_buffer))
            if n_success > 0:
                result.extend(random.sample(self.success_buffer, n_success))
            n -= n_success

        if n > 0 and self.buffer:
            n_general = min(n, len(self.buffer))
            result.extend(random.sample(self.buffer, n_general))

        return result

    def get_statistics(self) -> Dict:
        return {
            "buffer_size": len(self.buffer),
            "success_buffer_size": len(self.success_buffer),
            "total_added": self.total_added,
            "success_added": self.success_added,
            "rejected_low_model_match": self.rejected_low_model_match,
            "avg_match_ratio": sum(t.match_ratio for t in self.buffer) / max(1, len(self.buffer)),
            "avg_model_match_ratio": sum(t.model_match_ratio for t in self.buffer) / max(1, len(self.buffer)),
        }



class TwoLayerRollout:

    def __init__(
        self,
        tool_system: TwoLayerToolSystem,
        simulator,
        param_matcher: SmartParamMatcher,
        max_steps: int = 10,
        max_history: int = 5
    ):
        self.tool_system = tool_system
        self.simulator = simulator
        self.param_matcher = param_matcher
        self.max_steps = max_steps
        self.max_history = max_history

        self.error_patterns = [
            "error", "failed", "exception", "invalid",
            "not found", "permission denied", "timeout",
            "connection refused", "unauthorized", "forbidden",
            "bad request"
        ]

    def build_state_text(
        self,
        task_name: str,
        user_prompt: str,
        history: List[Dict]
    ) -> str:
        lines = [
            f"Task: {task_name[:80]}",
            f"Query: {user_prompt[:300]}",
            "History:"
        ]

        if not history:
            lines.append("  (none)")
        else:
            for h in history[-self.max_history:]:
                name = h.get('name', 'unknown')[:30]
                output = str(h.get('output', ''))[:150]
                lines.append(f"  - {name}: {output}")

        lines.append("Next tool?")
        return "\n".join(lines)

    def _build_task_text(self, task_name: str, user_prompt: str) -> str:
        return f"{task_name}. {user_prompt}"[:512]

    def _same_category(self, name1: str, name2: str) -> bool:
        cat1 = name1.split("-")[0] if "-" in name1 else "other"
        cat2 = name2.split("-")[0] if "-" in name2 else "other"
        return cat1 == cat2

    def _is_execution_error(self, output: str) -> bool:
        output_lower = output.lower()
        return any(p in output_lower for p in self.error_patterns)

    def _compute_trajectory_bonus(
        self,
        actions: List[str],
        gt_actions: List[str],
        match_ratio: float,
        gt_success: bool,
        cfg
    ) -> float:
        if not gt_actions:
            return 0.0

        bonus = 0.0

        if match_ratio >= 0.9:
            bonus += cfg.reward_trajectory_complete * (1.0 if gt_success else 0.7)
        elif match_ratio >= 0.7:
            bonus += cfg.reward_high_match * (1.0 if gt_success else 0.8)
        elif match_ratio >= 0.5:
            bonus += cfg.reward_medium_match * (1.0 if gt_success else 0.8)
        elif match_ratio >= 0.3:
            bonus += cfg.reward_partial_match
        elif match_ratio >= 0.2:
            bonus += cfg.reward_partial_match * 0.5

        first_n = min(3, len(gt_actions))
        first_match = sum(
            1 for a, g in zip(actions[:first_n], gt_actions[:first_n])
            if a == g
        )
        if first_match == first_n:
            bonus += cfg.reward_first_steps_match

        return bonus

    def _lookahead_select(
        self,
        policy,
        pheromone,
        state_text: str,
        prev_tool: str,
        task_text: str,
        cfg,
        n_candidates: int = 3
    ) -> Tuple[str, str, float]:
        with torch.no_grad():
            input_ids, attn = policy.encode_states([state_text], cfg.max_seq_length)
            logits = policy(input_ids, attn).squeeze(0)
            policy_probs = F.softmax(logits / cfg.temperature, dim=-1)
            top_k_probs, top_k_ids = torch.topk(
                policy_probs,
                k=min(n_candidates, policy.num_tools)
            )

        candidates = []
        for i, tool_id in enumerate(top_k_ids):
            tool_name = self.tool_system.original_id_to_name.get(tool_id.item(), "")
            if not tool_name:
                continue

            variant, confidence = self.param_matcher.match_variant(
                tool_name,
                context_keys=None,
                use_pheromone=True
            )

            output = self.simulator.get_output(variant, {})
            is_error = self._is_execution_error(output)

            log_prob = torch.log(top_k_probs[i] + 1e-10).item()
            tau_bonus = 0.2 * (pheromone.get_tool_tau(prev_tool, tool_name, task_text) / cfg.max_tau)

            score = log_prob + (0.0 if is_error else 1.0) + 0.3 * confidence + tau_bonus

            candidates.append({
                'action': tool_name,
                'variant': variant,
                'log_prob': log_prob,
                'score': score
            })

        if not candidates:
            return "unknown", "unknown", 0.0

        best = max(candidates, key=lambda x: x['score'])
        return best['action'], best['variant'], best['log_prob']

    def rollout_episode(
        self,
        policy: ToolSelectionPolicy,
        pheromone: HybridPheromoneSystem,
        episode: Dict,
        cfg,
        teacher_forcing_prob: float = 0.0,
        max_steps_override: int = None,
        beta_override: float = None,
        success_key: str = None,
        use_lookahead: bool = False,
        record_details: bool = False
    ):
        task_name = episode.get("task_name", "")
        user_prompt = episode.get("user_prompt", "")
        gt_extended_ids = episode.get("tool_ids", [])
        gt_outputs = episode.get("output_texts", [])

        task_text = self._build_task_text(task_name, user_prompt)
        pheromone.set_current_task(task_text)

        gt_original_names = []
        for ext_id in gt_extended_ids:
            ext_name = self.tool_system.extended_id_to_name.get(ext_id, "")
            original_name = self.tool_system.original_name_from_extended(ext_name)
            gt_original_names.append(original_name)

        if gt_original_names:
            max_steps = min(
                max_steps_override or self.max_steps,
                len(gt_original_names)
            )
        else:
            max_steps = max_steps_override or self.max_steps

        beta_to_use = beta_override if beta_override is not None else cfg.beta_pheromone

        trajectory = {
            'actions': [],
            'variants': [],
            'log_probs': [],
            'rewards': [],
            'layer1_rewards': [],
            'layer2_rewards': [],
            'states': [],
            'outputs': [],
            'gt_actions': gt_original_names[:max_steps],
            'gt_actions_full': gt_original_names,
            'success': False,
            'match_ratio': 0.0,
            'model_match_ratio': 0.0,
            'transitions': [],
            'param_selections': [],
            'old_log_probs': [],
            'tf_mask': [],
            'execution_errors': [],
            'episode': episode,
            'task_text': task_text,
            'completion_markers': [],
            'step_details': [],
        }

        history = []
        prev_tool = "__START__"
        prev_was_wrong = False
        model_correct = 0
        model_total = 0

        for step in range(max_steps):
            state_text = self.build_state_text(task_name, user_prompt, history)
            trajectory['states'].append(state_text)

            gt_action = gt_original_names[step] if step < len(gt_original_names) else None
            is_first_step = (step == 0)

            if use_lookahead and cfg.lookahead_enabled and not is_first_step:
                action, variant, log_prob = self._lookahead_select(
                    policy, pheromone, state_text, prev_tool, task_text, cfg,
                    n_candidates=cfg.lookahead_candidates
                )
                used_tf = False
            else:
                with torch.no_grad():
                    action, log_prob, _, used_tf = policy.sample_action(
                        state_text, prev_tool, pheromone, self.tool_system,
                        task_text=task_text,
                        gt_action=gt_action,
                        teacher_forcing_prob=teacher_forcing_prob,
                        top_k=cfg.top_k,
                        beta_pheromone=beta_to_use,
                        temperature=cfg.temperature,
                        epsilon=cfg.epsilon_greedy,
                        max_length=cfg.max_seq_length,
                        is_first_step=is_first_step,
                        use_pheromone_first_step=cfg.first_step_use_pheromone
                    )

                variant_result = self.param_matcher.match_variant(
                    action,
                    context_keys=None,
                    use_pheromone=(not is_first_step)
                )
                if isinstance(variant_result, tuple):
                    variant, confidence = variant_result
                else:
                    variant, confidence = variant_result, 1.0

            trajectory['actions'].append(action)
            trajectory['variants'].append(variant)
            trajectory['log_probs'].append(log_prob)
            trajectory['old_log_probs'].append(log_prob)
            trajectory['transitions'].append((prev_tool, action))
            trajectory['param_selections'].append((action, variant))
            trajectory['tf_mask'].append(used_tf)

            if not used_tf:
                model_total += 1
                if action == gt_action:
                    model_correct += 1

            layer1_reward = 0.0
            if gt_action is not None:
                if action == gt_action:
                    layer1_reward = cfg.reward_layer1_intent
                    if prev_was_wrong:
                        layer1_reward += cfg.reward_recovery
                    prev_was_wrong = False
                elif self._same_category(action, gt_action):
                    layer1_reward = cfg.reward_same_category * 0.5
                    prev_was_wrong = True
                else:
                    layer1_reward = cfg.reward_wrong
                    prev_was_wrong = True
            trajectory['layer1_rewards'].append(layer1_reward)

            is_complete = False

            if step < len(gt_outputs) and action == gt_action:
                output = str(gt_outputs[step])[:200]
                execution_error = False
                is_complete = False
            else:
                sim_history = [
                    {"name": h["name"], "output": h["output"]}
                    for h in history
                ]

                if hasattr(self.simulator, 'get_output_with_completion'):
                    output, is_complete = self.simulator.get_output_with_completion(
                        variant_name=variant,
                        params={},
                        task_name=task_name,
                        user_prompt=user_prompt,
                        history=sim_history
                    )
                else:
                    output = self.simulator.get_output(variant, {})
                    is_complete = False

                execution_error = self._is_execution_error(output)

            trajectory['outputs'].append(output)
            trajectory['execution_errors'].append(execution_error)
            trajectory['completion_markers'].append(is_complete)

            if record_details:
                trajectory['step_details'].append({
                    'step': step,
                    'state_prompt': state_text,
                    'action': action,
                    'variant': variant,
                    'gt_action': gt_action,
                    'simulator_output': output,
                    'is_complete': is_complete,
                    'execution_error': execution_error,
                    'used_tf': used_tf,
                    'log_prob': log_prob,
                    'layer1_reward': layer1_reward,
                })

            if not execution_error:
                layer2_reward = cfg.reward_layer2_execution
            else:
                layer2_reward = cfg.reward_execution_error
            trajectory['layer2_rewards'].append(layer2_reward)

            trajectory['rewards'].append(layer1_reward + layer2_reward)

            if hasattr(self.param_matcher, 'update_history'):
                self.param_matcher.update_history(
                    action, variant, None,
                    success=(action == gt_action),
                    execution_success=(not execution_error)
                )

            history.append({"name": action, "output": output})
            prev_tool = action


        gt_episode_success = episode.get("success", 0) == 1

        if gt_original_names:
            gt_len = len(gt_original_names)
            horizon = len(trajectory['actions'])

            match_count = sum(
                1 for a, g in zip(trajectory['actions'], gt_original_names[:horizon])
                if a == g
            )
            prefix_match_ratio = match_count / max(1, min(horizon, gt_len))
            full_match_ratio = match_count / max(1, gt_len)

            trajectory['match_ratio'] = prefix_match_ratio
            trajectory['prefix_match_ratio'] = prefix_match_ratio
            trajectory['full_match_ratio'] = full_match_ratio

            prefix_exact_match = (trajectory['actions'] == gt_original_names[:horizon])
            full_exact_match = prefix_exact_match and (horizon == gt_len)

            trajectory['prefix_exact_match'] = prefix_exact_match
            trajectory['actions_match'] = full_exact_match

            success_threshold = getattr(cfg, 'success_match_threshold', 0.7)
            trajectory['success'] = (prefix_match_ratio >= success_threshold) and gt_episode_success
            trajectory['full_success'] = full_exact_match and gt_episode_success
            trajectory['gt_success'] = gt_episode_success

            trajectory['model_match_ratio'] = model_correct / max(1, model_total)

            if trajectory['rewards']:
                trajectory['rewards'][-1] += self._compute_trajectory_bonus(
                    trajectory['actions'],
                    gt_original_names,
                    trajectory['match_ratio'],
                    gt_episode_success,
                    cfg
                )

        trajectory['any_completion_marked'] = any(trajectory['completion_markers'])
        trajectory['final_completion_marked'] = (
            trajectory['completion_markers'][-1]
            if trajectory['completion_markers'] else False
        )

        return trajectory

    def rollout_with_lookahead(
        self,
        policy: ToolSelectionPolicy,
        pheromone: HybridPheromoneSystem,
        episode: Dict,
        cfg,
        max_steps_override: int = None,
        record_details: bool = False
    ):
        return self.rollout_episode(
            policy, pheromone, episode, cfg,
            teacher_forcing_prob=0.0,
            max_steps_override=max_steps_override,
            use_lookahead=True,
            record_details=record_details
        )



class StepDataset(Dataset):

    def __init__(
        self,
        episodes: List[Dict],
        tool_system: TwoLayerToolSystem,
        max_steps: int = 10,
        max_history: int = 5,
        only_success: bool = True
    ):
        self.samples = []
        skipped = 0
        skipped_failed = 0

        for ep in episodes:
            if only_success and ep.get("success", 0) != 1:
                skipped_failed += 1
                continue

            ext_tool_ids = ep.get("tool_ids", [])
            output_texts = ep.get("output_texts", [])
            task_name = ep.get("task_name", "")
            user_prompt = ep.get("user_prompt", "")

            if not ext_tool_ids:
                skipped += 1
                continue

            original_names = []
            for ext_id in ext_tool_ids:
                ext_name = tool_system.extended_id_to_name.get(ext_id, "")
                original_name = tool_system.original_name_from_extended(ext_name)
                original_names.append(original_name)

            if not all(name in tool_system.original_name_to_id for name in original_names if name):
                skipped += 1
                continue

            L = min(len(original_names), max_steps)
            for t in range(L):
                history = []
                for prev_t in range(max(0, t - max_history), t):
                    out_text = str(output_texts[prev_t] if prev_t < len(output_texts) else "")[:100]
                    history.append({
                        "name": original_names[prev_t],
                        "output": out_text
                    })

                lines = [
                    f"Task: {task_name[:80]}",
                    f"Query: {user_prompt[:300]}",
                    "History:"
                ]

                if not history:
                    lines.append("  (none)")
                else:
                    for h in history:
                        lines.append(f"  - {h['name'][:30]}: {h['output'][:100]}")

                lines.append("Next tool?")
                state_text = "\n".join(lines)

                action_id = tool_system.original_name_to_id.get(original_names[t])
                if action_id is not None:
                    self.samples.append({
                        "state": state_text,
                        "action": action_id
                    })

        log(f"[StepDataset] Created {len(self.samples)} samples")
        log(f"    Skipped: {skipped_failed} failed, {skipped} invalid")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_steps(batch):
    return {
        "states": [s["state"] for s in batch],
        "actions": torch.tensor([s["action"] for s in batch], dtype=torch.long)
    }


def split_episodes(episodes, train_ratio, seed):
    random.seed(seed)
    episode_list = list(episodes.values()) if isinstance(episodes, dict) else list(episodes)
    indices = list(range(len(episode_list)))
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return (
        [episode_list[i] for i in train_indices],
        [episode_list[i] for i in test_indices]
    )


def split_episodes_three_way(episodes, train_ratio, val_ratio, seed):
    random.seed(seed)
    shuffled = episodes.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        shuffled[:train_end],
        shuffled[train_end:val_end],
        shuffled[val_end:]
    )





def train_supervised(
    policy,
    policy_ddp,
    train_dataset,
    test_dataset,
    cfg,
    device,
    file_naming: FileNaming
):
    log("=" * 70)
    log("STAGE 1: Supervised Pretraining")
    log("=" * 70)
    log(f"    Tools: {policy.num_tools}")
    log(f"    Epochs: {cfg.sl_epochs}")
    log(f"    Target Accuracy: {cfg.sl_target_acc}")
    log(f"    Batch Size: {cfg.sl_batch_size}")
    log(f"    Learning Rate: {cfg.sl_lr}")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if get_world_size() > 1 else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if get_world_size() > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.sl_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_steps,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.sl_batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_steps,
        num_workers=2,
        pin_memory=True
    )

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.sl_lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, len(train_loader) // cfg.gradient_accumulation * 5),
        eta_min=cfg.min_lr
    )

    best_acc = 0.0

    for epoch in range(cfg.sl_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        policy_ddp.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            states = batch["states"]
            actions = batch["actions"].to(device)

            input_ids, attn = policy.encode_states(states, cfg.max_seq_length)

            with autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_amp):
                logits = policy_ddp(input_ids, attn)
                loss = F.cross_entropy(logits, actions, label_smoothing=0.1)

            (loss / cfg.gradient_accumulation).backward()
            accum_count += 1

            epoch_correct += (logits.argmax(-1) == actions).sum().item()
            epoch_total += len(actions)
            epoch_loss += loss.item()

            if accum_count >= cfg.gradient_accumulation:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            del input_ids, attn, logits, loss
            if batch_idx % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        epoch_correct = reduce_value(epoch_correct, average=False)
        epoch_total = reduce_value(epoch_total, average=False)
        epoch_loss = reduce_value(epoch_loss, average=True)
        train_acc = epoch_correct / max(1, epoch_total)

        policy_ddp.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                states = batch["states"]
                actions = batch["actions"].to(device)

                input_ids, attn = policy.encode_states(states, cfg.max_seq_length)
                logits = policy_ddp(input_ids, attn)

                test_correct += (logits.argmax(-1) == actions).sum().item()
                test_total += len(actions)

                del input_ids, attn

        test_correct = reduce_value(test_correct, average=False)
        test_total = reduce_value(test_total, average=False)
        test_acc = test_correct / max(1, test_total)

        log(f"[SL] Epoch {epoch+1:3d}/{cfg.sl_epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Loss: {epoch_loss/len(train_loader):.4f}")

        if is_main_process() and test_acc > best_acc:
            best_acc = test_acc
            torch.save(policy.state_dict(), file_naming.sl_best)
            log(f"[SL] Saved best model (Acc={best_acc:.4f})")

        barrier()

        if epoch >= cfg.sl_min_epochs and test_acc >= cfg.sl_target_acc:
            log(f"[SL] Reached target accuracy, stopping early")
            break

    log(f"[SL] Stage 1 completed. Best accuracy: {best_acc:.4f}")
    return best_acc




@torch.no_grad()
def evaluate_pure_model(
    policy,
    pheromone,
    rollout_manager,
    test_episodes,
    cfg,
    n_episodes=50,
    beta_override=None,
    use_lookahead=False,
    verbose=False,
    save_path=None
):
    if not is_main_process():
        return {
            'match_ratio': 0.0,
            'success_rate': 0.0,
            'actions_matched': 0.0,
            'gt_success_rate': 0.0
        }

    policy.eval()

    old_eps = cfg.epsilon_greedy
    cfg.epsilon_greedy = 0.0

    total_match = 0.0
    total_success = 0
    n_eval = 0
    total_exec_errors = 0
    total_steps = 0
    actions_matched = 0
    gt_was_success = 0

    total_any_completion = 0
    total_final_completion = 0

    all_predictions = []

    record_details = (save_path is not None)

    for ep_idx, episode in enumerate(test_episodes[:n_episodes]):
        if use_lookahead and cfg.lookahead_enabled:
            traj = rollout_manager.rollout_with_lookahead(
                policy, pheromone, episode, cfg,
                record_details=record_details
            )
        else:
            traj = rollout_manager.rollout_episode(
                policy, pheromone, episode, cfg,
                teacher_forcing_prob=0.0,
                beta_override=beta_override,
                record_details=record_details
            )

        total_match += traj['match_ratio']
        total_success += int(traj['success'])
        n_eval += 1

        if traj.get('actions_match', False):
            actions_matched += 1
        if traj.get('gt_success', False):
            gt_was_success += 1

        if 'execution_errors' in traj:
            total_exec_errors += sum(traj['execution_errors'])
            total_steps += len(traj['execution_errors'])

        if traj.get('any_completion_marked', False):
            total_any_completion += 1
        if traj.get('final_completion_marked', False):
            total_final_completion += 1

        if verbose and ep_idx < cfg.max_verbose_episodes:
            gt_str = ' -> '.join(traj.get('gt_actions', []))
            pred_str = ' -> '.join(traj.get('actions', []))
            completion_str = ' -> '.join([str(c) for c in traj.get('completion_markers', [])])
            print(f"\n[Eval] Episode {ep_idx+1}:")
            print(f"    Task: {episode.get('task_name', '')[:50]}")
            print(f"    GT:   {gt_str}")
            print(f"    Pred: {pred_str}")
            print(f"    Completion: {completion_str}")
            print(f"    Match: {traj['match_ratio']:.2%}, Success: {traj['success']}")
            print(f"    Final Complete: {traj.get('final_completion_marked', False)}")

        prediction_entry = {
            'episode_idx': ep_idx,
            'task_name': episode.get('task_name', ''),
            'user_prompt': episode.get('user_prompt', ''),
            'gt_tools': traj.get('gt_actions', []),
            'gt_tools_full': traj.get('gt_actions_full', []),
            'pred_tools': traj.get('actions', []),
            'pred_variants': traj.get('variants', []),
            'match_ratio': traj['match_ratio'],
            'prefix_match_ratio': traj.get('prefix_match_ratio', traj['match_ratio']),
            'full_match_ratio': traj.get('full_match_ratio', 0.0),
            'model_match_ratio': traj.get('model_match_ratio', 0.0),
            'success': traj['success'],
            'full_success': traj.get('full_success', False),
            'gt_success': traj.get('gt_success', False),
            'completion_markers': traj.get('completion_markers', []),
            'any_completion_marked': traj.get('any_completion_marked', False),
            'final_completion_marked': traj.get('final_completion_marked', False),
            'execution_errors': traj.get('execution_errors', []),
            'outputs': traj.get('outputs', []),
        }

        if record_details and 'step_details' in traj:
            prediction_entry['step_details'] = traj['step_details']

        all_predictions.append(prediction_entry)

    cfg.epsilon_greedy = old_eps
    policy.train()

    result = {
        'match_ratio': total_match / max(1, n_eval),
        'success_rate': total_success / max(1, n_eval),
        'actions_matched': actions_matched / max(1, n_eval),
        'gt_success_rate': gt_was_success / max(1, n_eval),
        'n_evaluated': n_eval,
        'any_completion_rate': total_any_completion / max(1, n_eval),
        'final_completion_rate': total_final_completion / max(1, n_eval),
    }

    if total_steps > 0:
        result['exec_error_rate'] = total_exec_errors / total_steps

    if save_path:
        output_data = {
            'summary': result,
            'config': {
                'n_episodes': n_episodes,
                'beta_override': beta_override,
                'use_lookahead': use_lookahead,
            },
            'predictions': all_predictions,
            'completion_analysis': {
                'total_episodes': n_eval,
                'episodes_with_any_completion': total_any_completion,
                'episodes_with_final_completion': total_final_completion,
                'any_completion_rate': total_any_completion / max(1, n_eval),
                'final_completion_rate': total_final_completion / max(1, n_eval),
                'final_completion_and_success': sum(
                    1 for p in all_predictions
                    if p.get('final_completion_marked') and p.get('success')
                ),
                'final_completion_but_failed': sum(
                    1 for p in all_predictions
                    if p.get('final_completion_marked') and not p.get('success')
                ),
                'success_without_completion': sum(
                    1 for p in all_predictions
                    if not p.get('final_completion_marked') and p.get('success')
                ),
            }
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        log(f"[Eval] Saved detailed predictions to: {save_path}")
        log(f"[Eval] Completion analysis: "
            f"any={total_any_completion}/{n_eval}, "
            f"final={total_final_completion}/{n_eval}")

    return result


def train_mixed_imitation_rl(
    policy,
    policy_ddp,
    pheromone,
    rollout_manager,
    train_episodes,
    test_episodes,
    cfg,
    device,
    elite_buffer,
    pheromone_enhancer,
    file_naming: FileNaming
):
    log("\n" + "=" * 70)
    log("STAGE 2: Mixed Imitation-RL Training")
    log("=" * 70)
    log(f"    Curriculum Stages: {len(cfg.curriculum_stages)}")
    log(f"    Episodes per Epoch: {cfg.mixed_episodes_per_epoch}")
    log(f"    Learning Rate: {cfg.mixed_lr}")

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.mixed_lr, weight_decay=0.01)

    best_match_ratio = 0.0
    global_epoch = 0
    world_size = get_world_size()
    rank = get_rank()

    use_ddp = world_size > 1

    for stage_idx, stage in enumerate(cfg.curriculum_stages):
        stage_max_steps = stage["max_steps"]
        stage_epochs = stage["epochs"]
        stage_tf_start = stage.get("tf_start", cfg.tf_initial)
        stage_tf_end = stage.get("tf_end", cfg.tf_final)

        log(f"\n[CURRICULUM] Stage {stage_idx+1}/{len(cfg.curriculum_stages)}: "
            f"max_steps={stage_max_steps}, epochs={stage_epochs}")

        stage_episodes = [
            ep for ep in train_episodes
            if ep.get("tool_ids") and len(ep["tool_ids"]) <= stage_max_steps
        ]

        if not stage_episodes:
            log(f"[CURRICULUM] No episodes for stage {stage_idx+1}, skipping")
            continue

        log(f"[CURRICULUM] Using {len(stage_episodes)} episodes")

        for epoch in range(stage_epochs):
            global_epoch += 1

            pheromone.update_context_weight(global_epoch)

            stage_progress = epoch / max(1, stage_epochs - 1)
            current_tf = stage_tf_start - (stage_tf_start - stage_tf_end) * stage_progress

            global_progress = min(1.0, global_epoch / cfg.tf_decay_epochs)
            current_sl_weight = cfg.sl_weight_initial - \
                (cfg.sl_weight_initial - cfg.sl_weight_final) * global_progress
            current_beta = min(
                cfg.beta_pheromone_max,
                cfg.beta_pheromone + (cfg.beta_pheromone_max - cfg.beta_pheromone) * global_progress
            )

            policy_ddp.train()

            epoch_total_match = 0.0
            epoch_model_match = 0.0
            epoch_success = 0
            n_trajectories = 0

            log(f"\n[MIXED] Epoch {global_epoch} | "
                f"TF={current_tf:.2f} | SL_w={current_sl_weight:.2f} | beta={current_beta:.2f}")

            n_episodes = (cfg.mixed_episodes_per_epoch // world_size) * world_size
            n_episodes_per_gpu = max(1, n_episodes // world_size)

            random.seed(cfg.random_seed + global_epoch)
            all_selected = random.sample(stage_episodes, min(n_episodes, len(stage_episodes)))
            while len(all_selected) < n_episodes:
                all_selected.extend(random.sample(
                    stage_episodes,
                    min(n_episodes - len(all_selected), len(stage_episodes))
                ))

            selected_episodes = all_selected[rank * n_episodes_per_gpu:(rank + 1) * n_episodes_per_gpu]

            for ep_idx, episode in enumerate(selected_episodes):
                trajectories = []
                with torch.no_grad():
                    for _ in range(cfg.num_rollouts):
                        traj = rollout_manager.rollout_episode(
                            policy, pheromone, episode, cfg,
                            teacher_forcing_prob=current_tf,
                            max_steps_override=stage_max_steps,
                            beta_override=current_beta
                        )
                        trajectories.append(traj)

                for traj in trajectories:
                    pheromone_enhancer.record_trajectory(traj['transitions'], traj['success'])

                    if traj['success'] or traj['match_ratio'] >= 0.5:
                        pheromone.context.update_trajectory_memory(
                            traj['transitions'],
                            traj.get('task_text', ''),
                            traj['match_ratio']
                        )

                    if current_tf < cfg.elite_min_tf_to_collect:
                        if traj['success'] or traj['match_ratio'] >= cfg.elite_match_threshold:
                            elite_buffer.add(EliteTrajectory(
                                episode=episode,
                                actions=traj['actions'],
                                variants=traj['variants'],
                                gt_actions=traj['gt_actions'],
                                match_ratio=traj['match_ratio'],
                                model_match_ratio=traj['model_match_ratio'],
                                success=traj['success'],
                                transitions=traj['transitions'],
                                param_selections=traj['param_selections'],
                                rewards=traj['rewards'],
                                states=traj['states'],
                                tf_mask=traj['tf_mask'],
                                task_text=traj.get('task_text', '')
                            ))

                            if traj['success']:
                                epoch_success += 1

                returns = []
                for traj in trajectories:
                    discounted_return = 0
                    for r in reversed(traj['rewards']):
                        discounted_return = r + cfg.gamma_discount * discounted_return
                    returns.append(discounted_return)

                baseline = sum(returns) / len(returns)
                std_returns = (sum((r - baseline) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8
                advantages = [(r - baseline) / std_returns for r in returns]

                all_steps = []
                for traj_idx, (traj, adv) in enumerate(zip(trajectories, advantages)):
                    for step_idx, (state, action, old_log_prob) in enumerate(
                        zip(traj['states'], traj['actions'], traj['old_log_probs'])
                    ):
                        gt_action = traj['gt_actions'][step_idx] if step_idx < len(traj['gt_actions']) else None
                        all_steps.append({
                            'state': state,
                            'action': action,
                            'old_log_prob': old_log_prob,
                            'gt_action': gt_action,
                            'adv': adv,
                            'traj_idx': traj_idx,
                        })

                optimizer.zero_grad()
                n_total_steps = len(all_steps)
                n_valid_steps = 0

                for step_i, step_data in enumerate(all_steps):
                    is_last_step = (step_i == n_total_steps - 1)

                    if use_ddp and not is_last_step:
                        sync_context = policy_ddp.no_sync()
                    else:
                        sync_context = contextlib.nullcontext()

                    with sync_context:
                        input_ids, attn = policy.encode_states([step_data['state']], cfg.max_seq_length)

                        with autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_amp):
                            logits = policy_ddp(input_ids, attn).squeeze(0)
                            log_probs = F.log_softmax(logits, dim=-1)
                            probs = F.softmax(logits, dim=-1)

                            action_id = rollout_manager.tool_system.original_name_to_id.get(step_data['action'])
                            if action_id is None:
                                del input_ids, attn, logits, log_probs, probs
                                continue

                            gt_id = rollout_manager.tool_system.original_name_to_id.get(step_data['gt_action']) if step_data['gt_action'] else None

                            if gt_id is not None:
                                sl_loss = F.cross_entropy(
                                    logits.unsqueeze(0),
                                    torch.tensor([gt_id], device=device)
                                )
                            else:
                                sl_loss = torch.tensor(0.0, device=device)

                            adv = step_data['adv']
                            ratio = torch.exp(log_probs[action_id] - step_data['old_log_prob'])
                            clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                            pg_loss = torch.max(-ratio * adv, -clipped_ratio * adv)

                            entropy = -(probs * log_probs).sum()

                            step_loss = (
                                current_sl_weight * sl_loss +
                                (1 - current_sl_weight) * pg_loss -
                                cfg.alpha_entropy * entropy
                            )

                            step_loss.backward()
                            n_valid_steps += 1

                        del input_ids, attn, logits, log_probs, probs
                        del sl_loss, pg_loss, entropy, step_loss

                if n_valid_steps > 0:
                    for p in policy.parameters():
                        if p.grad is not None:
                            p.grad.data.div_(n_valid_steps)

                    torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                for traj in trajectories:
                    pheromone.update_trajectory(
                        traj['transitions'],
                        traj['success'],
                        traj['match_ratio'],
                        traj.get('task_text', ''),
                        cfg.elite_pheromone_multiplier if traj['success'] else 1.0
                    )

                    for (tool, variant) in traj['param_selections']:
                        pheromone.update_param_pheromone(
                            tool, variant, traj['success'], traj['match_ratio']
                        )

                    epoch_total_match += traj['match_ratio']
                    epoch_model_match += traj['model_match_ratio']
                    n_trajectories += 1

                del trajectories, all_steps

                if ep_idx % 20 == 0:
                    pheromone.evaporate()

                if (ep_idx + 1) % cfg.log_every == 0 and is_main_process():
                    avg_total = epoch_total_match / max(1, n_trajectories)
                    avg_model = epoch_model_match / max(1, n_trajectories)
                    avg_success = epoch_success / max(1, n_trajectories // cfg.num_rollouts)
                    log(f"  Ep {ep_idx+1:4d} | TotalMatch: {avg_total:.3f} | ModelMatch: {avg_model:.3f} | Success: {avg_success:.3f}")

                if ep_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            barrier()

            if global_epoch % cfg.eval_every_epochs == 0:
                pure_eval = evaluate_pure_model(
                    policy, pheromone, rollout_manager, test_episodes, cfg,
                    n_episodes=50
                )

                if is_main_process():
                    log(f"[MIXED] Epoch {global_epoch} | Pure Test Match: {pure_eval['match_ratio']:.4f}")

                    if pure_eval['match_ratio'] > best_match_ratio:
                        best_match_ratio = pure_eval['match_ratio']
                        torch.save(policy.state_dict(), file_naming.mixed_best)
                        pheromone.save(str(file_naming.pheromone_mixed_best))
                        log(f"[MIXED] Saved best model (Match={best_match_ratio:.4f})")

    if is_main_process():
        golden_stats = pheromone_enhancer.extract_golden_paths(pheromone)
        log(f"[MIXED] Extracted golden paths: {golden_stats['protected_edges']} protected edges")
        log(f"[MIXED] Stage 2 completed. Best match ratio: {best_match_ratio:.4f}")

    barrier()
    return best_match_ratio



def train_rl_finetune(
    policy,
    policy_ddp,
    pheromone,
    rollout_manager,
    train_episodes,
    test_episodes,
    cfg,
    device,
    elite_buffer,
    pheromone_enhancer,
    file_naming: FileNaming,
    training_logger: TrainingLogger
):
    log("\n" + "=" * 70)
    log("STAGE 3: RL Fine-tuning with Step Logging (v12)")
    log("=" * 70)
    log(f"    Epochs: {cfg.rl_epochs}")
    log(f"    Beta: {cfg.rl_beta_initial} -> {cfg.rl_beta_final}")
    log(f"    Learning Rate: {cfg.rl_lr}")

    optimizer = optim.AdamW(policy.parameters(), lr=cfg.rl_lr, weight_decay=0.01)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    world_size = get_world_size()
    rank = get_rank()

    use_ddp = world_size > 1

    initial_eval = evaluate_pure_model(
        policy, pheromone, rollout_manager, test_episodes, cfg,
        n_episodes=50
    )
    best_match_ratio = initial_eval['match_ratio'] if is_main_process() else 0.0
    log(f"[RL] Initial match ratio: {best_match_ratio:.4f}")

    if is_main_process():
        pheromone_enhancer.warmup_pheromone(
            pheromone, elite_buffer,
            n_rounds=cfg.pheromone_warmup_rounds,
            multiplier=cfg.pheromone_warmup_mult
        )
    barrier()

    for epoch in range(cfg.rl_epochs):
        policy_ddp.train()

        total_epoch = cfg.mixed_epochs + epoch
        pheromone.update_context_weight(total_epoch)

        epoch_progress = min(1.0, epoch / max(1, cfg.rl_beta_decay_epochs))
        current_beta = cfg.rl_beta_initial - \
            (cfg.rl_beta_initial - cfg.rl_beta_final) * epoch_progress
        current_epsilon = max(0.01, cfg.epsilon_greedy * (1 - epoch / cfg.rl_epochs))

        success_rate = elite_buffer.success_added / max(1, elite_buffer.total_added)
        if success_rate < 0.1:
            adaptive_tf = min(0.3, cfg.rl_min_tf * 3)
        else:
            adaptive_tf = cfg.rl_min_tf

        epoch_total_match = 0.0
        epoch_model_match = 0.0
        epoch_success = 0
        n_trajectories = 0

        log(f"\n[RL] Epoch {epoch+1}/{cfg.rl_epochs} | "
            f"beta={current_beta:.2f} | eps={current_epsilon:.3f} | TF={adaptive_tf:.2f}")

        n_episodes = cfg.mixed_episodes_per_epoch
        n_elite = int(n_episodes * cfg.elite_replay_ratio)
        n_train = (n_episodes - n_elite) // world_size * world_size
        n_train_per_gpu = max(1, n_train // world_size)

        random.seed(cfg.random_seed + epoch + 1000)
        all_selected = random.sample(train_episodes, min(n_train, len(train_episodes)))
        while len(all_selected) < n_train:
            all_selected.extend(random.sample(
                train_episodes,
                min(n_train - len(all_selected), len(train_episodes))
            ))

        selected_episodes = all_selected[rank * n_train_per_gpu:(rank + 1) * n_train_per_gpu]

        elite_episodes = [et.episode for et in elite_buffer.sample(max(1, n_elite // world_size))]
        all_episodes = selected_episodes + elite_episodes
        random.shuffle(all_episodes)

        for ep_idx, episode in enumerate(all_episodes):
            old_eps = cfg.epsilon_greedy
            cfg.epsilon_greedy = current_epsilon

            trajectories = []
            with torch.no_grad():
                for _ in range(cfg.num_rollouts):
                    traj = rollout_manager.rollout_episode(
                        policy, pheromone, episode, cfg,
                        teacher_forcing_prob=adaptive_tf,
                        beta_override=current_beta
                    )
                    trajectories.append(traj)

            cfg.epsilon_greedy = old_eps

            for traj in trajectories:
                pheromone_enhancer.record_trajectory(traj['transitions'], traj['success'])

                if traj['success'] or traj['match_ratio'] >= 0.5:
                    pheromone.context.update_trajectory_memory(
                        traj['transitions'],
                        traj.get('task_text', ''),
                        traj['match_ratio']
                    )

                if traj['success'] or traj['match_ratio'] >= cfg.elite_match_threshold:
                    elite_buffer.add(EliteTrajectory(
                        episode=episode,
                        actions=traj['actions'],
                        variants=traj['variants'],
                        gt_actions=traj['gt_actions'],
                        match_ratio=traj['match_ratio'],
                        model_match_ratio=traj['model_match_ratio'],
                        success=traj['success'],
                        transitions=traj['transitions'],
                        param_selections=traj['param_selections'],
                        rewards=traj['rewards'],
                        states=traj['states'],
                        tf_mask=traj['tf_mask'],
                        task_text=traj.get('task_text', '')
                    ))

                    if traj['success']:
                        epoch_success += 1

            returns = []
            for traj in trajectories:
                discounted_return = 0
                for r in reversed(traj['rewards']):
                    discounted_return = r + cfg.gamma_discount * discounted_return
                returns.append(discounted_return)

            mean_return = sum(returns) / len(returns)
            std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5 + 1e-8
            advantages = [(r - mean_return) / std_return for r in returns]

            all_steps = []
            for traj_idx, (traj, adv) in enumerate(zip(trajectories, advantages)):
                for step_idx, (state, action, old_log_prob) in enumerate(
                    zip(traj['states'], traj['actions'], traj['old_log_probs'])
                ):
                    all_steps.append({
                        'state': state,
                        'action': action,
                        'old_log_prob': old_log_prob,
                        'adv': adv,
                        'traj_idx': traj_idx,
                    })

            optimizer.zero_grad()
            n_total_steps = len(all_steps)
            n_valid_steps = 0

            accumulated_loss = 0.0
            accumulated_pg_loss = 0.0
            accumulated_entropy = 0.0

            for step_i, step_data in enumerate(all_steps):
                is_last_step = (step_i == n_total_steps - 1)

                if use_ddp and not is_last_step:
                    sync_context = policy_ddp.no_sync()
                else:
                    sync_context = contextlib.nullcontext()

                with sync_context:
                    input_ids, attn = policy.encode_states([step_data['state']], cfg.max_seq_length)

                    with autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_amp):
                        logits = policy_ddp(input_ids, attn).squeeze(0)
                        log_probs = F.log_softmax(logits, dim=-1)
                        probs = F.softmax(logits, dim=-1)

                        action_id = rollout_manager.tool_system.original_name_to_id.get(step_data['action'])
                        if action_id is None:
                            del input_ids, attn, logits, log_probs, probs
                            continue

                        adv = step_data['adv']
                        ratio = torch.exp(log_probs[action_id] - step_data['old_log_prob'])
                        clipped_ratio = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio)
                        pg_loss = torch.max(-ratio * adv, -clipped_ratio * adv)

                        entropy = -(probs * log_probs).sum()

                        step_loss = pg_loss - cfg.alpha_entropy * entropy

                        accumulated_loss += step_loss.item()
                        accumulated_pg_loss += pg_loss.item()
                        accumulated_entropy += entropy.item()

                        step_loss.backward()
                        n_valid_steps += 1

                    del input_ids, attn, logits, log_probs, probs
                    del pg_loss, entropy, step_loss

            if n_valid_steps > 0:
                for p in policy.parameters():
                    if p.grad is not None:
                        p.grad.data.div_(n_valid_steps)

                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

                if is_main_process():
                    avg_match = sum(t['match_ratio'] for t in trajectories) / len(trajectories)
                    avg_model_match = sum(t['model_match_ratio'] for t in trajectories) / len(trajectories)
                    any_success = any(t['success'] for t in trajectories)

                    pher_stats = pheromone.get_statistics()

                    all_rewards = []
                    for t in trajectories:
                        all_rewards.extend(t['rewards'])
                    avg_immediate_reward = np.mean(all_rewards) if all_rewards else 0
                    min_immediate_reward = min(all_rewards) if all_rewards else 0
                    max_immediate_reward = max(all_rewards) if all_rewards else 0
                    avg_return_value = np.mean(returns)
                    traj_bonus = sum(t['rewards'][-1] for t in trajectories if t['rewards']) / max(1, len(trajectories))

                    max_h = max(len(t.get('actions', [])) for t in trajectories) if trajectories else 0
                    pos_acc_all = []
                    pos_acc_model = []
                    for t_pos in range(max_h):
                        c_all = n_all = c_m = n_m = 0
                        for tr in trajectories:
                            actions = tr.get('actions', [])
                            if t_pos >= len(actions):
                                continue
                            gt_seq = tr.get('gt_actions_full', tr.get('gt_actions', []))
                            if t_pos >= len(gt_seq):
                                continue
                            n_all += 1
                            if actions[t_pos] == gt_seq[t_pos]:
                                c_all += 1
                            tf_mask = tr.get('tf_mask', [])
                            if t_pos < len(tf_mask) and (not tf_mask[t_pos]):
                                n_m += 1
                                if actions[t_pos] == gt_seq[t_pos]:
                                    c_m += 1
                        pos_acc_all.append(c_all / n_all if n_all else 0.0)
                        pos_acc_model.append(c_m / n_m if n_m else 0.0)

                    total_all = total_all_correct = 0
                    total_model = total_model_correct = 0
                    for tr in trajectories:
                        actions = tr.get('actions', [])
                        gt_seq = tr.get('gt_actions_full', tr.get('gt_actions', []))
                        tf_mask = tr.get('tf_mask', [])
                        h = min(len(actions), len(gt_seq))
                        for i_step in range(h):
                            total_all += 1
                            if actions[i_step] == gt_seq[i_step]:
                                total_all_correct += 1
                            used_tf = tf_mask[i_step] if i_step < len(tf_mask) else False
                            if not used_tf:
                                total_model += 1
                                if actions[i_step] == gt_seq[i_step]:
                                    total_model_correct += 1
                    step_acc_mean_all = (total_all_correct / total_all) if total_all else 0.0
                    step_acc_mean_model = (total_model_correct / total_model) if total_model else 0.0

                    record = StepRecord(
                        global_step=training_logger.global_step,
                        epoch=epoch,
                        episode_idx=ep_idx,
                        loss=accumulated_loss / n_valid_steps,
                        pg_loss=accumulated_pg_loss / n_valid_steps,
                        entropy=accumulated_entropy / n_valid_steps,
                        match_ratio=avg_match,
                        model_match_ratio=avg_model_match,
                        success=any_success,
                        pheromone_tool_mean=pher_stats['scalar']['tool']['mean'],
                        pheromone_tool_max=pher_stats['scalar']['tool']['max'],
                        pheromone_tool_min=pher_stats['scalar']['tool']['min'],
                        pheromone_num_edges=pher_stats['scalar']['tool']['num_edges'],
                        pheromone_context_memories=pher_stats['context'].get('total_memories', 0),
                        pheromone_hybrid_rate=pher_stats.get('hybrid_rate', 0),
                        beta=current_beta,
                        teacher_forcing=adaptive_tf,
                        context_weight=pheromone.context_weight,
                        learning_rate=get_lr(),
                        step_acc_pos_all=pos_acc_all,
                        step_acc_pos_model=pos_acc_model,
                        step_acc_mean_all=step_acc_mean_all,
                        step_acc_mean_model=step_acc_mean_model,
                        step_acc_correct_all=total_all_correct,
                        step_acc_total_all=total_all,
                        step_acc_correct_model=total_model_correct,
                        step_acc_total_model=total_model,
                        avg_reward=avg_immediate_reward,
                        avg_return=avg_return_value,
                        trajectory_bonus=traj_bonus,
                        min_reward=min_immediate_reward,
                        max_reward=max_immediate_reward,
                    )
                    training_logger.log_step(record)
                    training_logger.log_pheromone_snapshot(pheromone, training_logger.global_step)

            optimizer.zero_grad()

            for traj in trajectories:
                pheromone.update_trajectory(
                    traj['transitions'],
                    traj['success'],
                    traj['match_ratio'],
                    traj.get('task_text', ''),
                    cfg.elite_pheromone_multiplier if traj['success'] else 1.0
                )

                for (tool, variant) in traj['param_selections']:
                    pheromone.update_param_pheromone(
                        tool, variant, traj['success'], traj['match_ratio']
                    )

                epoch_total_match += traj['match_ratio']
                epoch_model_match += traj['model_match_ratio']
                n_trajectories += 1

            del trajectories, all_steps

            if ep_idx % 20 == 0:
                pheromone.protected_evaporate()

            if (ep_idx + 1) % cfg.log_every == 0 and is_main_process():
                avg_total = epoch_total_match / max(1, n_trajectories)
                avg_model = epoch_model_match / max(1, n_trajectories)
                log(f"  Ep {ep_idx+1:4d} | TotalMatch: {avg_total:.3f} | ModelMatch: {avg_model:.3f}")

            if ep_idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        barrier()

        if (epoch + 1) % cfg.pheromone_inject_every == 0 and is_main_process():
            n_injected = pheromone_enhancer.inject_experience(
                pheromone, elite_buffer,
                multiplier=cfg.pheromone_inject_mult
            )
            log(f"[RL] Injected {n_injected} edges")

        pure_eval = evaluate_pure_model(
            policy, pheromone, rollout_manager, test_episodes, cfg,
            n_episodes=50,
            beta_override=current_beta
        )

        if is_main_process():
            log(f"[RL] Epoch {epoch+1} | Test Match: {pure_eval['match_ratio']:.4f} | Best: {best_match_ratio:.4f} | Success: {epoch_success}")

            if pure_eval['match_ratio'] > best_match_ratio:
                best_match_ratio = pure_eval['match_ratio']
                torch.save(policy.state_dict(), file_naming.rl_best)
                pheromone.save(str(file_naming.pheromone_rl_best))
                log(f"[RL] Saved best model")

            elite_stats = elite_buffer.get_statistics()
            pher_stats = pheromone.get_statistics()
            log(f"  Elite: buffer={elite_stats['buffer_size']}, success={elite_stats['success_buffer_size']}")
            log(f"  Scalar: tool_mean={pher_stats['scalar']['tool']['mean']:.3f}")
            log(f"  Context: memories={pher_stats['context'].get('total_memories', 0)}")

            if (epoch + 1) % cfg.save_every_epochs == 0:
                torch.save(policy.state_dict(), file_naming.rl_epoch(epoch + 1))
                training_logger.save()

    if is_main_process():
        training_logger.save()
        log(f"[RL] Stage 3 completed. Best match ratio: {best_match_ratio:.4f}")
        log(f"[RL] Total training steps: {training_logger.global_step}")

    return best_match_ratio



def main():
    rank, world_size, local_rank = setup_distributed()

    cfg = Config()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if world_size > 1:
        cfg.use_llm_simulator = False
        if rank == 0:
            print("[WARNING] DDP mode: LLM simulator disabled to prevent NCCL timeout")
            print("          Will use pool-based simulator for training")
            print("          LLM simulator can be used in single-GPU evaluation")

    file_naming = FileNaming(cfg.EXPERIMENT_NAME, cfg.project_root)

    if is_main_process():
        file_naming.output_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    log("=" * 70)
    log("GRPO-ACO v12 TRAINING")
    log("=" * 70)
    log(f"    Experiment: {cfg.EXPERIMENT_NAME}")
    log(f"    Output Dir: {file_naming.output_dir}")
    log(f"    World Size: {world_size}")
    log(f"    Device: {device}")
    log(f"    Model: {'7B' if cfg.USE_7B else '1.5B'}")
    log("=" * 70)

    random.seed(cfg.random_seed + rank)
    torch.manual_seed(cfg.random_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.random_seed + rank)

    training_logger = TrainingLogger(file_naming)

    log("[Init] Loading tool system...")
    tool_system = TwoLayerToolSystem(
        str(cfg.original_tools_path),
        str(cfg.extended_tools_path)
    )

    log("[Init] Creating pheromone system...")
    pheromone = HybridPheromoneSystem(tool_system, cfg, device)

    log("[Init] Loading simulator...")
    simulator = create_simulator_from_config(tool_system, cfg)

    param_matcher = SmartParamMatcher(tool_system, pheromone, cfg)

    log("[Init] Loading dataset...")
    with open(cfg.rl_dataset_path, 'r', encoding='utf-8') as f:
        episodes_data = json.load(f)

    if isinstance(episodes_data, dict):
        episodes = episodes_data.get("episodes", episodes_data)
    else:
        episodes = episodes_data

    log(f"[Init] Loaded {len(episodes)} episodes")

    original_count = len(episodes)
    episodes = [ep for ep in episodes if ep.get("success", 0) == 1]
    log(f"[Init] Filtered: {original_count} -> {len(episodes)} episodes (success only)")


    train_episodes, val_episodes, test_episodes = split_episodes_three_way(
        episodes, cfg.train_ratio, cfg.val_ratio, cfg.random_seed
    )
    full_train_episodes = train_episodes + val_episodes

    log(f"[Init] Train: {len(full_train_episodes)}, Test: {len(test_episodes)}")

    train_dataset = StepDataset(
        full_train_episodes, tool_system,
        cfg.max_steps_per_episode, cfg.max_history,
        only_success=True
    )
    test_dataset = StepDataset(
        test_episodes, tool_system,
        cfg.max_steps_per_episode, cfg.max_history,
        only_success=True
    )

    elite_buffer = EliteTrajectoryBuffer(
        max_size=cfg.elite_buffer_size,
        match_threshold=cfg.elite_match_threshold,
        min_model_match=cfg.elite_min_model_match
    )
    pheromone_enhancer = PheromoneEnhancer(cfg)

    log(f"[Init] Loading model... (GPU Memory: {gpu_mem():.2f}GB)")
    policy = ToolSelectionPolicy(
        cfg.model_path,
        tool_system.n_original_tools,
        cfg.lora_r,
        cfg.lora_alpha,
        device
    )
    policy.device = device

    if cfg.USE_7B:
        policy.model.gradient_checkpointing_enable()

    policy.print_trainable_parameters()
    log(f"[Init] Model loaded. GPU Memory: {gpu_mem():.2f}GB")

    if world_size > 1:
        policy_ddp = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.find_unused_parameters,
            broadcast_buffers=False,
            gradient_as_bucket_view=True
        )
        log("[Init] Model wrapped with DDP")
    else:
        policy_ddp = policy

    rollout_manager = TwoLayerRollout(
        tool_system, simulator, param_matcher,
        max_steps=cfg.max_steps_per_episode,
        max_history=cfg.max_history
    )


    sl_acc = train_supervised(
        policy, policy_ddp, train_dataset, test_dataset,
        cfg, device, file_naming
    )

    mixed_match = train_mixed_imitation_rl(
        policy, policy_ddp, pheromone, rollout_manager,
        full_train_episodes, test_episodes,
        cfg, device, elite_buffer, pheromone_enhancer,
        file_naming
    )

    rl_match = train_rl_finetune(
        policy, policy_ddp, pheromone, rollout_manager,
        full_train_episodes, test_episodes,
        cfg, device, elite_buffer, pheromone_enhancer,
        file_naming, training_logger
    )


    if is_main_process():
        log("\n" + "=" * 70)
        log("TRAINING COMPLETED - FINAL EVALUATION")
        log("=" * 70)

        best_model_path = file_naming.rl_best
        if not best_model_path.exists():
            best_model_path = file_naming.mixed_best
        if not best_model_path.exists():
            best_model_path = file_naming.sl_best

        if best_model_path.exists():
            log(f"[Final] Loading best model from: {best_model_path}")
            policy.load_state_dict(torch.load(best_model_path, map_location=device))

        policy.eval()

        log("\n[Final] Evaluating WITHOUT pheromone...")
        final_eval_no_pher = evaluate_pure_model(
            policy, pheromone, rollout_manager, test_episodes, cfg,
            n_episodes=100,
            beta_override=0.0
        )

        log("[Final] Evaluating WITH pheromone...")
        final_eval_with_pher = evaluate_pure_model(
            policy, pheromone, rollout_manager, test_episodes, cfg,
            n_episodes=100,
            beta_override=cfg.rl_beta_final,
            save_path=str(file_naming.predictions_detailed) if cfg.save_predictions else None
        )

        if cfg.lookahead_enabled and cfg.lookahead_in_eval:
            log("[Final] Evaluating WITH lookahead...")
            final_eval_lookahead = evaluate_pure_model(
                policy, pheromone, rollout_manager, test_episodes, cfg,
                n_episodes=100,
                beta_override=cfg.rl_beta_final,
                use_lookahead=True
            )
        else:
            final_eval_lookahead = None

        log("\n[Final] Computing Pass@K metrics...")
        pass_at_k_results = PassAtKEvaluator.evaluate_pass_at_k(
            policy, pheromone, rollout_manager, test_episodes, cfg,
            k_values=cfg.pass_at_k_values,
            n_episodes=cfg.pass_at_k_episodes,
            n_runs_per_episode=cfg.pass_at_k_runs,
            beta_override=cfg.rl_beta_final
        )

        log(f"\n{'='*70}")
        log(f"FINAL RESULTS - {cfg.EXPERIMENT_NAME}")
        log(f"{'='*70}")
        log(f"    Version: v12 Context-Aware Pheromone")
        log(f"    World Size: {world_size}")
        log(f"    ")
        log(f"    After SL:    Accuracy = {sl_acc:.4f}")
        log(f"    After Mixed: Match    = {mixed_match:.4f}")
        log(f"    After RL:    Match    = {rl_match:.4f}")
        log(f"    {'-'*50}")
        log(f"    Final (no pheromone):     Match = {final_eval_no_pher['match_ratio']:.4f}")
        log(f"    Final (with pheromone):   Match = {final_eval_with_pher['match_ratio']:.4f}")
        log(f"    Pheromone boost: +{(final_eval_with_pher['match_ratio'] - final_eval_no_pher['match_ratio']):.4f}")

        if final_eval_lookahead:
            log(f"    Final (with lookahead):   Match = {final_eval_lookahead['match_ratio']:.4f}")
            log(f"    Total boost: +{(final_eval_lookahead['match_ratio'] - final_eval_no_pher['match_ratio']):.4f}")

        log(f"    {'-'*50}")
        log(f"    Pass@1: {pass_at_k_results.get('pass@1', 0):.4f}")
        log(f"    Pass@3: {pass_at_k_results.get('pass@3', 0):.4f}")
        log(f"    {'-'*50}")
        log(f"    Avg Steps (all trajectories): {pass_at_k_results.get('avg_steps_all', 0):.2f}")
        log(f"    Avg Steps (successful only):  {pass_at_k_results.get('avg_steps_success', 0):.2f}")
        log(f"    Std Steps (successful):       {pass_at_k_results.get('std_steps_success', 0):.2f}")
        log(f"    {'-'*50}")
        log(f"    Total Training Steps (RL): {training_logger.global_step}")

        pher_stats = pheromone.get_statistics()
        log(f"\n【v12】Context-Aware Pheromone Statistics:")
        log(f"    Context weight: {pher_stats['context_weight']:.2f}")
        log(f"    Scalar edges: {pher_stats['scalar']['tool']['num_edges']}")
        log(f"    Context memories: {pher_stats['context'].get('total_memories', 0)}")
        log(f"    Hybrid rate: {pher_stats.get('hybrid_rate', 0):.2%}")

        pheromone.save(str(file_naming.pheromone_final))
        torch.save(policy.state_dict(), file_naming.policy_final)

        results = {
            "version": "v12",
            "experiment_name": cfg.EXPERIMENT_NAME,
            "world_size": world_size,
            "sl_accuracy": sl_acc,
            "mixed_match_ratio": mixed_match,
            "rl_match_ratio": rl_match,
            "final_match_no_pheromone": final_eval_no_pher['match_ratio'],
            "final_match_with_pheromone": final_eval_with_pher['match_ratio'],
            "pheromone_boost": final_eval_with_pher['match_ratio'] - final_eval_no_pher['match_ratio'],
            "context_pheromone_stats": pher_stats['context'],
            "config": {
                "context_weight_initial": cfg.context_weight_initial,
                "context_weight_final": cfg.context_weight_final,
                "similarity_threshold": cfg.similarity_threshold,
                "max_memory_per_edge": cfg.max_memory_per_edge,
                "embedding_model": cfg.embedding_model,
            },
            "pass_at_k": pass_at_k_results,
            "training_summary": training_logger.get_summary(),
            "rl_next_tool_acc_mean": training_logger.get_summary().get("stepwise_accuracy", {}).get("mean_model", 0.0),
        }

        if final_eval_lookahead:
            results["final_match_with_lookahead"] = final_eval_lookahead['match_ratio']

        with open(file_naming.results_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log(f"\n{'='*70}")
        log(f"All outputs saved to: {file_naming.output_dir}")
        log(f"    - Model:          {file_naming.policy_final.name}")
        log(f"    - Results:        {file_naming.results_json.name}")
        log(f"    - Training log:   {file_naming.training_log_json.name}")
        log(f"    - Training CSV:   {file_naming.training_log_csv.name}")
        log(f"    - Pheromone:      {file_naming.pheromone_evolution.name}")
        log(f"    - Loss curve:     {file_naming.loss_curve.name}")
        log(f"    - Accuracy curve: {file_naming.accuracy_curve.name}")
        try:
            swa = training_logger.get_summary().get('stepwise_accuracy', {})
            if swa:
                log(f"RL next-tool accuracy (model-only mean): {swa.get('mean_model', 0.0):.4f}")
        except Exception:
            pass

        log(f"{'='*70}")
        log("v12 Training Complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()