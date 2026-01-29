import json
import math
import random
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import torch
import torch.nn.functional as F



def load_mcp_graph(path: str) -> Tuple[int, Dict[str, int], Dict[int, str], Dict[int, Dict], Dict[int, Set[int]]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tools = data.get("tools", [])

    name_to_id = {}
    id_to_name = {}
    tool_info = {}

    for i, tool in enumerate(tools):
        tool_id = tool.get("id", i)
        name = tool.get("name")
        original_name = tool.get("original_name", name)

        if name is None:
            continue

        name_to_id[name] = tool_id
        id_to_name[tool_id] = name

        if original_name and original_name != name and original_name not in name_to_id:
            name_to_id[original_name] = tool_id

        params = tool.get("params", {})
        if not params:
            params = tool.get("parameters", {})

        parameters = {"type": "object", "properties": {}, "required": []}

        if isinstance(params, dict):
            if params and "type" not in params:
                for key, spec in params.items():
                    if isinstance(spec, dict):
                        parameters["properties"][key] = {
                            "type": spec.get("type", "string"),
                            "description": spec.get("description", ""),
                        }
                        if spec.get("required", False):
                            parameters["required"].append(key)
            else:
                parameters = params

        tool_info[tool_id] = {
            "name": name,
            "original_name": original_name,
            "description": tool.get("description", ""),
            "parameters": parameters,
            "actual_keys": tool.get("actual_keys", []),
            "key_hash": tool.get("key_hash", ""),
            "call_count": tool.get("call_count", 0),
        }

    adjacency: Dict[int, Set[int]] = defaultdict(set)

    for tool in tools:
        tool_id = tool.get("id", tools.index(tool) if tool in tools else None)
        if tool_id is None:
            continue

        next_tools = tool.get("next_tools", {})
        for next_name in next_tools.keys():
            next_id = name_to_id.get(next_name)
            if next_id is not None:
                adjacency[tool_id].add(next_id)

    print(f"[load_mcp_graph] Loaded {len(tools)} tools")
    print(f"[load_mcp_graph] name_to_id entries: {len(name_to_id)}")
    print(f"[load_mcp_graph] Adjacency entries: {len(adjacency)}")

    return len(tools), name_to_id, id_to_name, tool_info, dict(adjacency)


def load_rl_dataset(path: str, name_to_id: Dict[str, int]) -> Tuple[Dict, List[Dict]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    raw_episodes = data.get("episodes", [])

    episodes = []
    skipped_empty = 0
    skipped_invalid = 0

    for ep in raw_episodes:
        tool_ids = ep.get("tool_ids", [])

        if tool_ids:
            if all(isinstance(tid, int) for tid in tool_ids):
                episodes.append(ep)
            else:
                skipped_invalid += 1
        else:
            tool_names = ep.get("tool_names", [])
            if not tool_names:
                skipped_empty += 1
                continue

            converted_ids = []
            valid = True
            for name in tool_names:
                tid = name_to_id.get(name)
                if tid is None:
                    valid = False
                    break
                converted_ids.append(tid)

            if valid and converted_ids:
                ep_copy = {**ep, "tool_ids": converted_ids}
                episodes.append(ep_copy)
            else:
                skipped_invalid += 1

    print(f"[load_rl_dataset] Raw episodes: {len(raw_episodes)}")
    print(f"[load_rl_dataset] Valid episodes: {len(episodes)}")
    print(f"[load_rl_dataset] Skipped (empty): {skipped_empty}")
    print(f"[load_rl_dataset] Skipped (invalid): {skipped_invalid}")

    if episodes:
        all_tool_ids = set()
        for ep in episodes:
            all_tool_ids.update(ep.get("tool_ids", []))
        print(f"[load_rl_dataset] Tool ID range: {min(all_tool_ids)} - {max(all_tool_ids)}")

    return meta, episodes



class ToolOutputSimulator:

    def __init__(self, database_path: str = None):
        self.database = {}
        self.embedder = None
        self.name_to_id = {}
        self.id_to_name = {}
        self.has_embeddings = False

        if database_path:
            self.load(database_path)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        meta = data.get("meta", {})
        self.has_embeddings = meta.get("has_embeddings", False)

        tools_data = data.get("tools", {})

        for tool_name, tool_info in tools_data.items():
            tool_id = tool_info.get("tool_id", len(self.name_to_id))
            self.name_to_id[tool_name] = tool_id
            self.id_to_name[tool_id] = tool_name

            schemas = tool_info.get("schemas", {})
            self.database[tool_name] = schemas

        print(f"[Simulator] Loaded {len(self.database)} tools, embeddings={self.has_embeddings}")

    def get_output(self, tool_name: str, arguments: Any) -> str:
        if tool_name not in self.database:
            return f"Tool {tool_name} executed successfully."

        schemas = self.database[tool_name]
        if not schemas:
            return f"Tool {tool_name} executed successfully."

        query_keys = set()
        if isinstance(arguments, dict):
            query_keys = set(arguments.keys())
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    query_keys = set(parsed.keys())
            except:
                pass

        query_hash = hashlib.md5(str(sorted(query_keys)).encode()).hexdigest()[:12]

        if query_hash in schemas:
            calls = schemas[query_hash].get("calls", [])
            if calls:
                return random.choice(calls).get("output", "Success")

        best_schema = None
        best_overlap = -1
        for schema_hash, schema_data in schemas.items():
            schema_keys = set(schema_data.get("keys", []))
            overlap = len(query_keys & schema_keys)
            if overlap > best_overlap:
                best_overlap = overlap
                best_schema = schema_data

        if best_schema:
            calls = best_schema.get("calls", [])
            if calls:
                return random.choice(calls).get("output", "Success")

        return f"Tool {tool_name} executed successfully."

    def get_available_schemas(self, tool_name: str) -> List[Tuple[str, List[str], float]]:
        if tool_name not in self.database:
            return []

        result = []
        for schema_hash, schema_data in self.database[tool_name].items():
            keys = schema_data.get("keys", [])
            pheromone = schema_data.get("pheromone", 1.0)
            result.append((schema_hash, keys, pheromone))

        return result

    def get_statistics(self) -> Dict:
        num_patterns = sum(len(schemas) for schemas in self.database.values())
        total_outputs = sum(
            len(schema.get("calls", []))
            for schemas in self.database.values()
            for schema in schemas.values()
        )
        return {
            "num_tools": len(self.database),
            "num_param_patterns": num_patterns,
            "total_outputs": total_outputs,
            "has_embeddings": self.has_embeddings,
        }



class ParameterEvaluator:

    def __init__(self, tool_info: Dict[int, Dict]):
        self.tool_info = tool_info

    def evaluate(self, tool_id: int, arguments: Any) -> Tuple[float, Dict]:
        if tool_id not in self.tool_info:
            return 0.5, {"status": "unknown_tool"}

        info = self.tool_info[tool_id]
        expected_keys = set(info.get("actual_keys", []))

        if not expected_keys:
            return 0.5, {"status": "no_schema"}

        provided_keys = set()
        if isinstance(arguments, dict):
            provided_keys = set(arguments.keys())
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    provided_keys = set(parsed.keys())
            except:
                pass

        if not provided_keys:
            return 0.3, {"status": "no_args"}

        overlap = len(expected_keys & provided_keys)
        union = len(expected_keys | provided_keys)

        if union == 0:
            return 0.5, {"status": "empty"}

        jaccard = overlap / union
        return jaccard, {"jaccard": jaccard, "overlap": overlap, "union": union}



class HierarchicalPheromoneSystemV3:

    def __init__(
        self,
        num_tools: int,
        adjacency: Dict[int, Set[int]] = None,
        tau0: float = 1.0,
        rho: float = 0.02,
        alpha_global: float = 0.5,
        alpha_local: float = 0.2,
        beta_penalty: float = 0.05,
        min_tau: float = 0.1,
        max_tau: float = 20.0,
    ):
        self.num_tools = num_tools
        self.adjacency = adjacency or {}
        self.tau0 = tau0
        self.rho = rho
        self.alpha_global = alpha_global
        self.alpha_local = alpha_local
        self.beta_penalty = beta_penalty
        self.min_tau = min_tau
        self.max_tau = max_tau

        self.tau_tool = defaultdict(lambda: defaultdict(lambda: tau0))

        self.tau_param = defaultdict(lambda: defaultdict(lambda: tau0))

        self.visit_count = defaultdict(int)
        self.success_count = defaultdict(int)


    def set_adjacency(self, adjacency: Dict[int, Set[int]]):
        self.adjacency = adjacency
        print(f"[Pheromone] Set adjacency for {len(adjacency)} tools")

    def get_reachable_tools(self, from_tool: int) -> Set[int]:
        if from_tool in self.adjacency:
            return self.adjacency[from_tool]
        return set(range(self.num_tools))

    def is_edge_valid(self, from_tool: int, to_tool: int) -> bool:
        if not self.adjacency:
            return True
        if from_tool not in self.adjacency:
            return True
        return to_tool in self.adjacency[from_tool]


    def initialize_param_pheromone_from_simulator(
        self,
        simulator: ToolOutputSimulator,
        use_frequency: bool = True,
    ):
        count = 0
        for tool_name, schemas in simulator.database.items():
            tool_id = simulator.name_to_id.get(tool_name)
            if tool_id is None:
                continue

            for schema_hash, schema_data in schemas.items():
                if use_frequency:
                    pheromone = schema_data.get("pheromone", self.tau0)
                else:
                    pheromone = self.tau0

                self.tau_param[tool_id][schema_hash] = pheromone
                count += 1

        print(f"[Pheromone] Initialized {count} param patterns")

    def initialize_tool_pheromone_uniform(self):
        self.tau_tool = defaultdict(lambda: defaultdict(lambda: self.tau0))
        self.visit_count = defaultdict(int)
        self.success_count = defaultdict(int)
        print(f"[Pheromone] Tool pheromone initialized to uniform (tau0={self.tau0})")

    def initialize_tool_pheromone_with_warmup(
        self,
        episodes: List[Dict],
        name_to_id: Dict[str, int],
        start_tool_id: int,
        warmup_weight: float = 0.1,
    ):
        transition_count = defaultdict(lambda: defaultdict(int))

        for ep in episodes:
            tool_ids = ep.get("tool_ids", [])
            if not tool_ids:
                continue

            prev = start_tool_id
            for tid in tool_ids:
                transition_count[prev][tid] += 1
                prev = tid

        for prev, nexts in transition_count.items():
            total = sum(nexts.values())
            for next_tool, count in nexts.items():
                adjustment = warmup_weight * math.log(1 + count)
                self.tau_tool[prev][next_tool] = self.tau0 * (1 + adjustment)

        print(f"[Pheromone] Tool pheromone warm-started with weight={warmup_weight}")


    def get_tool_tau(self, prev_tool: int, next_tool: int) -> float:
        return self.tau_tool[prev_tool][next_tool]

    def get_tool_tau_distribution(
        self,
        prev_tool: int,
        temperature: float = 1.0,
        use_adjacency_mask: bool = False,
    ) -> torch.Tensor:
        tau_values = torch.zeros(self.num_tools)

        if use_adjacency_mask and prev_tool in self.adjacency:
            reachable = self.adjacency[prev_tool]
            for next_tool in range(self.num_tools):
                if next_tool in reachable:
                    tau_values[next_tool] = self.tau_tool[prev_tool][next_tool]
                else:
                    tau_values[next_tool] = self.min_tau * 0.01
        else:
            for next_tool in range(self.num_tools):
                tau_values[next_tool] = self.tau_tool[prev_tool][next_tool]

        log_tau = torch.log(tau_values.clamp(min=1e-10))
        return F.softmax(log_tau / temperature, dim=-1)

    def get_param_tau(self, tool_id: int, schema_hash: str) -> float:
        return self.tau_param[tool_id][schema_hash]

    def get_param_tau_distribution(
        self,
        tool_id: int,
        available_schemas: List[Tuple[str, List[str], float]],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        if not available_schemas:
            return {}

        tau_values = {}
        for schema_hash, keys, _ in available_schemas:
            tau_values[schema_hash] = self.tau_param[tool_id][schema_hash]

        max_tau = max(tau_values.values())
        exp_values = {k: math.exp((v - max_tau) / temperature) for k, v in tau_values.items()}
        total = sum(exp_values.values())

        return {k: v / total for k, v in exp_values.items()}


    def deposit_episode(
        self,
        transitions: List[Tuple[int, int]],
        param_hashes: List[Tuple[int, str]],
        step_advantages: List[float],
        success: bool,
        trajectory_quality: float = 1.0,
    ):
        if not transitions:
            return

        for (prev, next_tool) in transitions:
            self.visit_count[(prev, next_tool)] += 1
            if success:
                self.success_count[(prev, next_tool)] += 1

        if success:
            for (prev, next_tool) in transitions:
                delta = self.alpha_global * trajectory_quality
                self.tau_tool[prev][next_tool] = min(
                    self.max_tau,
                    self.tau_tool[prev][next_tool] + delta
                )
        else:
            for (prev, next_tool) in transitions:
                delta = self.beta_penalty * trajectory_quality
                self.tau_tool[prev][next_tool] = max(
                    self.min_tau,
                    self.tau_tool[prev][next_tool] - delta
                )

        for (prev, next_tool), adv in zip(transitions, step_advantages):
            if adv > 0:
                delta = self.alpha_local * adv
                self.tau_tool[prev][next_tool] = min(
                    self.max_tau,
                    self.tau_tool[prev][next_tool] + delta
                )
            else:
                delta = self.alpha_local * abs(adv) * 0.5
                self.tau_tool[prev][next_tool] = max(
                    self.min_tau,
                    self.tau_tool[prev][next_tool] - delta
                )

        if success:
            for (tool_id, phash), adv in zip(param_hashes, step_advantages):
                if phash and adv > 0:
                    delta = self.alpha_local * adv * 0.5
                    self.tau_param[tool_id][phash] = min(
                        self.max_tau,
                        self.tau_param[tool_id][phash] + delta
                    )

    def evaporate(self):
        for prev in list(self.tau_tool.keys()):
            for next_tool in list(self.tau_tool[prev].keys()):
                old_val = self.tau_tool[prev][next_tool]
                new_val = (1 - self.rho) * old_val + self.rho * self.tau0
                self.tau_tool[prev][next_tool] = max(self.min_tau, new_val)

        for tool_id in list(self.tau_param.keys()):
            for phash in list(self.tau_param[tool_id].keys()):
                old_val = self.tau_param[tool_id][phash]
                new_val = (1 - self.rho) * old_val + self.rho * self.tau0
                self.tau_param[tool_id][phash] = max(self.min_tau, new_val)


    def _hash_params(self, args: Any) -> str:
        keys = set()
        if isinstance(args, dict):
            keys = set(args.keys())
        elif isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    keys = set(parsed.keys())
            except:
                pass

        if not keys:
            return ""
        return hashlib.md5(str(sorted(keys)).encode()).hexdigest()[:12]

    def get_statistics(self) -> Dict:
        tool_values = []
        for prev in self.tau_tool:
            for next_tool in self.tau_tool[prev]:
                tool_values.append(self.tau_tool[prev][next_tool])

        param_values = []
        for tool_id in self.tau_param:
            for phash in self.tau_param[tool_id]:
                param_values.append(self.tau_param[tool_id][phash])

        return {
            "tool": {
                "num_edges": len(tool_values),
                "mean": np.mean(tool_values) if tool_values else self.tau0,
                "std": np.std(tool_values) if tool_values else 0,
                "min": min(tool_values) if tool_values else self.tau0,
                "max": max(tool_values) if tool_values else self.tau0,
            },
            "param": {
                "num_patterns": len(param_values),
                "mean": np.mean(param_values) if param_values else self.tau0,
                "min": min(param_values) if param_values else self.tau0,
                "max": max(param_values) if param_values else self.tau0,
            },
            "visits": {
                "total": sum(self.visit_count.values()),
                "success": sum(self.success_count.values()),
            }
        }

    def get_exploration_stats(self) -> Dict:
        visited_edges = len(self.visit_count)
        total_possible = self.num_tools * self.num_tools

        success_edges = len([k for k, v in self.success_count.items() if v > 0])

        return {
            "visited_edges": visited_edges,
            "total_possible_edges": total_possible,
            "coverage": visited_edges / max(1, total_possible),
            "success_edges": success_edges,
        }

    def save(self, path: str):
        data = {
            "tau0": self.tau0,
            "rho": self.rho,
            "tau_tool": {str(k): dict(v) for k, v in self.tau_tool.items()},
            "tau_param": {str(k): dict(v) for k, v in self.tau_param.items()},
            "visit_count": {str(k): v for k, v in self.visit_count.items()},
            "success_count": {str(k): v for k, v in self.success_count.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)

        self.tau0 = data.get("tau0", self.tau0)
        self.rho = data.get("rho", self.rho)

        for k, v in data.get("tau_tool", {}).items():
            prev = int(k)
            for next_tool, tau in v.items():
                self.tau_tool[prev][int(next_tool)] = tau

        for k, v in data.get("tau_param", {}).items():
            tool_id = int(k)
            for phash, tau in v.items():
                self.tau_param[tool_id][phash] = tau



def compute_grpo_aco_loss_v3(
    policy_logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    tau_distribution: torch.Tensor,
    alpha_entropy: float = 0.01,
    beta_kl: float = 0.05,
    adjacency_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict]:
    if adjacency_mask is not None:
        policy_logits = policy_logits.masked_fill(~adjacency_mask, float('-inf'))

    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)

    action_log_probs = log_policy.gather(1, actions.unsqueeze(1)).squeeze(1)
    pg_loss = -(action_log_probs * advantages).mean()

    tau_dist = tau_distribution.clamp(min=1e-10)
    kl_div = (policy * (log_policy - torch.log(tau_dist))).sum(dim=-1).mean()

    entropy = -(policy * log_policy).sum(dim=-1).mean()

    total_loss = pg_loss + beta_kl * kl_div - alpha_entropy * entropy

    return total_loss, {
        "pg_loss": pg_loss.item(),
        "kl_div": kl_div.item(),
        "entropy": entropy.item(),
        "total_loss": total_loss.item(),
    }



def validate_tool_ids(episodes: List[Dict], num_tools: int) -> bool:
    for ep in episodes:
        for tid in ep.get("tool_ids", []):
            if not (0 <= tid < num_tools):
                print(f"[ERROR] Invalid tool_id: {tid}, expected 0..{num_tools-1}")
                return False
    return True



def build_system_v3(
    mcp_graph_path: str,
    rl_dataset_path: str,
    simulator_database_path: str,
    pheromone_config: Dict = None,
    warmup_pheromone: bool = False,
    warmup_weight: float = 0.1,
) -> Tuple[HierarchicalPheromoneSystemV3, ToolOutputSimulator, ParameterEvaluator, Dict]:
    pheromone_config = pheromone_config or {}

    num_tools, name_to_id, id_to_name, tool_info, adjacency = load_mcp_graph(mcp_graph_path)

    meta, episodes = load_rl_dataset(rl_dataset_path, name_to_id)

    start_tool_id = num_tools
    n_tools = num_tools + 1

    print(f"[System] Loaded {num_tools} tools, {len(episodes)} episodes")
    print(f"[System] Graph structure: {len(adjacency)} tools have adjacency info")

    simulator = ToolOutputSimulator(simulator_database_path)
    print(f"[System] Simulator: {simulator.get_statistics()}")

    pheromone = HierarchicalPheromoneSystemV3(
        num_tools=n_tools,
        adjacency=adjacency,
        tau0=pheromone_config.get("tau0", 1.0),
        rho=pheromone_config.get("rho", 0.02),
        alpha_global=pheromone_config.get("alpha_global", 0.5),
        alpha_local=pheromone_config.get("alpha_local", 0.2),
        beta_penalty=pheromone_config.get("beta_penalty", 0.05),
    )

    if warmup_pheromone:
        pheromone.initialize_tool_pheromone_with_warmup(
            episodes, name_to_id, start_tool_id, warmup_weight
        )
    else:
        pheromone.initialize_tool_pheromone_uniform()

    pheromone.initialize_param_pheromone_from_simulator(simulator)

    print(f"[System] Pheromone: {pheromone.get_statistics()}")

    evaluator = ParameterEvaluator(tool_info)

    meta_extended = {
        **meta,
        "num_tools": num_tools,
        "n_tools": n_tools,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "tool_info": tool_info,
        "episodes": episodes,
        "start_tool_id": start_tool_id,
        "adjacency": adjacency,
    }

    return pheromone, simulator, evaluator, meta_extended