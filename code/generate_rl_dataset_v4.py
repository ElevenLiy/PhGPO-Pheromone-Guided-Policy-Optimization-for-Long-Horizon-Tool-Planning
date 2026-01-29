import json
import hashlib
import os
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm


def compute_param_hash(args: Any) -> str:
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


def deduplicate_consecutive(
    tool_ids: List[int],
    tool_names: List[str],
    tool_args: List[str],
    output_texts: List[str],
    param_hashes: List[str],
) -> Tuple[List[int], List[str], List[str], List[str], List[str], int]:
    if not tool_ids:
        return [], [], [], [], [], 0

    new_ids = [tool_ids[0]]
    new_names = [tool_names[0]]
    new_args = [tool_args[0] if tool_args else "{}"]
    new_outputs = [output_texts[0] if output_texts else ""]
    new_hashes = [param_hashes[0] if param_hashes else ""]

    removed_count = 0

    for i in range(1, len(tool_ids)):
        if tool_ids[i] != tool_ids[i - 1]:
            new_ids.append(tool_ids[i])
            new_names.append(tool_names[i] if i < len(tool_names) else "")
            new_args.append(tool_args[i] if i < len(tool_args) else "{}")
            new_outputs.append(output_texts[i] if i < len(output_texts) else "")
            new_hashes.append(param_hashes[i] if i < len(param_hashes) else "")
        else:
            removed_count += 1

    return new_ids, new_names, new_args, new_outputs, new_hashes, removed_count


def load_mcp_graph_v2(path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[int, str], int]:
    with open(path, 'r') as f:
        data = json.load(f)

    tools = data.get("tools", [])

    tool_variant_map: Dict[str, Dict[str, int]] = defaultdict(dict)
    id_to_name: Dict[int, str] = {}

    for tool in tools:
        tool_id = tool.get("id")
        name = tool.get("name")
        original_name = tool.get("original_name", name)
        key_hash = tool.get("key_hash", "")

        if tool_id is None or name is None:
            continue

        id_to_name[tool_id] = name

        if key_hash:
            tool_variant_map[original_name][key_hash] = tool_id
        else:
            tool_variant_map[original_name][""] = tool_id

    print(f"[load_mcp_graph_v2] Loaded {len(tools)} tools")
    print(f"[load_mcp_graph_v2] Unique original names: {len(tool_variant_map)}")

    multi_variant = sum(1 for v in tool_variant_map.values() if len(v) > 1)
    print(f"[load_mcp_graph_v2] Tools with multiple variants: {multi_variant}")

    return dict(tool_variant_map), id_to_name, len(tools)


def resolve_tool_id(
    tool_name: str,
    arguments: Any,
    tool_variant_map: Dict[str, Dict[str, int]],
) -> Optional[int]:
    if tool_name not in tool_variant_map:
        return None

    variants = tool_variant_map[tool_name]

    param_hash = compute_param_hash(arguments)

    if param_hash in variants:
        return variants[param_hash]

    if "" in variants:
        return variants[""]

    return next(iter(variants.values()))


def parse_arguments(args_str: str) -> Dict:
    if not args_str:
        return {}

    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass

    return {}


def process_trajectory_file(
    filepath: str,
    tool_variant_map: Dict[str, Dict[str, int]],
    enable_dedup: bool = True,
) -> Tuple[List[Dict], int, int]:
    episodes = []
    total_removed = 0
    total_original = 0

    with open(filepath, 'r') as f:
        for line in f:
            try:
                traj = json.loads(line.strip())
            except:
                continue

            task_name = traj.get("task_name", "")

            task_status_str = traj.get("task_status", "{}")
            try:
                task_status = json.loads(task_status_str) if isinstance(task_status_str, str) else task_status_str

                running_done = task_status.get("running") == "done"
                evaluation_passed = task_status.get("evaluation") == True
                success = 1 if (running_done and evaluation_passed) else 0

            except:
                success = 0

            messages_str = traj.get("messages", "[]")
            try:
                messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str
            except:
                messages = []

            if not isinstance(messages, list):
                continue

            user_prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_prompt = content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_prompt = item.get("text", "")
                                break
                    break

            tool_ids = []
            tool_names = []
            tool_args = []
            output_texts = []
            param_hashes = []

            tool_call_map = {}

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                role = msg.get("role", "")

                if role == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue

                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue

                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        if not isinstance(func, dict):
                            continue

                        name = func.get("name", "")
                        args_str = func.get("arguments", "")

                        if not name:
                            continue

                        args = parse_arguments(args_str)

                        tool_id = resolve_tool_id(name, args, tool_variant_map)

                        if tool_id is None:
                            continue

                        tool_ids.append(tool_id)
                        tool_names.append(name)
                        tool_args.append(args_str if args_str else "{}")
                        param_hashes.append(compute_param_hash(args))

                        tool_call_map[tc_id] = len(tool_ids) - 1

                elif role == "tool":
                    tc_id = msg.get("tool_call_id", "")
                    content = msg.get("content", "")

                    if isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)

                    output_texts.append(str(content)[:500])

            while len(output_texts) < len(tool_ids):
                output_texts.append("")
            output_texts = output_texts[:len(tool_ids)]

            if not tool_ids:
                continue

            original_len = len(tool_ids)
            total_original += original_len

            if enable_dedup:
                (tool_ids, tool_names, tool_args, output_texts, param_hashes, removed) = \
                    deduplicate_consecutive(tool_ids, tool_names, tool_args, output_texts, param_hashes)
                total_removed += removed

            episodes.append({
                "task_name": task_name,
                "user_prompt": user_prompt[:2000],
                "success": int(success),
                "tool_ids": tool_ids,
                "tool_names": tool_names,
                "tool_args": tool_args,
                "output_texts": output_texts,
                "param_hashes": param_hashes,
                "original_length": original_len,
                "dedup_length": len(tool_ids),
            })

    return episodes, total_removed, total_original


def convert_from_v2(v2_path: str, output_path: str):
    print("=" * 70)
    print("Converting RL Dataset V2 -> V3")
    print("=" * 70)

    with open(v2_path, 'r') as f:
        v2_data = json.load(f)

    v2_episodes = v2_data.get("episodes", [])
    print(f"[Input] Loaded {len(v2_episodes)} episodes from V2")

    v3_episodes = []
    total_removed = 0
    total_original = 0

    for ep in tqdm(v2_episodes, desc="Deduplicating"):
        tool_ids = ep.get("tool_ids", [])
        tool_names = ep.get("tool_names", [])
        tool_args = ep.get("tool_args", [])
        output_texts = ep.get("output_texts", [])
        param_hashes = ep.get("param_hashes", [])

        original_len = len(tool_ids)
        total_original += original_len

        (new_ids, new_names, new_args, new_outputs, new_hashes, removed) = \
            deduplicate_consecutive(tool_ids, tool_names, tool_args, output_texts, param_hashes)

        total_removed += removed

        v3_episodes.append({
            "task_name": ep.get("task_name", ""),
            "user_prompt": ep.get("user_prompt", ""),
            "success": ep.get("success", 0),
            "tool_ids": new_ids,
            "tool_names": new_names,
            "tool_args": new_args,
            "output_texts": new_outputs,
            "param_hashes": new_hashes,
            "original_length": original_len,
            "dedup_length": len(new_ids),
        })

    print(f"\n[Dedup Stats]")
    print(f"  Original tool calls: {total_original}")
    print(f"  After dedup:         {total_original - total_removed}")
    print(f"  Removed:             {total_removed} ({100*total_removed/total_original:.1f}%)")

    avg_original = sum(ep["original_length"] for ep in v3_episodes) / len(v3_episodes)
    avg_dedup = sum(ep["dedup_length"] for ep in v3_episodes) / len(v3_episodes)
    print(f"  Avg length before:   {avg_original:.2f}")
    print(f"  Avg length after:    {avg_dedup:.2f}")

    success_count = sum(1 for ep in v3_episodes if ep["success"])

    output_data = {
        "meta": {
            "version": "v3",
            "description": "Deduplicated consecutive tool calls",
            "num_episodes": len(v3_episodes),
            "num_tools": v2_data.get("meta", {}).get("num_tools", 0),
            "success_episodes": success_count,
            "source": "Converted from V2",
            "success_criteria": "running=='done' AND evaluation==true",
            "features": [
                "tool_variants",
                "param_hashes",
                "consecutive_dedup",
            ],
            "dedup_stats": {
                "original_tool_calls": total_original,
                "after_dedup": total_original - total_removed,
                "removed": total_removed,
                "removal_rate": f"{100*total_removed/total_original:.1f}%",
                "avg_length_before": round(avg_original, 2),
                "avg_length_after": round(avg_dedup, 2),
            },
        },
        "episodes": v3_episodes,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved] {output_path}")
    print("=" * 70)


def generate_rl_dataset_v3(
    trajectories_dir: str,
    mcp_graph_path: str,
    output_path: str,
):
    print("=" * 70)
    print("Generating RL Dataset V3")
    print("Feature: Deduplicate consecutive tool calls")
    print("=" * 70)

    tool_variant_map, id_to_name, num_tools = load_mcp_graph_v2(mcp_graph_path)

    trajectories_dir = Path(trajectories_dir)
    all_episodes = []
    total_removed = 0
    total_original = 0

    jsonl_files = list(trajectories_dir.glob("*.jsonl"))
    print(f"\n[Processing] Found {len(jsonl_files)} trajectory files")

    for filepath in tqdm(jsonl_files, desc="Processing trajectories"):
        episodes, removed, original = process_trajectory_file(
            str(filepath), tool_variant_map, enable_dedup=True
        )
        all_episodes.extend(episodes)
        total_removed += removed
        total_original += original

    print(f"\n[Result] Generated {len(all_episodes)} episodes")

    print(f"\n[Dedup Stats]")
    print(f"  Original tool calls: {total_original}")
    print(f"  After dedup:         {total_original - total_removed}")
    print(f"  Removed:             {total_removed} ({100*total_removed/max(1,total_original):.1f}%)")

    if all_episodes:
        avg_original = sum(ep["original_length"] for ep in all_episodes) / len(all_episodes)
        avg_dedup = sum(ep["dedup_length"] for ep in all_episodes) / len(all_episodes)
        print(f"  Avg length before:   {avg_original:.2f}")
        print(f"  Avg length after:    {avg_dedup:.2f}")
    else:
        avg_original = avg_dedup = 0

    success_count = sum(1 for ep in all_episodes if ep["success"])
    total_tools_after = sum(len(ep["tool_ids"]) for ep in all_episodes)
    unique_tools = set()
    for ep in all_episodes:
        unique_tools.update(ep["tool_ids"])

    print(f"\n[General Stats]")
    print(f"  Success episodes: {success_count}/{len(all_episodes)} ({100*success_count/max(1,len(all_episodes)):.1f}%)")
    print(f"  Total tool calls (after dedup): {total_tools_after}")
    print(f"  Unique tools used: {len(unique_tools)}")
    if unique_tools:
        print(f"  Tool ID range: {min(unique_tools)} - {max(unique_tools)}")

    output_data = {
        "meta": {
            "version": "v3",
            "description": "Deduplicated consecutive tool calls",
            "num_episodes": len(all_episodes),
            "num_tools": num_tools,
            "success_episodes": success_count,
            "source": "Toolathlon-Trajectories",
            "success_criteria": "running=='done' AND evaluation==true",
            "features": [
                "tool_variants",
                "param_hashes",
                "consecutive_dedup",
            ],
            "dedup_stats": {
                "original_tool_calls": total_original,
                "after_dedup": total_original - total_removed,
                "removed": total_removed,
                "removal_rate": f"{100*total_removed/max(1,total_original):.1f}%",
                "avg_length_before": round(avg_original, 2),
                "avg_length_after": round(avg_dedup, 2),
            },
        },
        "episodes": all_episodes,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Saved] {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate RL Dataset V3 (with consecutive dedup)")
    parser.add_argument(
        "--from-v2",
        type=str,
        default=None,
        help="Convert from existing V2 JSON file instead of regenerating from trajectories"
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default=None,
        help="Directory containing trajectory JSONL files"
    )
    parser.add_argument(
        "--mcp-graph",
        type=str,
        default=None,
        help="Path to mcp_rl_graph_v2.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for V3 JSON file"
    )

    args = parser.parse_args()

    PROJECT_ROOT = Path(" ")

    if args.from_v2:
        output_path = args.output or str(PROJECT_ROOT / "GRPO-ACO" / "data" / "rl_dataset_llm_v3.json")
        convert_from_v2(args.from_v2, output_path)
    else:
        trajectories_dir = args.trajectories_dir or str(PROJECT_ROOT / "Toolathlon-Trajectories")
        mcp_graph_path = args.mcp_graph or str(PROJECT_ROOT / "json_file" / "mcp_rl_graph_v2.json")
        output_path = args.output or str(PROJECT_ROOT / "GRPO-ACO" / "data" / "rl_dataset_llm_v3.json")

        generate_rl_dataset_v3(trajectories_dir, mcp_graph_path, output_path)


if __name__ == "__main__":
    main()
