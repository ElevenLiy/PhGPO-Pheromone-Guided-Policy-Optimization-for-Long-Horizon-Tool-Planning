#!/usr/bin/env python3

import json
import random
import hashlib
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] openai package not installed. Run: pip install openai")
    print("         LLM simulator will be disabled, falling back to pool-based simulator.")


@dataclass
class TaskCompletionResult:
    is_complete: bool
    confidence: float
    reason: str
    quality_score: float
    missing_steps: List[str]

    def to_dict(self) -> Dict:
        return {
            "is_complete": self.is_complete,
            "confidence": self.confidence,
            "reason": self.reason,
            "quality_score": self.quality_score,
            "missing_steps": self.missing_steps,
        }


@dataclass
class LLMSimulatorConfig:
    api_key: str = "sk-PovCrGTefqSW0POpIed5jFWF3HN6Cc95PXoWa70Zx1MKLNg4"
    base_url: str = "http://172.22.2.242:3010/v1"
    model: str = "qwen3-235b-a22b-thinking-2507"

    max_retries: int = 3
    timeout: float = 30.0
    max_tokens: int = 300
    temperature: float = 0.7

    enable_cache: bool = True
    cache_size: int = 5000

    use_llm: bool = True
    use_hybrid: bool = True
    fallback_to_pool: bool = True

    max_history_length: int = 5
    max_prompt_length: int = 2000

    enable_completion_check: bool = True

    verbose: bool = False
    log_api_calls: bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> 'LLMSimulatorConfig':
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    @classmethod
    def from_training_config(cls, cfg) -> 'LLMSimulatorConfig':
        llm_config = getattr(cfg, 'llm_simulator_config', {})
        use_llm = getattr(cfg, 'use_llm_simulator', False)
        use_hybrid = getattr(cfg, 'use_hybrid_simulator', True)
        enable_completion = getattr(cfg, 'enable_completion_check', True)

        return cls.from_dict({
            **llm_config,
            "use_llm": use_llm,
            "use_hybrid": use_hybrid,
            "enable_completion_check": enable_completion,
        })

    def to_dict(self) -> Dict:
        return {
            "api_key": self.api_key[:20] + "..." if len(self.api_key) > 20 else self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enable_cache": self.enable_cache,
            "cache_size": self.cache_size,
            "use_llm": self.use_llm,
            "use_hybrid": self.use_hybrid,
            "fallback_to_pool": self.fallback_to_pool,
            "max_history_length": self.max_history_length,
            "enable_completion_check": self.enable_completion_check,
            "verbose": self.verbose,
        }


class LLMToolSimulator:
    SYSTEM_PROMPT = """You are a professional tool execution simulator. Your task is to generate realistic, reasonable simulation outputs based on given tool call information.

## Output Rules:
1. Output the tool's return result directly, without any explanation, prefix or suffix
2. Output should look like real API/tool return values
3. Generate reasonable simulation data based on tool type, parameters and context
4. Keep output length between 50-200 characters
5. Maintain diversity and authenticity of outputs

## [IMPORTANT] Task Completion Detection:
After generating tool output, determine whether the current tool call sequence has completed the user's original task.

**If task is completed**: Add `<<END>>` on the last line alone
**If task is not completed**: Output tool result normally without any marker

## Output Guidelines for Different Tool Types:
- Search tools: Return search result summary or list
- Data retrieval tools: Return JSON format or structured data
- File operation tools: Return operation status and results
- API call tools: Return API response format
- Calculation tools: Return calculation results

## Notes:
- Reference historical call output style for consistency
- Generate relevant output content based on parameter values
- For error cases, return reasonable error messages"""

    END_MARKER = "<<END>>"

    ERROR_PATTERNS = [
        "error", "failed", "exception", "invalid",
        "not found", "permission denied", "timeout",
        "connection refused", "unauthorized", "forbidden",
        "bad request", "rate limit", "quota exceeded",
        "internal server error", "service unavailable"
    ]

    PARAM_ALIASES = {
        "query": ["q", "search", "keyword", "text", "input", "question"],
        "path": ["file", "filepath", "filename", "dir", "directory", "folder"],
        "url": ["link", "uri", "address", "endpoint", "href"],
        "content": ["text", "body", "data", "message", "payload"],
        "name": ["title", "label", "id", "identifier", "key"],
        "limit": ["max", "count", "num", "size", "top", "n"],
        "page": ["offset", "start", "skip", "from"],
        "format": ["type", "mode", "style", "output_format"],
    }

    def __init__(
            self,
            tool_system,
            config: LLMSimulatorConfig = None,
            **kwargs
    ):
        self.tool_system = tool_system

        if config is not None:
            self.config = config
        else:
            self.config = LLMSimulatorConfig()

        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        if not OPENAI_AVAILABLE:
            self._log("[LLMToolSimulator] OpenAI not available, will return default outputs")
            self.client = None
        else:
            try:
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except Exception as e:
                self._log(f"[LLMToolSimulator] Failed to initialize OpenAI client: {e}")
                self.client = None

        self._cache: Dict[str, str] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()

        self._completion_cache: Dict[str, bool] = {}
        self._completion_cache_lock = threading.Lock()

        self._stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "api_successes": 0,
            "api_errors": 0,
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "end_detected": 0,
        }
        self._stats_lock = threading.Lock()

        self._api_log: List[Dict] = []
        self._api_log_lock = threading.Lock()
        self._max_api_log_size = 100

        self._alias_to_canonical = {}
        for canonical, aliases in self.PARAM_ALIASES.items():
            self._alias_to_canonical[canonical] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical

        if self.config.verbose:
            self._log(f"[LLMToolSimulator] Initialized (V1.1 with END marker support):")
            self._log(f"    Model: {self.config.model}")
            self._log(f"    Base URL: {self.config.base_url}")
            self._log(f"    Cache enabled: {self.config.enable_cache}")
            self._log(f"    Cache size: {self.config.cache_size}")
            self._log(f"    Completion check: {self.config.enable_completion_check}")
            self._log(f"    Max retries: {self.config.max_retries}")

    def _log(self, message: str):
        if self.config.verbose:
            print(message, flush=True)

    def _compute_cache_key(
            self,
            variant_name: str,
            task_name: str,
            user_prompt: str,
            history: List[Dict],
            params: Dict
    ) -> str:
        max_history = self.config.max_history_length
        recent_history = history[-max_history:] if history else []

        history_str = json.dumps(
            [{"name": h.get("name", ""), "output": str(h.get("output", ""))[:100]}
             for h in recent_history],
            sort_keys=True,
            ensure_ascii=False
        )

        params_str = json.dumps(
            {k: str(v)[:50] for k, v in (params or {}).items()},
            sort_keys=True,
            ensure_ascii=False
        )

        content = f"{variant_name}|{task_name[:100]}|{user_prompt[:200]}|{history_str}|{params_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def _normalize_param_name(self, name: str) -> str:
        return self._alias_to_canonical.get(name.lower().strip(), name.lower().strip())

    def _get_tool_info(self, variant_name: str) -> Tuple[str, str, Dict]:
        original_name = self.tool_system.original_name_from_extended(variant_name)
        tool_info = self.tool_system.original_tools.get(original_name, {})

        description = tool_info.get("description", "Execute tool operation")
        parameters = tool_info.get("parameters", {})

        return original_name, description, parameters

    def _build_prompt(
            self,
            variant_name: str,
            params: Dict,
            task_name: str,
            user_prompt: str,
            history: List[Dict]
    ) -> str:
        original_name, tool_desc, tool_params = self._get_tool_info(variant_name)

        variant_info = self.tool_system.extended_tools.get(variant_name, {})
        actual_keys = variant_info.get("actual_keys", [])

        lines = []

        lines.append(f"[Task Name] {task_name[:150]}")
        lines.append(f"[User Request] {user_prompt[:400]}")
        lines.append("")

        if history:
            lines.append("[Historical Tool Calls]")
            max_history = self.config.max_history_length
            for h in history[-max_history:]:
                name = h.get('name', 'unknown')[:40]
                output = str(h.get('output', ''))[:150]
                output = output.replace(self.END_MARKER, "").strip()
                lines.append(f"  [{name}] → {output}")
            lines.append("")

        lines.append(f"[Current Tool Call] {variant_name}")
        lines.append(f"[Original Tool Name] {original_name}")
        lines.append(f"[Tool Description] {tool_desc[:300]}")

        if params:
            lines.append("[Call Parameters]")
            for key, value in list(params.items())[:15]:
                value_str = str(value)[:150]
                lines.append(f"  • {key}: {value_str}")
        elif actual_keys:
            lines.append(f"[Expected Parameters] {', '.join(actual_keys[:10])}")

        lines.append("")
        lines.append("Please simulate the execution output of this tool. If task is complete, add <<END>> at the end:")

        prompt = "\n".join(lines)

        if len(prompt) > self.config.max_prompt_length:
            prompt = prompt[:self.config.max_prompt_length] + "\n...[truncated]"

        return prompt

    def _call_llm(self, prompt: str) -> Optional[str]:
        if self.client is None:
            return None

        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

                response = completion.choices[0].message.content.strip()
                elapsed = time.time() - start_time

                with self._stats_lock:
                    self._stats["api_successes"] += 1
                    self._stats["total_response_time"] += elapsed

                    if hasattr(completion, 'usage') and completion.usage:
                        self._stats["total_tokens"] += completion.usage.total_tokens
                        self._stats["total_prompt_tokens"] += completion.usage.prompt_tokens
                        self._stats["total_completion_tokens"] += completion.usage.completion_tokens

                if self.config.log_api_calls:
                    self._log_api_call(prompt, response, elapsed, None)

                return response

            except Exception as e:
                last_error = e

                with self._stats_lock:
                    self._stats["api_errors"] += 1

                if self.config.verbose:
                    self._log(f"[LLMToolSimulator] API error (attempt {attempt + 1}/{self.config.max_retries}): {e}")

                if attempt < self.config.max_retries - 1:
                    wait_time = min(30, 2 ** attempt + random.random())
                    time.sleep(wait_time)

        if self.config.log_api_calls:
            self._log_api_call(prompt, None, time.time() - start_time, str(last_error))

        return None

    def _log_api_call(self, prompt: str, response: Optional[str], elapsed: float, error: Optional[str]):
        with self._api_log_lock:
            self._api_log.append({
                "timestamp": time.time(),
                "prompt_preview": prompt[:200],
                "response_preview": response[:200] if response else None,
                "elapsed": elapsed,
                "error": error,
            })

            if len(self._api_log) > self._max_api_log_size:
                self._api_log = self._api_log[-self._max_api_log_size:]

    def _post_process_response(self, response: str, variant_name: str) -> Tuple[str, bool]:
        if not response:
            return f"[{variant_name}] executed successfully", False

        is_complete = self.END_MARKER in response

        clean_output = response.replace(self.END_MARKER, "").strip()

        prefixes_to_remove = [
            "Tool output:", "Output:", "Result:", "Return:",
            "Simulated output:",
        ]

        for prefix in prefixes_to_remove:
            if clean_output.lower().startswith(prefix.lower()):
                clean_output = clean_output[len(prefix):].strip()

        if clean_output.startswith("```"):
            lines = clean_output.split("\n")
            if len(lines) > 2:
                clean_output = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        if len(clean_output) > 500:
            clean_output = clean_output[:500] + "..."

        if not clean_output.strip():
            clean_output = f"[{variant_name}] executed successfully"

        return clean_output.strip(), is_complete

    def get_output(
            self,
            variant_name: str,
            params: Dict = None,
            task_name: str = "",
            user_prompt: str = "",
            history: List[Dict] = None
    ) -> str:
        with self._stats_lock:
            self._stats["total_calls"] += 1

        params = params or {}
        history = history or []

        cache_key = None
        if self.config.enable_cache:
            cache_key = self._compute_cache_key(
                variant_name, task_name, user_prompt, history, params
            )

            with self._cache_lock:
                if cache_key in self._cache:
                    with self._stats_lock:
                        self._stats["cache_hits"] += 1
                    return self._cache[cache_key]

        prompt = self._build_prompt(variant_name, params, task_name, user_prompt, history)

        with self._stats_lock:
            self._stats["api_calls"] += 1

        response = self._call_llm(prompt)

        if response:
            clean_output, is_complete = self._post_process_response(response, variant_name)

            if is_complete:
                with self._stats_lock:
                    self._stats["end_detected"] += 1

            if self.config.enable_cache and cache_key:
                with self._cache_lock:
                    self._cache[cache_key] = clean_output
                    self._cache_order.append(cache_key)

                    while len(self._cache_order) > self.config.cache_size:
                        old_key = self._cache_order.pop(0)
                        self._cache.pop(old_key, None)

                with self._completion_cache_lock:
                    self._completion_cache[cache_key] = is_complete

            return clean_output

        default_output = self._generate_default_output(variant_name, params)
        return default_output

    def get_output_simple(self, variant_name: str, params: Dict = None) -> str:
        return self.get_output(variant_name, params, "", "", [])

    def _generate_default_output(self, variant_name: str, params: Dict) -> str:
        original_name = self.tool_system.original_name_from_extended(variant_name)

        name_lower = original_name.lower()

        if any(kw in name_lower for kw in ["search", "find", "query", "lookup"]):
            return f"Found 3 results for the query. Top result: Relevant information about the topic."
        elif any(kw in name_lower for kw in ["get", "fetch", "retrieve", "read"]):
            return f"Successfully retrieved data: {{'status': 'ok', 'items': 5}}"
        elif any(kw in name_lower for kw in ["create", "add", "insert", "write"]):
            return f"Successfully created new item with id: {random.randint(1000, 9999)}"
        elif any(kw in name_lower for kw in ["update", "modify", "edit", "patch"]):
            return f"Successfully updated item. Changes applied."
        elif any(kw in name_lower for kw in ["delete", "remove", "clear"]):
            return f"Successfully deleted item."
        elif any(kw in name_lower for kw in ["send", "post", "submit"]):
            return f"Message sent successfully. Response code: 200"
        elif any(kw in name_lower for kw in ["calculate", "compute", "analyze"]):
            return f"Calculation complete. Result: {random.uniform(0.1, 100.0):.2f}"
        else:
            param_str = ", ".join(list(params.keys())[:3]) if params else "default"
            return f"[{variant_name}] executed successfully with params: {param_str}"

    def check_task_completion(
            self,
            task_name: str,
            user_prompt: str,
            history: List[Dict],
            smart_mode: bool = True
    ) -> TaskCompletionResult:
        if not history:
            return TaskCompletionResult(
                is_complete=False,
                confidence=1.0,
                quality_score=0.0,
                reason="No tools executed",
                missing_steps=["Start execution"]
            )

        last_entry = history[-1]
        last_tool = last_entry.get("name", "")

        cache_key = self._compute_cache_key(
            last_tool, task_name, user_prompt, history[:-1], {}
        )

        with self._completion_cache_lock:
            if cache_key in self._completion_cache:
                is_complete = self._completion_cache[cache_key]
                if is_complete:
                    return TaskCompletionResult(
                        is_complete=True,
                        confidence=0.85,
                        quality_score=0.8,
                        reason="END marker detected in output",
                        missing_steps=[]
                    )

        if smart_mode:
            return self._heuristic_check(history)

        return TaskCompletionResult(
            is_complete=False,
            confidence=0.5,
            quality_score=0.5,
            reason="No END marker found in cache",
            missing_steps=[]
        )

    def _heuristic_check(self, history: List[Dict]) -> TaskCompletionResult:
        if not history:
            return TaskCompletionResult(
                is_complete=False,
                confidence=1.0,
                quality_score=0.0,
                reason="No history",
                missing_steps=[]
            )

        last_output = str(history[-1].get("output", "")).lower()
        last_tool = str(history[-1].get("name", "")).lower()

        completion_signals = [
            "successfully", "completed", "done", "finished",
            "saved", "sent", "created", "updated"
        ]
        error_signals = [
            "error", "failed", "not found", "timeout",
            "exception", "invalid"
        ]

        has_completion = any(s in last_output for s in completion_signals)
        has_error = any(s in last_output for s in error_signals)

        final_tools = [
            "save", "send", "submit", "create", "generate",
            "export", "post", "write", "publish"
        ]
        is_final = any(t in last_tool for t in final_tools)

        if has_error:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.8,
                quality_score=0.2,
                reason="Error detected in output",
                missing_steps=["Fix error"]
            )

        if has_completion and is_final:
            return TaskCompletionResult(
                is_complete=True,
                confidence=0.7,
                quality_score=0.7,
                reason="Completion signal with final tool",
                missing_steps=[]
            )

        if has_completion:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.5,
                quality_score=0.5,
                reason="Completion signal but may need more steps",
                missing_steps=[]
            )

        return TaskCompletionResult(
            is_complete=False,
            confidence=0.5,
            quality_score=0.4,
            reason="Heuristic: uncertain",
            missing_steps=[]
        )

    def get_last_completion_status(self, cache_key: str = None) -> Optional[bool]:
        if cache_key is None:
            return None

        with self._completion_cache_lock:
            return self._completion_cache.get(cache_key)

    def is_execution_error(self, output: str) -> bool:
        output_lower = output.lower()
        return any(pattern in output_lower for pattern in self.ERROR_PATTERNS)

    def get_statistics(self) -> Dict:
        with self._stats_lock:
            stats = self._stats.copy()

        stats["cache_size"] = len(self._cache)
        stats["completion_cache_size"] = len(self._completion_cache)
        stats["cache_hit_rate"] = (
                stats["cache_hits"] / max(1, stats["total_calls"])
        )
        stats["api_success_rate"] = (
                stats["api_successes"] / max(1, stats["api_calls"])
        )
        stats["api_error_rate"] = (
                stats["api_errors"] / max(1, stats["api_calls"])
        )
        stats["end_detection_rate"] = (
                stats["end_detected"] / max(1, stats["api_successes"])
        )

        if stats["api_successes"] > 0:
            stats["avg_response_time"] = (
                    stats["total_response_time"] / stats["api_successes"]
            )
            stats["avg_tokens_per_call"] = (
                    stats["total_tokens"] / stats["api_successes"]
            )
        else:
            stats["avg_response_time"] = 0.0
            stats["avg_tokens_per_call"] = 0.0

        return stats

    def get_api_log(self) -> List[Dict]:
        with self._api_log_lock:
            return self._api_log.copy()

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()
        with self._completion_cache_lock:
            self._completion_cache.clear()

    def reset_statistics(self):
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0 if isinstance(self._stats[key], int) else 0.0

        with self._api_log_lock:
            self._api_log.clear()

    def get_output_with_completion(
            self,
            variant_name: str,
            params: Dict = None,
            task_name: str = "",
            user_prompt: str = "",
            history: List[Dict] = None
    ) -> Tuple[str, bool]:
        with self._stats_lock:
            self._stats["total_calls"] += 1

        params = params or {}
        history = history or []

        cache_key = None
        if self.config.enable_cache:
            cache_key = self._compute_cache_key(
                variant_name, task_name, user_prompt, history, params
            )

            with self._cache_lock:
                if cache_key in self._cache:
                    with self._stats_lock:
                        self._stats["cache_hits"] += 1
                    output = self._cache[cache_key]
                    with self._completion_cache_lock:
                        is_complete = self._completion_cache.get(cache_key, False)
                    return output, is_complete

        prompt = self._build_prompt(variant_name, params, task_name, user_prompt, history)

        with self._stats_lock:
            self._stats["api_calls"] += 1

        response = self._call_llm(prompt)

        if response:
            clean_output, is_complete = self._post_process_response(response, variant_name)

            if is_complete:
                with self._stats_lock:
                    self._stats["end_detected"] += 1

            if self.config.enable_cache and cache_key:
                with self._cache_lock:
                    self._cache[cache_key] = clean_output
                    self._cache_order.append(cache_key)

                    while len(self._cache_order) > self.config.cache_size:
                        old_key = self._cache_order.pop(0)
                        self._cache.pop(old_key, None)

                with self._completion_cache_lock:
                    self._completion_cache[cache_key] = is_complete

            return clean_output, is_complete

        default_output = self._generate_default_output(variant_name, params)
        return default_output, False


class HybridToolSimulator:

    def __init__(
            self,
            tool_system,
            database_path: str = None,
            config: LLMSimulatorConfig = None,
            use_llm: bool = True,
            **kwargs
    ):
        self.tool_system = tool_system

        if config is not None:
            self.config = config
        else:
            self.config = LLMSimulatorConfig.from_dict({
                "use_llm": use_llm,
                **kwargs
            })

        self.llm_simulator: Optional[LLMToolSimulator] = None
        if self.config.use_llm and OPENAI_AVAILABLE:
            self.llm_simulator = LLMToolSimulator(tool_system, config=self.config)

        self.pool_outputs: Dict[str, List[str]] = {}
        self.database_path = database_path
        if database_path:
            self._load_pool(database_path)

        self._stats = {
            "total_calls": 0,
            "llm_calls": 0,
            "llm_successes": 0,
            "pool_fallbacks": 0,
            "pool_hits": 0,
            "default_outputs": 0,
        }
        self._stats_lock = threading.Lock()

        self.error_patterns = LLMToolSimulator.ERROR_PATTERNS

        self._log_init()

    def _log_init(self):
        print(f"[HybridToolSimulator] Initialized (V1.1 with END marker support):")
        print(f"    LLM enabled: {self.llm_simulator is not None}")
        print(f"    LLM model: {self.config.model if self.llm_simulator else 'N/A'}")
        print(f"    Pool tools: {len(self.pool_outputs)}")
        print(f"    Fallback enabled: {self.config.fallback_to_pool}")

    def _load_pool(self, database_path: str):
        path = Path(database_path)
        if not path.exists():
            print(f"[HybridToolSimulator] Pool database not found: {database_path}")
            return

        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for tool_name, tool_data in data.items():
                if isinstance(tool_data, dict):
                    outputs = tool_data.get("outputs", [])
                    if outputs:
                        self.pool_outputs[tool_name] = (
                            outputs if isinstance(outputs, list) else [outputs]
                        )
                elif isinstance(tool_data, list):
                    self.pool_outputs[tool_name] = tool_data
                elif isinstance(tool_data, str):
                    self.pool_outputs[tool_name] = [tool_data]

            print(f"[HybridToolSimulator] Loaded {len(self.pool_outputs)} tools from pool")

        except Exception as e:
            print(f"[HybridToolSimulator] Failed to load pool: {e}")

    def _get_pool_output(self, variant_name: str) -> Optional[str]:
        if variant_name in self.pool_outputs:
            outputs = self.pool_outputs[variant_name]
            if outputs:
                return random.choice(outputs)

        original_name = self.tool_system.original_name_from_extended(variant_name)
        if original_name in self.pool_outputs:
            outputs = self.pool_outputs[original_name]
            if outputs:
                return random.choice(outputs)

        for pool_name, outputs in self.pool_outputs.items():
            if pool_name in variant_name or variant_name in pool_name:
                if outputs:
                    return random.choice(outputs)

        return None

    def get_output(
            self,
            variant_name: str,
            params: Dict = None,
            task_name: str = "",
            user_prompt: str = "",
            history: List[Dict] = None
    ) -> str:
        with self._stats_lock:
            self._stats["total_calls"] += 1

        params = params or {}
        history = history or []

        if self.llm_simulator is not None:
            with self._stats_lock:
                self._stats["llm_calls"] += 1

            output = self.llm_simulator.get_output(
                variant_name, params, task_name, user_prompt, history
            )

            if output and not self._is_default_output(output, variant_name):
                with self._stats_lock:
                    self._stats["llm_successes"] += 1
                return output

        if self.config.fallback_to_pool:
            pool_output = self._get_pool_output(variant_name)
            if pool_output:
                with self._stats_lock:
                    self._stats["pool_fallbacks"] += 1
                    self._stats["pool_hits"] += 1
                return pool_output

        with self._stats_lock:
            self._stats["default_outputs"] += 1

        return self._generate_default_output(variant_name, params)

    def _is_default_output(self, output: str, variant_name: str) -> bool:
        default_patterns = [
            f"[{variant_name}] executed",
            "executed successfully with params",
            "executed successfully",
        ]
        return any(pattern in output for pattern in default_patterns)

    def _generate_default_output(self, variant_name: str, params: Dict) -> str:
        param_str = ", ".join(list((params or {}).keys())[:3]) if params else "default"
        return f"[{variant_name}] executed successfully with params: {param_str}"

    def check_task_completion(
            self,
            task_name: str,
            user_prompt: str,
            history: List[Dict],
            smart_mode: bool = True
    ) -> TaskCompletionResult:
        if self.llm_simulator is not None:
            return self.llm_simulator.check_task_completion(
                task_name, user_prompt, history, smart_mode
            )

        if not history:
            return TaskCompletionResult(
                is_complete=False,
                confidence=0.0,
                quality_score=0.0,
                reason="No LLM simulator available",
                missing_steps=[]
            )

        last_output = str(history[-1].get("output", "")).lower()
        has_success = any(s in last_output for s in ["success", "completed", "done"])

        return TaskCompletionResult(
            is_complete=has_success,
            confidence=0.5 if has_success else 0.3,
            quality_score=0.5 if has_success else 0.3,
            reason="Pool-based heuristic",
            missing_steps=[]
        )

    def is_execution_error(self, output: str) -> bool:
        if self.llm_simulator:
            return self.llm_simulator.is_execution_error(output)

        output_lower = output.lower()
        return any(p in output_lower for p in self.error_patterns)

    def get_statistics(self) -> Dict:
        with self._stats_lock:
            stats = self._stats.copy()

        stats["llm_success_rate"] = (
                stats["llm_successes"] / max(1, stats["llm_calls"])
        )
        stats["fallback_rate"] = (
                stats["pool_fallbacks"] / max(1, stats["total_calls"])
        )

        if self.llm_simulator:
            stats["llm_stats"] = self.llm_simulator.get_statistics()

        stats["pool_size"] = len(self.pool_outputs)

        return stats

    def clear_cache(self):
        if self.llm_simulator:
            self.llm_simulator.clear_cache()

    def get_output_with_completion(
            self,
            variant_name: str,
            params: Dict = None,
            task_name: str = "",
            user_prompt: str = "",
            history: List[Dict] = None
    ) -> Tuple[str, bool]:
        with self._stats_lock:
            self._stats["total_calls"] += 1

        params = params or {}
        history = history or []

        if self.llm_simulator is not None:
            with self._stats_lock:
                self._stats["llm_calls"] += 1

            output, is_complete = self.llm_simulator.get_output_with_completion(
                variant_name, params, task_name, user_prompt, history
            )

            if output and not self._is_default_output(output, variant_name):
                with self._stats_lock:
                    self._stats["llm_successes"] += 1
                return output, is_complete

        if self.config.fallback_to_pool:
            pool_output = self._get_pool_output(variant_name)
            if pool_output:
                with self._stats_lock:
                    self._stats["pool_fallbacks"] += 1
                    self._stats["pool_hits"] += 1
                return pool_output, False

        with self._stats_lock:
            self._stats["default_outputs"] += 1

        return self._generate_default_output(variant_name, params), False


class PoolToolSimulator:

    def __init__(self, database_path: str, tool_system):
        self.tool_system = tool_system
        self.outputs: Dict[str, List[str]] = {}

        self.error_patterns = [
            "error", "failed", "exception", "invalid",
            "not found", "permission denied", "timeout",
            "connection refused", "unauthorized", "forbidden",
            "bad request"
        ]

        if Path(database_path).exists():
            self._load_database(database_path)

        print(f"[PoolToolSimulator] Loaded outputs for {len(self.outputs)} tools")

    def _load_database(self, database_path: str):
        try:
            with open(database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for tool_name, tool_data in data.items():
                if isinstance(tool_data, dict):
                    outputs = tool_data.get("outputs", [])
                    if outputs:
                        self.outputs[tool_name] = (
                            outputs if isinstance(outputs, list) else [outputs]
                        )
                elif isinstance(tool_data, list):
                    self.outputs[tool_name] = tool_data
                elif isinstance(tool_data, str):
                    self.outputs[tool_name] = [tool_data]

        except Exception as e:
            print(f"[PoolToolSimulator] Failed to load database: {e}")

    def get_output(self, variant_name: str, params: Dict = None, **kwargs) -> str:
        if variant_name in self.outputs:
            outputs = self.outputs[variant_name]
            if outputs:
                return random.choice(outputs)

        original_name = self.tool_system.original_name_from_extended(variant_name)
        if original_name in self.outputs:
            outputs = self.outputs[original_name]
            if outputs:
                return random.choice(outputs)

        return f"[{variant_name}] executed successfully"

    def check_task_completion(self, *args, **kwargs) -> TaskCompletionResult:
        return TaskCompletionResult(
            is_complete=False,
            confidence=0.0,
            quality_score=0.0,
            reason="Pool simulator does not support completion check",
            missing_steps=[]
        )

    def is_execution_error(self, output: str) -> bool:
        output_lower = output.lower()
        return any(p in output_lower for p in self.error_patterns)

    def get_statistics(self) -> Dict:
        return {
            "type": "pool",
            "num_tools": len(self.outputs),
            "total_outputs": sum(len(v) for v in self.outputs.values()),
        }


def create_simulator(
        tool_system,
        config: LLMSimulatorConfig = None,
        database_path: str = None,
        use_llm: bool = True,
        use_hybrid: bool = True,
        **kwargs
) -> Any:
    if config is None:
        config = LLMSimulatorConfig.from_dict({
            "use_llm": use_llm,
            "use_hybrid": use_hybrid,
            **kwargs
        })

    if config.use_llm and not OPENAI_AVAILABLE:
        print("[create_simulator] OpenAI not available, falling back to pool simulator")
        config.use_llm = False

    if not config.use_llm:
        if database_path and Path(database_path).exists():
            print("[create_simulator] Creating PoolToolSimulator")
            return PoolToolSimulator(database_path, tool_system)
        else:
            print("[create_simulator] Warning: No database, creating LLM simulator with defaults")
            return LLMToolSimulator(tool_system, config=config)

    if config.use_hybrid and database_path:
        print("[create_simulator] Creating HybridToolSimulator (LLM + pool fallback)")
        return HybridToolSimulator(
            tool_system,
            database_path=database_path,
            config=config
        )

    print("[create_simulator] Creating LLMToolSimulator (pure LLM)")
    return LLMToolSimulator(tool_system, config=config)


def create_simulator_from_config(tool_system, cfg) -> Any:
    use_llm = getattr(cfg, 'use_llm_simulator', False)
    use_hybrid = getattr(cfg, 'use_hybrid_simulator', True)
    llm_config_dict = getattr(cfg, 'llm_simulator_config', {})
    database_path = getattr(cfg, 'simulator_database_path', None)
    enable_completion = getattr(cfg, 'enable_completion_check', True)

    if database_path is not None:
        database_path = str(database_path)

    if use_llm:
        config = LLMSimulatorConfig.from_dict({
            **llm_config_dict,
            "use_llm": True,
            "use_hybrid": use_hybrid,
            "fallback_to_pool": use_hybrid,
            "enable_completion_check": enable_completion,
        })

        return create_simulator(
            tool_system,
            config=config,
            database_path=database_path if use_hybrid else None
        )
    else:
        if database_path and Path(database_path).exists():
            return PoolToolSimulator(database_path, tool_system)
        else:
            print("[create_simulator_from_config] Warning: use_llm=False but no valid database")
            return PoolToolSimulator(database_path or "", tool_system)


ToolSimulator = PoolToolSimulator


def test_llm_connection(config: LLMSimulatorConfig = None) -> bool:
    print("\n" + "=" * 50)
    print("Testing LLM API Connection")
    print("=" * 50)

    if not OPENAI_AVAILABLE:
        print("✗ OpenAI package not installed")
        print("  Run: pip install openai")
        return False

    config = config or LLMSimulatorConfig()

    print(f"  API Base: {config.base_url}")
    print(f"  Model: {config.model}")

    try:
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=15.0
        )

        print("  Sending test request...")

        start_time = time.time()
        completion = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": "Say 'API connection successful' in exactly those words."}],
            max_tokens=20
        )
        elapsed = time.time() - start_time

        response = completion.choices[0].message.content
        print(f"\n✓ Connection successful!")
        print(f"  Response: {response}")
        print(f"  Latency: {elapsed:.2f}s")

        if hasattr(completion, 'usage') and completion.usage:
            print(f"  Tokens: {completion.usage.total_tokens}")

        return True

    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        return False


def test_simulator(tool_system=None) -> bool:
    print("\n" + "=" * 50)
    print("Testing LLM Tool Simulator (V1.1 with END marker)")
    print("=" * 50)

    if tool_system is None:
        print("  No tool_system provided, creating mock...")

        class MockToolSystem:
            def __init__(self):
                self.original_tools = {
                    "web_search": {
                        "description": "Search the web for information",
                        "parameters": {"query": {"type": "string"}}
                    }
                }
                self.extended_tools = {
                    "web_search_query": {
                        "original_name": "web_search",
                        "actual_keys": ["query"]
                    }
                }

            def original_name_from_extended(self, name):
                return self.extended_tools.get(name, {}).get("original_name", name)

        tool_system = MockToolSystem()

    config = LLMSimulatorConfig(verbose=True)

    try:
        simulator = LLMToolSimulator(tool_system, config=config)

        print("\n  Testing get_output()...")
        output = simulator.get_output(
            variant_name="web_search_query",
            params={"query": "latest AI news"},
            task_name="Search for information",
            user_prompt="Find the latest developments in artificial intelligence",
            history=[{"name": "init", "output": "System initialized"}]
        )

        print(f"\n  Output: {output}")

        print("\n  Testing check_task_completion()...")
        history = [
            {"name": "web_search_query", "output": output}
        ]
        completion = simulator.check_task_completion(
            "Search for information",
            "Find the latest developments in artificial intelligence",
            history
        )
        print(f"  Completion: is_complete={completion.is_complete}, confidence={completion.confidence:.2f}")
        print(f"  Reason: {completion.reason}")

        print(f"\n  Statistics: {json.dumps(simulator.get_statistics(), indent=2)}")

        print("\n✓ Simulator test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("LLM Tool Simulator V1.1 - Test Suite")
    print("(With END marker task completion detection)")
    print("=" * 60)

    api_ok = test_llm_connection()

    if api_ok:
        sim_ok = test_simulator()
    else:
        sim_ok = False
        print("\nSkipping simulator test due to API connection failure")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  API Connection: {'✓ PASS' if api_ok else '✗ FAIL'}")
    print(f"  Simulator:      {'✓ PASS' if sim_ok else '✗ FAIL'}")
    print("=" * 60)

    sys.exit(0 if (api_ok and sim_ok) else 1)