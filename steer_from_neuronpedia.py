from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests
import torch
from default_arg_values import (
    ALL_INTERVENTION_SCOPES,
    STEER_DEFAULT_DEVICE,
    STEER_DEFAULT_INTERVENTION_SCOPE,
    STEER_DEFAULT_INTERVENTION_STEPS,
    STEER_DEFAULT_LLM_NAME,
    STEER_DEFAULT_LOGIT_ANALYSIS_MAX_STEPS,
    STEER_DEFAULT_LOGIT_ANALYSIS_REFERENCE,
    STEER_DEFAULT_LOGIT_ANALYSIS_TOP_K,
    STEER_DEFAULT_MAX_NEW_TOKENS,
    STEER_DEFAULT_MAX_PREFIX_TOKENS,
    STEER_DEFAULT_MODEL_ID,
    STEER_DEFAULT_OUTPUT_FILENAME,
    STEER_DEFAULT_OUTPUT_ROOT,
    STEER_DEFAULT_SAE_RELEASE,
    STEER_DEFAULT_STRENGTH_SCALES,
    STEER_DEFAULT_TEMPERATURE,
    STEER_DEFAULT_TIMEOUT,
    STEER_DEFAULT_TOP_K_EXAMPLES,
    STEER_DEFAULT_WIDTH,
)
from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path

SENTENCE_END_RE = re.compile(r"[.!?。！？]")
EPS = 1e-8
NSM_ACTIVATION_THRESHOLD = EPS
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch top-activating Neuronpedia examples for one (layer_id, feature_id), "
            "build truncated prompts around the max token, then run clean + steered decoding."
        )
    )
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--model-id", type=str, default=STEER_DEFAULT_MODEL_ID, help="Neuronpedia model id.")
    parser.add_argument("--llm-name", type=str, default=STEER_DEFAULT_LLM_NAME, help="HF model name/path.")
    parser.add_argument("--width", type=str, default=STEER_DEFAULT_WIDTH)
    parser.add_argument("--sae-release", type=str, default=STEER_DEFAULT_SAE_RELEASE)
    parser.add_argument(
        "--sae-path",
        type=str,
        default=None,
        help="Optional local SAE checkpoint path (directory or file). If provided, this takes priority over --sae-release/--width canonical mapping.",
    )
    parser.add_argument("--canonical-map-path", type=str, default=str(DEFAULT_CANONICAL_MAP_PATH))
    parser.add_argument("--device", type=str, default=STEER_DEFAULT_DEVICE)
    parser.add_argument("--temperature", type=float, default=STEER_DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=STEER_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--intervention-scope",
        type=str,
        choices=ALL_INTERVENTION_SCOPES,
        default=STEER_DEFAULT_INTERVENTION_SCOPE,
    )
    parser.add_argument(
        "--intervention-steps",
        type=int,
        default=STEER_DEFAULT_INTERVENTION_STEPS,
        help="Apply steering only for the first N predicted tokens, then continue with clean decoding.",
    )
    parser.add_argument("--top-k-examples", type=int, default=STEER_DEFAULT_TOP_K_EXAMPLES)
    parser.add_argument("--max-prefix-tokens", type=int, default=STEER_DEFAULT_MAX_PREFIX_TOKENS)
    parser.add_argument(
        "--strength-scales",
        type=str,
        nargs="*",
        default=None,
        help=(
            "List of steering scales. Supports space-separated values and/or comma-separated values, "
            "including fractions like 2/3. Example: --strength-scales 0 2/3 1.5"
        ),
    )
    parser.add_argument("--neuronpedia-api-key", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=STEER_DEFAULT_TIMEOUT)
    parser.add_argument("--output-root", type=str, default=STEER_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-filename", type=str, default=STEER_DEFAULT_OUTPUT_FILENAME)
    parser.add_argument(
        "--disable-logit-analysis",
        action="store_true",
        help="Disable logits-delta analysis between clean and steered runs.",
    )
    parser.add_argument(
        "--logit-analysis-max-steps",
        type=int,
        default=STEER_DEFAULT_LOGIT_ANALYSIS_MAX_STEPS,
        help="Maximum generation steps to analyze for logits shifts.",
    )
    parser.add_argument(
        "--logit-analysis-top-k",
        type=int,
        default=STEER_DEFAULT_LOGIT_ANALYSIS_TOP_K,
        help="Top-k tokens to report for positive/negative logits deltas per step.",
    )
    parser.add_argument(
        "--logit-analysis-reference",
        type=str,
        choices=("clean", "steered"),
        default=STEER_DEFAULT_LOGIT_ANALYSIS_REFERENCE,
        help="Reference trajectory used for teacher-forced logits analysis.",
    )
    parser.add_argument(
        "--logit-analysis-include-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens in top-k logits-delta lists.",
    )
    return parser.parse_args()


def _fetch_feature_json(
    *,
    model_id: str,
    source: str,
    feature_id: str,
    api_key: Optional[str],
    timeout: int,
    retry_count: int = 3,
    retry_sleep_seconds: float = 3.0,
) -> Dict[str, Any]:
    url = f"{NEURONPEDIA_BASE_URL}/api/feature/{model_id}/{source}/{feature_id}"
    token = api_key or os.getenv("NEURONPEDIA_API_KEY")
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    total_attempts = max(1, int(retry_count))
    last_error: Optional[Exception] = None
    for attempt_idx in range(total_attempts):
        try:
            response = requests.get(url, headers=headers, timeout=int(timeout))
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            if attempt_idx == total_attempts - 1:
                break
            time.sleep(float(retry_sleep_seconds))

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"Failed to fetch Neuronpedia feature payload: model={model_id}, source={source}, feature={feature_id}"
    )


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_single_scale(text: str) -> float:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty scale value.")
    if "/" in raw:
        parts = raw.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid fractional scale: {raw!r}")
        num = float(parts[0].strip())
        den = float(parts[1].strip())
        if abs(den) <= EPS:
            raise ValueError(f"Scale denominator cannot be zero: {raw!r}")
        return float(num / den)
    return float(raw)


def _resolve_strength_scales(raw_scales: Optional[Sequence[str]]) -> List[float]:
    if not raw_scales:
        return [float(x) for x in STEER_DEFAULT_STRENGTH_SCALES]

    parsed: List[float] = []
    for item in raw_scales:
        text = str(item).strip()
        if not text:
            continue
        for piece in text.split(","):
            token = piece.strip()
            if not token:
                continue
            parsed.append(_parse_single_scale(token))

    if not parsed:
        raise ValueError("No valid strength scales provided.")
    return parsed


def _safe_max_token(activation: Dict[str, Any]) -> str:
    tokens = activation.get("tokens")
    max_idx = activation.get("maxValueTokenIndex")
    if not isinstance(tokens, list) or not isinstance(max_idx, int):
        return ""
    if max_idx < 0 or max_idx >= len(tokens):
        return ""
    token = tokens[max_idx]
    return token if isinstance(token, str) else str(token)


def _select_activations_method_2(
    activations: List[Dict[str, Any]],
    n: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    selected_indices: List[int] = []
    for idx, item in enumerate(activations):
        if not selected_indices:
            selected_indices.append(idx)
        else:
            current_token = _safe_max_token(item)
            last_token = _safe_max_token(activations[selected_indices[-1]])
            if current_token != last_token:
                selected_indices.append(idx)
        if len(selected_indices) >= n:
            break
    return [activations[i] for i in selected_indices], selected_indices


def _find_first_max_index(
    values: Sequence[Any],
    declared_max: float,
    fallback_index: Optional[int],
) -> int:
    parsed_values: List[float] = [_to_float(v, default=0.0) for v in values]
    if parsed_values:
        if abs(declared_max) <= EPS and all(abs(v) <= EPS for v in parsed_values):
            return 0
        for idx, value in enumerate(parsed_values):
            if abs(value - declared_max) <= EPS:
                return idx
        return int(max(range(len(parsed_values)), key=lambda i: parsed_values[i]))

    if isinstance(fallback_index, int) and fallback_index >= 0:
        return fallback_index
    return 0


def _token_ends_sentence(token_text: str) -> bool:
    if not token_text:
        return False
    if "\n" in token_text:
        return True
    return bool(SENTENCE_END_RE.search(token_text))


def _find_sentence_start_token_idx(tokens: Sequence[str], max_idx: int) -> int:
    if max_idx <= 0:
        return 0
    for idx in range(max_idx - 1, -1, -1):
        if _token_ends_sentence(str(tokens[idx])):
            return idx + 1
    return 0


def _find_positive_activation_positions(values: Sequence[Any], threshold: float = EPS) -> List[int]:
    positions: List[int] = []
    for idx, raw_value in enumerate(values):
        if _to_float(raw_value, default=0.0) > float(threshold):
            positions.append(int(idx))
    return positions


def _build_legacy_truncation_info(
    *,
    tokens: Sequence[str],
    declared_max: float,
    values: Sequence[Any],
    fallback_idx: Optional[int],
    max_prefix_tokens: int,
) -> Dict[str, Any]:
    max_idx = _find_first_max_index(values, declared_max, fallback_idx)
    max_idx = max(0, min(max_idx, len(tokens) - 1))

    sentence_start = _find_sentence_start_token_idx(tokens, max_idx)
    max_prefix_tokens = max(0, int(max_prefix_tokens))
    window_start = max(sentence_start, max_idx - max_prefix_tokens)
    window_end = max_idx

    truncated_tokens = list(tokens[window_start : window_end + 1])
    prompt_text = "".join(truncated_tokens).strip()
    if not prompt_text:
        prompt_text = "".join(tokens[: max_idx + 1]).strip()

    return {
        "prompt_text": prompt_text,
        "max_token": str(tokens[max_idx]),
        "max_value": declared_max,
        "max_token_index": max_idx,
        "sentence_start_index": sentence_start,
        "window_start_index": window_start,
        "window_end_index": window_end,
        "window_token_count": len(truncated_tokens),
        "source_sentence": "".join(tokens),
        "source_token_count": len(tokens),
        "truncation_strategy": "legacy_first_max_token",
        "neuronpedia_support_positions": [],
        "support_start_index": max_idx,
        "support_end_index": max_idx,
        "support_start_token": str(tokens[max_idx]),
        "support_end_token": str(tokens[max_idx]),
        "support_token_count": 0,
        "covered_support_positions": [],
        "covered_support_count": 0,
        "support_window_start_index": window_start,
        "support_window_end_index": window_end,
        "support_window_token_count": len(truncated_tokens),
        "support_span_fallback_used": True,
    }


def _truncate_prompt_from_activation(
    activation: Dict[str, Any],
    *,
    max_prefix_tokens: int,
    scope: str,
) -> Dict[str, Any]:
    tokens_raw = activation.get("tokens", [])
    values_raw = activation.get("values", [])
    tokens: List[str] = [str(t) for t in tokens_raw] if isinstance(tokens_raw, list) else []
    values: List[Any] = list(values_raw) if isinstance(values_raw, list) else []
    declared_max = _to_float(activation.get("maxValue"), default=0.0)
    fallback_idx = activation.get("maxValueTokenIndex")
    if not isinstance(fallback_idx, int):
        fallback_idx = None

    if not tokens:
        return {
            "prompt_text": "",
            "max_token": "",
            "max_value": declared_max,
            "max_token_index": 0,
            "sentence_start_index": 0,
            "window_start_index": 0,
            "window_end_index": 0,
            "window_token_count": 0,
            "source_sentence": "",
            "source_token_count": 0,
            "truncation_strategy": "empty_tokens",
            "neuronpedia_support_positions": [],
            "support_start_index": 0,
            "support_end_index": 0,
            "support_start_token": "",
            "support_end_token": "",
            "support_token_count": 0,
            "covered_support_positions": [],
            "covered_support_count": 0,
            "support_window_start_index": 0,
            "support_window_end_index": 0,
            "support_window_token_count": 0,
            "support_span_fallback_used": False,
        }

    if str(scope) != "natural_support_mask":
        return _build_legacy_truncation_info(
            tokens=tokens,
            declared_max=declared_max,
            values=values,
            fallback_idx=fallback_idx,
            max_prefix_tokens=max_prefix_tokens,
        )

    support_positions = _find_positive_activation_positions(values, threshold=EPS)
    if not support_positions:
        legacy = _build_legacy_truncation_info(
            tokens=tokens,
            declared_max=declared_max,
            values=values,
            fallback_idx=fallback_idx,
            max_prefix_tokens=max_prefix_tokens,
        )
        legacy["truncation_strategy"] = "natural_support_mask_fallback_legacy_first_max_token"
        legacy["support_span_fallback_used"] = True
        return legacy

    support_start = max(0, min(int(support_positions[0]), len(tokens) - 1))
    support_end = max(0, min(int(support_positions[-1]), len(tokens) - 1))
    sentence_start = _find_sentence_start_token_idx(tokens, support_start)
    max_prefix_tokens = max(0, int(max_prefix_tokens))
    window_start = max(sentence_start, support_start - max_prefix_tokens)
    window_end = support_end

    truncated_tokens = list(tokens[window_start : window_end + 1])
    prompt_text = "".join(truncated_tokens).strip()
    if not prompt_text:
        prompt_text = "".join(tokens[window_start : support_end + 1]).strip()

    max_idx = _find_first_max_index(values, declared_max, fallback_idx)
    max_idx = max(0, min(max_idx, len(tokens) - 1))
    return {
        "prompt_text": prompt_text,
        "max_token": str(tokens[max_idx]),
        "max_value": declared_max,
        "max_token_index": max_idx,
        "sentence_start_index": sentence_start,
        "window_start_index": window_start,
        "window_end_index": window_end,
        "window_token_count": len(truncated_tokens),
        "source_sentence": "".join(tokens),
        "source_token_count": len(tokens),
        "truncation_strategy": "natural_support_span",
        "neuronpedia_support_positions": support_positions,
        "support_start_index": support_start,
        "support_end_index": support_end,
        "support_start_token": str(tokens[support_start]),
        "support_end_token": str(tokens[support_end]),
        "support_token_count": len(support_positions),
        "covered_support_positions": support_positions,
        "covered_support_count": len(support_positions),
        "support_window_start_index": window_start,
        "support_window_end_index": window_end,
        "support_window_token_count": len(truncated_tokens),
        "support_span_fallback_used": False,
    }


def _prepare_prompt_tensors(module: Any, prompt_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = module.tokenizer(str(prompt_text), return_tensors="pt")
    input_ids = encoded["input_ids"].to(module.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=module.device)
    else:
        attention_mask = attention_mask.to(module.device)
    return input_ids, attention_mask


def _sample_next_token(last_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(last_logits, dim=-1, keepdim=True)
    probs = torch.softmax(last_logits / float(temperature), dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def _run_base_logits(
    module: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if module.model is None:
        raise RuntimeError("Model not loaded.")
    input_ids = input_ids.to(module.device)
    attention_mask = attention_mask.to(module.device)

    if module.use_hooked_transformer:
        try:
            logits = module.model(input_ids, return_type="logits")
            if isinstance(logits, torch.Tensor):
                return logits
        except Exception:
            pass

        try:
            logits = module.model.run_with_saes(input_ids, saes=[])
            if isinstance(logits, torch.Tensor):
                return logits
        except Exception:
            pass

    outputs = module.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    if not hasattr(outputs, "logits") or outputs.logits is None:
        raise RuntimeError("Model output does not contain logits.")
    return outputs.logits


def _build_prompt_position_value_tensor(
    *,
    seq_len: int,
    prompt_positions: Sequence[int],
    steer_value: float,
    device: torch.device,
) -> torch.Tensor:
    value = torch.zeros((1, int(seq_len)), dtype=torch.float32, device=device)
    if int(seq_len) <= 0:
        return value
    for pos in prompt_positions:
        idx = int(pos)
        if 0 <= idx < int(seq_len):
            value[:, idx] = float(steer_value)
    return value


def _resolve_prompt_positions_for_scope(
    *,
    scope: str,
    prompt_len: int,
    nsm_positions: Optional[Sequence[int]] = None,
) -> List[int]:
    effective_prompt_len = max(0, int(prompt_len))
    if effective_prompt_len <= 0:
        return []
    if scope == "all_original_tokens":
        return list(range(effective_prompt_len))
    if scope == "last_original_token_only":
        return [effective_prompt_len - 1]
    if scope == "natural_support_mask":
        if not nsm_positions:
            return []
        return sorted(
            {
                int(pos)
                for pos in nsm_positions
                if 0 <= int(pos) < effective_prompt_len
            }
        )
    raise ValueError(f"Unsupported prompt-only scope: {scope}")


def _build_natural_support_metadata(
    *,
    trace: Dict[str, Any],
    prompt_len: int,
    threshold: float = NSM_ACTIVATION_THRESHOLD,
) -> Dict[str, Any]:
    per_token_activation = trace.get("per_token_activation", [])
    tokens = trace.get("tokens", [])
    selected_positions: List[int] = []
    selected_activations: List[float] = []
    selected_tokens: List[str] = []

    if isinstance(per_token_activation, list):
        limit = min(len(per_token_activation), max(0, int(prompt_len)))
        for idx in range(limit):
            activation = _to_float(per_token_activation[idx], default=0.0)
            if activation <= float(threshold):
                continue
            selected_positions.append(int(idx))
            selected_activations.append(float(activation))
            if isinstance(tokens, list) and idx < len(tokens):
                selected_tokens.append(str(tokens[idx]))
            else:
                selected_tokens.append("")

    return {
        "activation_threshold": float(threshold),
        "selected_positions": selected_positions,
        "selected_count": len(selected_positions),
        "selected_activations": selected_activations,
        "selected_tokens": selected_tokens,
        "trace_token_count": len(tokens) if isinstance(tokens, list) else 0,
        "trace_activation_count": len(per_token_activation) if isinstance(per_token_activation, list) else 0,
        "prompt_token_count": max(0, int(prompt_len)),
    }


@torch.no_grad()
def _run_steered_logits(
    module: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    feature_id: int,
    steer_value: float,
    scope: str,
    prompt_len: int,
    prompt_positions: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    if scope in {"all_tokens", "last_token_only"}:
        return module.run_logits_with_feature_intervention(
            input_ids=input_ids,
            feature_index=int(feature_id),
            value=float(steer_value),
            mode="add",
            attention_mask=attention_mask,
            intervention_scope=str(scope),
        )

    resolved_prompt_positions = _resolve_prompt_positions_for_scope(
        scope=str(scope),
        prompt_len=int(prompt_len),
        nsm_positions=prompt_positions,
    )
    value_tensor = _build_prompt_position_value_tensor(
        seq_len=int(input_ids.shape[1]),
        prompt_positions=resolved_prompt_positions,
        steer_value=float(steer_value),
        device=input_ids.device,
    )
    return module.run_logits_with_feature_intervention(
        input_ids=input_ids,
        feature_index=int(feature_id),
        value=value_tensor,
        mode="add",
        attention_mask=attention_mask,
        intervention_scope="all_tokens",
    )


@torch.no_grad()
def _generate_text(
    module: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    feature_id: Optional[int] = None,
    steer_value: Optional[float] = None,
    intervention_scope: str = "last_token_only",
    intervention_steps: int = 1,
    prompt_positions: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    tokens = input_ids.clone()
    mask = attention_mask.clone()
    prompt_len = int(tokens.shape[1])
    eos_id = getattr(module.tokenizer, "eos_token_id", None)
    steer_steps = max(0, int(intervention_steps))

    for step_idx in range(int(max_new_tokens)):
        should_steer = (
            feature_id is not None
            and steer_value is not None
            and step_idx < steer_steps
        )
        if should_steer:
            logits = _run_steered_logits(
                module,
                input_ids=tokens,
                attention_mask=mask,
                feature_id=int(feature_id),
                steer_value=float(steer_value),
                scope=str(intervention_scope),
                prompt_len=prompt_len,
                prompt_positions=prompt_positions,
            )
        else:
            logits = _run_base_logits(module, input_ids=tokens, attention_mask=mask)

        next_logits = logits[:, -1, :]
        next_token = _sample_next_token(next_logits, temperature=float(temperature))
        tokens = torch.cat([tokens, next_token], dim=1)
        mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)

        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    completion_ids = tokens[0, prompt_len:]
    completion_text = module.tokenizer.decode(completion_ids, skip_special_tokens=True)
    full_text = module.tokenizer.decode(tokens[0], skip_special_tokens=True)
    completion_id_list = [int(x) for x in completion_ids.detach().cpu().tolist()]
    try:
        completion_tokens = [str(tok) for tok in module.tokenizer.convert_ids_to_tokens(completion_id_list)]
    except Exception:
        completion_tokens = [str(x) for x in completion_id_list]
    return {
        "completion_text": completion_text,
        "full_text": full_text,
        "generated_token_count": int(completion_ids.numel()),
        "completion_token_ids": completion_id_list,
        "completion_tokens": completion_tokens,
    }


def _safe_decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens([int(token_id)])
        if isinstance(token, list) and token:
            return str(token[0])
    except Exception:
        pass
    return str(int(token_id))


def _build_special_token_id_set(tokenizer: Any, vocab_size: int) -> Set[int]:
    special_ids: Set[int] = set()
    if tokenizer is None or not hasattr(tokenizer, "all_special_ids"):
        return special_ids
    for token_id in getattr(tokenizer, "all_special_ids", []):
        try:
            idx = int(token_id)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < int(vocab_size):
            special_ids.add(idx)
    return special_ids


def _safe_token_rank_desc(logits: torch.Tensor, token_id: int) -> int:
    if logits.ndim != 1:
        raise ValueError("logits must be 1-D for ranking.")
    vocab_size = int(logits.shape[0])
    if token_id < 0 or token_id >= vocab_size:
        return vocab_size + 1
    token_score = logits[token_id]
    higher = int((logits > token_score).sum().item())
    return higher + 1


def _collect_top_delta_records(
    *,
    module: Any,
    delta_logits: torch.Tensor,
    clean_logits: torch.Tensor,
    steered_logits: torch.Tensor,
    clean_probs: torch.Tensor,
    steered_probs: torch.Tensor,
    top_k: int,
    special_token_ids: Set[int],
    include_special_tokens: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if delta_logits.ndim != 1:
        raise ValueError("delta_logits must be 1-D.")
    vocab_size = int(delta_logits.shape[0])
    k = max(0, min(int(top_k), vocab_size))
    if k <= 0:
        return [], []

    positive_scores = delta_logits.clone()
    negative_scores = delta_logits.clone()
    if not include_special_tokens and special_token_ids:
        special_indices = torch.tensor(sorted(special_token_ids), dtype=torch.long, device=delta_logits.device)
        positive_scores.index_fill_(0, special_indices, float("-inf"))
        negative_scores.index_fill_(0, special_indices, float("inf"))

    pos_vals, pos_ids = torch.topk(positive_scores, k=k, largest=True)
    neg_vals, neg_ids = torch.topk(negative_scores, k=k, largest=False)

    def _build_records(token_ids: torch.Tensor, delta_vals: torch.Tensor) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for idx, delta_val in zip(token_ids.tolist(), delta_vals.tolist()):
            if not math.isfinite(float(delta_val)):
                continue
            token_id = int(idx)
            records.append(
                {
                    "token_id": token_id,
                    "token": _safe_decode_token(module.tokenizer, token_id),
                    "delta_logit": float(delta_logits[token_id].item()),
                    "clean_logit": float(clean_logits[token_id].item()),
                    "steered_logit": float(steered_logits[token_id].item()),
                    "clean_prob": float(clean_probs[token_id].item()),
                    "steered_prob": float(steered_probs[token_id].item()),
                }
            )
        return records

    return _build_records(pos_ids, pos_vals), _build_records(neg_ids, neg_vals)


def _jaccard_similarity(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return float(len(inter) / len(union))


def _first_step_below_ratio(values: Sequence[float], ratio: float) -> Optional[int]:
    if not values:
        return None
    base = float(values[0])
    if abs(base) <= EPS:
        return 0
    threshold = abs(base) * float(ratio)
    for idx, value in enumerate(values):
        if abs(float(value)) <= threshold:
            return int(idx)
    return None


def _summarize_logit_shift_steps(step_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_records:
        return {
            "observed_steps": 0,
            "argmax_switch_rate": 0.0,
            "mean_js_divergence": 0.0,
            "max_js_divergence": 0.0,
            "max_js_step": None,
            "mean_reference_rank_lift": 0.0,
            "mean_reference_prob_delta": 0.0,
            "mean_delta_l2": 0.0,
            "step0_top_positive_persistence_jaccard": 0.0,
            "half_life_step_l2_ratio_0_5": None,
        }

    delta_l2_values = [float(step.get("delta_l2", 0.0)) for step in step_records]
    js_values = [float(step.get("js_divergence", 0.0)) for step in step_records]
    rank_lifts = [float(step.get("reference_rank_lift", 0.0)) for step in step_records]
    reference_prob_deltas = [float(step.get("reference_delta_prob", 0.0)) for step in step_records]
    argmax_switches = [1.0 if bool(step.get("argmax_changed", False)) else 0.0 for step in step_records]

    max_js = max(js_values) if js_values else 0.0
    max_js_step = js_values.index(max_js) if js_values else None

    top_positive_sets: List[Set[int]] = []
    for step in step_records:
        ids = {
            int(item.get("token_id"))
            for item in step.get("top_positive_delta_tokens", [])
            if isinstance(item, dict) and "token_id" in item
        }
        top_positive_sets.append(ids)

    step0_set = top_positive_sets[0] if top_positive_sets else set()
    if top_positive_sets:
        persistence_scores = [_jaccard_similarity(step0_set, current) for current in top_positive_sets]
        mean_persistence = float(sum(persistence_scores) / len(persistence_scores))
    else:
        mean_persistence = 0.0

    return {
        "observed_steps": len(step_records),
        "argmax_switch_rate": float(sum(argmax_switches) / len(argmax_switches)),
        "mean_js_divergence": float(sum(js_values) / len(js_values)),
        "max_js_divergence": float(max_js),
        "max_js_step": int(max_js_step) if max_js_step is not None else None,
        "mean_reference_rank_lift": float(sum(rank_lifts) / len(rank_lifts)),
        "mean_reference_prob_delta": float(sum(reference_prob_deltas) / len(reference_prob_deltas)),
        "mean_delta_l2": float(sum(delta_l2_values) / len(delta_l2_values)),
        "step0_top_positive_persistence_jaccard": float(mean_persistence),
        "half_life_step_l2_ratio_0_5": _first_step_below_ratio(delta_l2_values, ratio=0.5),
    }


@torch.no_grad()
def _collect_logit_shift_trace(
    module: Any,
    *,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    reference_completion_token_ids: Sequence[int],
    reference_name: str,
    max_steps: int,
    top_k: int,
    include_special_tokens: bool,
    feature_id: int,
    steer_value: float,
    intervention_scope: str,
    intervention_steps: int,
    prompt_positions: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    reference_ids = [int(x) for x in reference_completion_token_ids]
    steps_to_analyze = max(0, min(int(max_steps), len(reference_ids)))
    if steps_to_analyze <= 0:
        return {
            "reference": str(reference_name),
            "configured_max_steps": int(max_steps),
            "observed_steps": 0,
            "top_k": int(top_k),
            "include_special_tokens": bool(include_special_tokens),
            "summary": _summarize_logit_shift_steps([]),
            "steps": [],
        }

    tokens = prompt_input_ids.clone().to(module.device)
    mask = prompt_attention_mask.clone().to(module.device)
    prompt_len = int(tokens.shape[1])
    steer_steps = max(0, int(intervention_steps))
    special_token_ids: Optional[Set[int]] = None

    step_records: List[Dict[str, Any]] = []
    for step_idx in range(steps_to_analyze):
        clean_full = _run_base_logits(module, input_ids=tokens, attention_mask=mask)
        clean_next = clean_full[:, -1, :].detach().float().squeeze(0).cpu()

        steered_applied = step_idx < steer_steps
        if steered_applied:
            steered_full = _run_steered_logits(
                module,
                input_ids=tokens,
                attention_mask=mask,
                feature_id=int(feature_id),
                steer_value=float(steer_value),
                scope=str(intervention_scope),
                prompt_len=prompt_len,
                prompt_positions=prompt_positions,
            )
            steered_next = steered_full[:, -1, :].detach().float().squeeze(0).cpu()
        else:
            steered_next = clean_next.clone()

        delta = steered_next - clean_next
        clean_probs = torch.softmax(clean_next, dim=-1)
        steered_probs = torch.softmax(steered_next, dim=-1)
        mixture_probs = 0.5 * (clean_probs + steered_probs)

        if special_token_ids is None:
            special_token_ids = _build_special_token_id_set(module.tokenizer, vocab_size=int(delta.shape[0]))
        pos_records, neg_records = _collect_top_delta_records(
            module=module,
            delta_logits=delta,
            clean_logits=clean_next,
            steered_logits=steered_next,
            clean_probs=clean_probs,
            steered_probs=steered_probs,
            top_k=int(top_k),
            special_token_ids=special_token_ids,
            include_special_tokens=bool(include_special_tokens),
        )

        reference_token_id = int(reference_ids[step_idx])
        clean_rank = _safe_token_rank_desc(clean_next, reference_token_id)
        steered_rank = _safe_token_rank_desc(steered_next, reference_token_id)
        clean_argmax = int(torch.argmax(clean_next).item())
        steered_argmax = int(torch.argmax(steered_next).item())
        kl_clean_to_steered = float(
            torch.sum(clean_probs * (torch.log(clean_probs + EPS) - torch.log(steered_probs + EPS))).item()
        )
        kl_steered_to_clean = float(
            torch.sum(steered_probs * (torch.log(steered_probs + EPS) - torch.log(clean_probs + EPS))).item()
        )
        js_divergence = float(
            0.5
            * (
                torch.sum(clean_probs * (torch.log(clean_probs + EPS) - torch.log(mixture_probs + EPS)))
                + torch.sum(steered_probs * (torch.log(steered_probs + EPS) - torch.log(mixture_probs + EPS)))
            ).item()
        )
        step_records.append(
            {
                "step": int(step_idx),
                "steering_applied": bool(steered_applied),
                "reference_token_id": reference_token_id,
                "reference_token": _safe_decode_token(module.tokenizer, reference_token_id),
                "clean_argmax_token_id": clean_argmax,
                "clean_argmax_token": _safe_decode_token(module.tokenizer, clean_argmax),
                "steered_argmax_token_id": steered_argmax,
                "steered_argmax_token": _safe_decode_token(module.tokenizer, steered_argmax),
                "argmax_changed": bool(clean_argmax != steered_argmax),
                "reference_clean_logit": float(clean_next[reference_token_id].item())
                if 0 <= reference_token_id < clean_next.shape[0]
                else None,
                "reference_steered_logit": float(steered_next[reference_token_id].item())
                if 0 <= reference_token_id < steered_next.shape[0]
                else None,
                "reference_delta_logit": float(delta[reference_token_id].item())
                if 0 <= reference_token_id < delta.shape[0]
                else None,
                "reference_clean_prob": float(clean_probs[reference_token_id].item())
                if 0 <= reference_token_id < clean_probs.shape[0]
                else None,
                "reference_steered_prob": float(steered_probs[reference_token_id].item())
                if 0 <= reference_token_id < steered_probs.shape[0]
                else None,
                "reference_delta_prob": float((steered_probs[reference_token_id] - clean_probs[reference_token_id]).item())
                if 0 <= reference_token_id < clean_probs.shape[0]
                else None,
                "reference_clean_rank": int(clean_rank),
                "reference_steered_rank": int(steered_rank),
                "reference_rank_lift": int(clean_rank - steered_rank),
                "delta_l2": float(torch.linalg.vector_norm(delta, ord=2).item()),
                "delta_linf": float(torch.max(torch.abs(delta)).item()),
                "kl_clean_to_steered": max(0.0, kl_clean_to_steered),
                "kl_steered_to_clean": max(0.0, kl_steered_to_clean),
                "js_divergence": max(0.0, js_divergence),
                "top_positive_delta_tokens": pos_records,
                "top_negative_delta_tokens": neg_records,
            }
        )

        forced_token = torch.tensor([[reference_token_id]], dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([tokens, forced_token], dim=1)
        forced_mask = torch.ones_like(forced_token, dtype=mask.dtype, device=mask.device)
        mask = torch.cat([mask, forced_mask], dim=1)

    return {
        "reference": str(reference_name),
        "configured_max_steps": int(max_steps),
        "observed_steps": len(step_records),
        "top_k": int(top_k),
        "include_special_tokens": bool(include_special_tokens),
        "summary": _summarize_logit_shift_steps(step_records),
        "steps": step_records,
    }


def _resolve_strength_base(
    *,
    scope: str,
    neuronpedia_max_value: float,
    last_prompt_token_activation: float,
) -> Tuple[float, str]:
    if scope in {"last_token_only", "last_original_token_only"}:
        if abs(last_prompt_token_activation) > EPS:
            return float(last_prompt_token_activation), "last_prompt_token_activation"
        return float(neuronpedia_max_value), "fallback_neuronpedia_max_value"
    return float(neuronpedia_max_value), "neuronpedia_max_value"


def _build_source(layer_id: int, width: str) -> str:
    return f"{int(layer_id)}-gemmascope-res-{str(width)}"


def _resolve_output_path(
    *,
    output_root: str,
    layer_id: int,
    feature_id: int,
    scope: str,
    intervention_steps: int,
    filename: str,
) -> Path:
    return (
        Path(output_root)
        / f"layer-{int(layer_id)}"
        / f"feature-{int(feature_id)}"
        / str(scope)
        / f"steer_steps_{int(intervention_steps)}"
        / str(filename)
    )


def run_neuronpedia_steer(
    args: argparse.Namespace,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    sae: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from model_with_sae import ModelWithSAEModule

    source = _build_source(layer_id=int(args.layer_id), width=str(args.width))
    payload = _fetch_feature_json(
        model_id=str(args.model_id),
        source=source,
        feature_id=str(args.feature_id),
        api_key=args.neuronpedia_api_key,
        timeout=int(args.timeout),
    )

    activations_raw = payload.get("activations", [])
    activations = [item for item in activations_raw if isinstance(item, dict)]
    selected, selected_indices = _select_activations_method_2(
        activations,
        n=max(1, int(args.top_k_examples)),
    )

    if args.sae_path:
        sae_uri = str(args.sae_path)
        resolved_average_l0 = "from_sae_path"
    else:
        sae_uri, resolved_average_l0 = build_default_sae_path(
            layer_id=str(args.layer_id),
            width=str(args.width),
            release=str(args.sae_release),
            average_l0=None,
            canonical_map_path=Path(args.canonical_map_path),
        )

    module = ModelWithSAEModule(
        llm_name=str(args.llm_name),
        sae_path=sae_uri,
        sae_layer=int(args.layer_id),
        feature_index=int(args.feature_id),
        device=str(args.device),
        model=model,
        tokenizer=tokenizer,
        sae=sae,
    )

    scales = _resolve_strength_scales(args.strength_scales)
    enable_logit_analysis = not bool(args.disable_logit_analysis)
    logit_analysis_max_steps = max(0, int(args.logit_analysis_max_steps))
    logit_analysis_top_k = max(0, int(args.logit_analysis_top_k))
    include_special_tokens = bool(args.logit_analysis_include_special_tokens)
    logit_reference_mode = str(args.logit_analysis_reference)
    sample_results: List[Dict[str, Any]] = []
    for rank, activation in tqdm(enumerate(selected, start=1), desc="Selected samples"):
        trunc_info = _truncate_prompt_from_activation(
            activation,
            max_prefix_tokens=max(0, int(args.max_prefix_tokens)),
            scope=str(args.intervention_scope),
        )
        prompt_text = str(trunc_info["prompt_text"])
        input_ids, attention_mask = _prepare_prompt_tensors(module, prompt_text)

        clean_output = _generate_text(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max(1, int(args.max_new_tokens)),
            temperature=float(args.temperature),
            feature_id=None,
            steer_value=None,
            intervention_scope=str(args.intervention_scope),
            intervention_steps=max(0, int(args.intervention_steps)),
        )
        clean_completion_token_ids = clean_output.get("completion_token_ids", [])

        trace = module.get_activation_trace(prompt_text)
        per_token_activation = trace.get("per_token_activation", [])
        if isinstance(per_token_activation, list) and per_token_activation:
            last_prompt_token_activation = _to_float(per_token_activation[-1], default=0.0)
        else:
            last_prompt_token_activation = 0.0
        natural_support = _build_natural_support_metadata(
            trace=trace,
            prompt_len=int(input_ids.shape[1]),
        )
        prompt_positions: Optional[List[int]]
        if str(args.intervention_scope) == "natural_support_mask":
            prompt_positions = list(natural_support["selected_positions"])
        else:
            prompt_positions = None

        neuronpedia_max_value = _to_float(trunc_info["max_value"], default=0.0)
        base_activation, base_source = _resolve_strength_base(
            scope=str(args.intervention_scope),
            neuronpedia_max_value=neuronpedia_max_value,
            last_prompt_token_activation=last_prompt_token_activation,
        )

        interventions: List[Dict[str, Any]] = []
        for scale in tqdm(scales, desc="Strength scales"):
            steer_value = float(base_activation * scale)
            steered_output = _generate_text(
                module,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max(1, int(args.max_new_tokens)),
                temperature=float(args.temperature),
                feature_id=int(args.feature_id),
                steer_value=steer_value,
                intervention_scope=str(args.intervention_scope),
                intervention_steps=max(0, int(args.intervention_steps)),
                prompt_positions=prompt_positions,
            )
            steered_completion_token_ids = steered_output.get("completion_token_ids", [])

            if enable_logit_analysis and logit_analysis_max_steps > 0 and logit_analysis_top_k > 0:
                if logit_reference_mode == "steered":
                    reference_name = "steered_completion"
                    reference_ids = steered_completion_token_ids
                else:
                    reference_name = "clean_completion"
                    reference_ids = clean_completion_token_ids

                logit_analysis = _collect_logit_shift_trace(
                    module,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attention_mask,
                    reference_completion_token_ids=reference_ids,
                    reference_name=reference_name,
                    max_steps=logit_analysis_max_steps,
                    top_k=logit_analysis_top_k,
                    include_special_tokens=include_special_tokens,
                    feature_id=int(args.feature_id),
                    steer_value=steer_value,
                    intervention_scope=str(args.intervention_scope),
                    intervention_steps=max(0, int(args.intervention_steps)),
                    prompt_positions=prompt_positions,
                )
            else:
                logit_analysis = {
                    "reference": "disabled",
                    "configured_max_steps": logit_analysis_max_steps,
                    "observed_steps": 0,
                    "top_k": logit_analysis_top_k,
                    "include_special_tokens": include_special_tokens,
                    "summary": _summarize_logit_shift_steps([]),
                    "steps": [],
                }
            interventions.append(
                {
                    "scale": float(scale),
                    "steer_value": steer_value,
                    "steered_output": steered_output,
                    "logit_analysis": logit_analysis,
                }
            )

        sample_results.append(
            {
                "rank": int(rank),
                "activation_original_index": int(selected_indices[rank - 1]),
                "truncated_prompt": trunc_info,
                "prompt_last_token_activation": float(last_prompt_token_activation),
                "prompt_activation_trace": trace,
                "natural_support_mask": natural_support,
                "strength_base_activation": float(base_activation),
                "strength_base_source": str(base_source),
                "clean_output": clean_output,
                "interventions": interventions,
            }
        )

    output_path = _resolve_output_path(
        output_root=str(args.output_root),
        layer_id=int(args.layer_id),
        feature_id=int(args.feature_id),
        scope=str(args.intervention_scope),
        intervention_steps=int(args.intervention_steps),
        filename=str(args.output_filename),
    )
    result_payload: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "layer_id": int(args.layer_id),
            "feature_id": int(args.feature_id),
            "model_id": str(args.model_id),
            "llm_name": str(args.llm_name),
            "device": str(args.device),
            "width": str(args.width),
            "sae_release": str(args.sae_release),
            "sae_uri": str(sae_uri),
            "resolved_average_l0": str(resolved_average_l0),
            "intervention_scope": str(args.intervention_scope),
            "intervention_steps": int(args.intervention_steps),
            "temperature": float(args.temperature),
            "max_new_tokens": int(args.max_new_tokens),
            "top_k_examples": int(args.top_k_examples),
            "selection_method": 2,
            "max_prefix_tokens": int(args.max_prefix_tokens),
            "strength_scales": scales,
            "logit_analysis_enabled": enable_logit_analysis,
            "logit_analysis_max_steps": logit_analysis_max_steps,
            "logit_analysis_top_k": logit_analysis_top_k,
            "logit_analysis_reference": logit_reference_mode,
            "logit_analysis_include_special_tokens": include_special_tokens,
            "output_path": str(output_path),
        },
        "neuronpedia": {
            "source": source,
            "total_activations": len(activations),
            "selected_indices": selected_indices,
        },
        "samples": sample_results,
    }
    _write_json_atomic(output_path, result_payload)
    return result_payload


def main(
    cli_args: Optional[argparse.Namespace] = None,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    sae: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    args = cli_args or _parse_args()
    return run_neuronpedia_steer(args, model=model, tokenizer=tokenizer, sae=sae)


if __name__ == "__main__":
    final_payload = main()
    summary = {
        "output_path": final_payload.get("metadata", {}).get("output_path"),
        "sample_count": len(final_payload.get("samples", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
