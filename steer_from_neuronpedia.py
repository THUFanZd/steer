from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import torch
from function import DEFAULT_CANONICAL_MAP_PATH, build_default_sae_path

SENTENCE_END_RE = re.compile(r"[.!?。！？]")
EPS = 1e-8
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
    parser.add_argument("--model-id", type=str, default="gemma-2-2b", help="Neuronpedia model id.")
    parser.add_argument("--llm-name", type=str, default="google/gemma-2-2b", help="HF model name/path.")
    parser.add_argument("--width", type=str, default="16k")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res")
    parser.add_argument("--canonical-map-path", type=str, default=str(DEFAULT_CANONICAL_MAP_PATH))
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--intervention-scope",
        type=str,
        choices=(
            "all_tokens",
            "last_token_only",
            "all_original_tokens",
            "last_original_token_only",
        ),
        default="last_token_only",
    )
    parser.add_argument(
        "--intervention-steps",
        type=int,
        default=1,
        help="Apply steering only for the first N predicted tokens, then continue with clean decoding.",
    )
    parser.add_argument("--top-k-examples", type=int, default=3)
    parser.add_argument("--max-prefix-tokens", type=int, default=20)
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
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--output-filename", type=str, default="steer_from_neuronpedia.json")
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
        return [0.0, 2.0 / 3.0, 1.5]

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


def _truncate_prompt_from_activation(
    activation: Dict[str, Any],
    *,
    max_prefix_tokens: int,
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
            "window_token_count": 0,
            "source_sentence": "",
            "source_token_count": 0,
        }

    max_idx = _find_first_max_index(values, declared_max, fallback_idx)
    max_idx = max(0, min(max_idx, len(tokens) - 1))

    sentence_start = _find_sentence_start_token_idx(tokens, max_idx)
    max_prefix_tokens = max(0, int(max_prefix_tokens))
    window_start = max(sentence_start, max_idx - max_prefix_tokens)

    truncated_tokens = tokens[window_start : max_idx + 1]
    prompt_text = "".join(truncated_tokens).strip()
    if not prompt_text:
        prompt_text = "".join(tokens[: max_idx + 1]).strip()

    max_token = tokens[max_idx]
    return {
        "prompt_text": prompt_text,
        "max_token": max_token,
        "max_value": declared_max,
        "max_token_index": max_idx,
        "sentence_start_index": sentence_start,
        "window_start_index": window_start,
        "window_token_count": len(truncated_tokens),
        "source_sentence": "".join(tokens),
        "source_token_count": len(tokens),
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


def _build_original_scope_value_tensor(
    *,
    seq_len: int,
    prompt_len: int,
    steer_value: float,
    scope: str,
    device: torch.device,
) -> torch.Tensor:
    value = torch.zeros((1, int(seq_len)), dtype=torch.float32, device=device)
    effective_prompt_len = max(0, min(int(prompt_len), int(seq_len)))
    if effective_prompt_len <= 0:
        return value

    if scope == "all_original_tokens":
        value[:, :effective_prompt_len] = float(steer_value)
        return value
    if scope == "last_original_token_only":
        value[:, effective_prompt_len - 1] = float(steer_value)
        return value
    raise ValueError(f"Unsupported original-token scope: {scope}")


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

    value_tensor = _build_original_scope_value_tensor(
        seq_len=int(input_ids.shape[1]),
        prompt_len=int(prompt_len),
        steer_value=float(steer_value),
        scope=str(scope),
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
    return {
        "completion_text": completion_text,
        "full_text": full_text,
        "generated_token_count": int(completion_ids.numel()),
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
    sample_results: List[Dict[str, Any]] = []
    for rank, activation in enumerate(selected, start=1):
        trunc_info = _truncate_prompt_from_activation(
            activation,
            max_prefix_tokens=max(0, int(args.max_prefix_tokens)),
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

        trace = module.get_activation_trace(prompt_text)
        per_token_activation = trace.get("per_token_activation", [])
        if isinstance(per_token_activation, list) and per_token_activation:
            last_prompt_token_activation = _to_float(per_token_activation[-1], default=0.0)
        else:
            last_prompt_token_activation = 0.0

        neuronpedia_max_value = _to_float(trunc_info["max_value"], default=0.0)
        base_activation, base_source = _resolve_strength_base(
            scope=str(args.intervention_scope),
            neuronpedia_max_value=neuronpedia_max_value,
            last_prompt_token_activation=last_prompt_token_activation,
        )

        interventions: List[Dict[str, Any]] = []
        for scale in scales:
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
            )
            interventions.append(
                {
                    "scale": float(scale),
                    "steer_value": steer_value,
                    "steered_output": steered_output,
                }
            )

        sample_results.append(
            {
                "rank": int(rank),
                "activation_original_index": int(selected_indices[rank - 1]),
                "truncated_prompt": trunc_info,
                "prompt_last_token_activation": float(last_prompt_token_activation),
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
