from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

EPS = 1e-8
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"
SCOPES = {
    "all_tokens",
    "last_token_only",
    "all_original_tokens",
    "last_original_token_only",
    "natural_support_mask",
    "online_reactivation_gating",
}
SENTENCEPIECE_SPACE = chr(9601)  # '▁'


def _parse_pair_text(text: str) -> Tuple[int, int]:
    raw = str(text).strip()
    if not raw:
        raise ValueError("Empty target pair.")

    normalized = raw.replace("(", "").replace(")", "").replace(" ", "")
    if ":" in normalized:
        parts = normalized.split(":")
    elif "," in normalized:
        parts = normalized.split(",")
    else:
        raise ValueError(f"Invalid pair format: {text!r}. Use layer,feature or layer:feature.")

    if len(parts) != 2:
        raise ValueError(f"Invalid pair format: {text!r}.")
    return int(parts[0]), int(parts[1])


def _collect_target_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []

    if args.target_pairs:
        for group in args.target_pairs:
            for item in group:
                text = str(item).strip()
                if not text:
                    continue
                for token in text.split(";"):
                    token = token.strip()
                    if token:
                        pairs.append(_parse_pair_text(token))

    if args.target_pairs_file:
        path = Path(args.target_pairs_file)
        if not path.exists():
            raise FileNotFoundError(f"target pairs file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pairs.append(_parse_pair_text(line))

    deduped: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)

    if not deduped:
        raise ValueError(
            "No target pairs provided. Use --target-pairs and/or --target-pairs-file."
        )
    return deduped


def _collect_timestamps(args: argparse.Namespace) -> List[str]:
    values: List[str] = []

    if args.timestamp:
        values.extend(str(x) for x in args.timestamp)
    if args.timestamps:
        values.extend(str(x) for x in args.timestamps)

    parsed: List[str] = []
    seen: Set[str] = set()
    for item in values:
        text = item.strip()
        if not text:
            continue
        for token in text.split(","):
            ts = token.strip()
            if ts and ts not in seen:
                seen.add(ts)
                parsed.append(ts)

    if not parsed:
        raise ValueError(
            "No timestamps provided. Use --timestamp (repeatable) or --timestamps."
        )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read run_steer_batch outputs under outputs/<timestamp>/... for selected features, "
            "exclude scale=0 interventions, and compute hit_ratio where token after truncation "
            "max token belongs to Neuronpedia pos_str/neg_str."
        )
    )
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument(
        "--timestamp",
        action="append",
        default=None,
        help="Single timestamp. Can be passed multiple times.",
    )
    parser.add_argument(
        "--timestamps",
        nargs="*",
        default=None,
        help="Space/comma separated timestamps.",
    )
    parser.add_argument(
        "--target-pairs",
        nargs="+",
        action="append",
        default=None,
        help="One or more pairs per occurrence, e.g. --target-pairs 6,123 6,124",
    )
    parser.add_argument("--target-pairs-file", type=str, default=None)
    parser.add_argument(
        "--result-filename",
        type=str,
        default="steer_from_neuronpedia.json",
        help="Result filename produced by steer_from_neuronpedia.py",
    )
    parser.add_argument("--model-id", type=str, default="gemma-2-2b")
    parser.add_argument("--width", type=str, default="16k")
    parser.add_argument("--neuronpedia-api-key", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--count-token-source",
        type=str,
        choices=("generated", "activation"),
        default="generated",
        help=(
            "generated: count using first token in steered_output.completion_text; "
            "activation: count using token after maxValueTokenIndex in Neuronpedia activation."
        ),
    )
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def fetch_feature_json(
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
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Neuronpedia response is not a JSON object.")
            return payload
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


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _find_first_max_index(values: Sequence[Any], declared_max: float, fallback_index: Optional[int]) -> int:
    parsed_values = [_to_float(v, default=0.0) for v in values]
    if parsed_values:
        for idx, value in enumerate(parsed_values):
            if abs(value - declared_max) <= EPS:
                return idx
        return int(max(range(len(parsed_values)), key=lambda i: parsed_values[i]))
    if isinstance(fallback_index, int) and fallback_index >= 0:
        return int(fallback_index)
    return 0


def _extract_next_token(activation: Dict[str, Any]) -> str:
    tokens_raw = activation.get("tokens", [])
    if not isinstance(tokens_raw, list) or not tokens_raw:
        return ""
    tokens = [str(t) for t in tokens_raw]

    max_idx_raw = activation.get("maxValueTokenIndex")
    if isinstance(max_idx_raw, int):
        max_idx = max(0, min(int(max_idx_raw), len(tokens) - 1))
    else:
        values_raw = activation.get("values", [])
        values: List[Any] = list(values_raw) if isinstance(values_raw, list) else []
        declared_max = _to_float(activation.get("maxValue"), default=0.0)
        max_idx = _find_first_max_index(values, declared_max, None)
        max_idx = max(0, min(int(max_idx), len(tokens) - 1))

    next_idx = max_idx + 1
    if next_idx >= len(tokens):
        return ""
    return tokens[next_idx]


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON payload in {path}")
    return payload


def _find_result_files(feature_root: Path, filename: str) -> List[Path]:
    return sorted(p for p in feature_root.rglob(filename) if p.is_file())


def _build_source(layer_id: int, width: str) -> str:
    return f"{int(layer_id)}-gemmascope-res-{str(width)}"


def _build_token_set(payload: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()
    for field in ("pos_str", "neg_str"):
        raw = payload.get(field, [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            text = str(item)
            if text:
                tokens.add(text)
    return tokens


def _extract_first_generated_piece(completion_text: str) -> str:
    text = str(completion_text or "").lstrip()
    if not text:
        return ""
    match = re.match(r"\S+", text)
    if not match:
        return ""
    return match.group(0)


def _candidate_generated_tokens(piece: str, token_set: Set[str]) -> Set[str]:
    if not piece:
        return set()

    trimmed = piece.strip()
    stripped = trimmed.strip(".,;:!?()[]{}\"'`")
    base_variants = {piece, trimmed, stripped}
    base_variants = {x for x in base_variants if x}

    prefix_chars = {SENTENCEPIECE_SPACE, "Ġ", " "}
    # Also infer prefixes from payload tokens.
    for tok in token_set:
        if tok and not tok[0].isalnum():
            prefix_chars.add(tok[0])

    candidates: Set[str] = set()
    for base in base_variants:
        candidates.add(base)
        for p in prefix_chars:
            candidates.add(f"{p}{base}")
    return candidates


def _resolve_model_width_from_result(
    result_payload: Dict[str, Any],
    fallback_model: str,
    fallback_width: str,
) -> Tuple[str, str]:
    metadata = result_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return fallback_model, fallback_width
    model_id = str(metadata.get("model_id") or fallback_model)
    width = str(metadata.get("width") or fallback_width)
    return model_id, width


def _parse_steps_from_path(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.match(r"^steer_steps_(\d+)$", str(part))
        if match:
            return int(match.group(1))
    return None


def _parse_scope_from_path(path: Path) -> str:
    for part in path.parts:
        text = str(part)
        if text in SCOPES:
            return text
    return "unknown"


def _resolve_method_info(result_payload: Dict[str, Any], result_path: Path) -> Dict[str, Any]:
    metadata = result_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    scope = str(metadata.get("intervention_scope") or _parse_scope_from_path(result_path))

    steps_raw = metadata.get("intervention_steps")
    if isinstance(steps_raw, int):
        steps = int(steps_raw)
    else:
        steps = _parse_steps_from_path(result_path)

    if steps is None:
        label = f"scope={scope}|steps=unknown"
    else:
        label = f"scope={scope}|steps={steps}"

    return {
        "method_label": label,
        "intervention_scope": scope,
        "intervention_steps": steps,
    }


def _new_counter() -> Dict[str, int]:
    return {
        "result_file_count": 0,
        "hit_count": 0,
        "eligible_intervention_count": 0,
        "skipped_zero_scale_count": 0,
        "skipped_missing_activation_count": 0,
        "skipped_missing_next_token_count": 0,
        "skipped_bad_intervention_count": 0,
    }


def _add_counter(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key in dst.keys():
        dst[key] += int(src.get(key, 0))


def _finalize_counter(counter: Dict[str, int]) -> Dict[str, Any]:
    eligible = int(counter["eligible_intervention_count"])
    hit = int(counter["hit_count"])
    ratio = (float(hit) / float(eligible)) if eligible > 0 else 0.0
    out = dict(counter)
    out["hit_ratio"] = ratio
    return out


def _compute_file_counter(
    *,
    result_payload: Dict[str, Any],
    token_set: Set[str],
    activations: Sequence[Dict[str, Any]],
    count_token_source: str,
) -> Dict[str, int]:
    counter = _new_counter()
    counter["result_file_count"] = 1

    samples = result_payload.get("samples", [])
    if not isinstance(samples, list):
        return counter

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        idx = sample.get("activation_original_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(activations):
            sample_activation = None
            next_token = ""
        else:
            sample_activation = activations[idx]
            next_token = _extract_next_token(sample_activation)

        interventions = sample.get("interventions", [])
        if not isinstance(interventions, list):
            continue

        for intervention in interventions:
            if not isinstance(intervention, dict):
                counter["skipped_bad_intervention_count"] += 1
                continue

            scale = _to_float(intervention.get("scale"), default=float("nan"))
            if math.isnan(scale):
                counter["skipped_bad_intervention_count"] += 1
                continue
            if abs(scale) <= EPS:
                counter["skipped_zero_scale_count"] += 1
                continue

            counter["eligible_intervention_count"] += 1
            if str(count_token_source) == "generated":
                steered_output = intervention.get("steered_output", {})
                if not isinstance(steered_output, dict):
                    counter["skipped_missing_next_token_count"] += 1
                    continue
                piece = _extract_first_generated_piece(
                    str(steered_output.get("completion_text", ""))
                )
                if not piece:
                    counter["skipped_missing_next_token_count"] += 1
                    continue
                candidates = _candidate_generated_tokens(piece, token_set)
                if any(candidate in token_set for candidate in candidates):
                    counter["hit_count"] += 1
            else:
                if sample_activation is None:
                    counter["skipped_missing_activation_count"] += 1
                    continue
                if not next_token:
                    counter["skipped_missing_next_token_count"] += 1
                    continue
                if next_token in token_set:
                    counter["hit_count"] += 1

    return counter


def main() -> None:
    args = _parse_args()
    timestamps = _collect_timestamps(args)
    pairs = _collect_target_pairs(args)

    outputs_root = Path(args.outputs_root)

    overall_counter = _new_counter()
    by_feature_counter: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(_new_counter)
    by_layer_counter: Dict[int, Dict[str, int]] = defaultdict(_new_counter)
    by_method_counter: Dict[Tuple[str, Optional[int]], Dict[str, int]] = defaultdict(_new_counter)

    feature_timestamps: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    feature_methods: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
    layer_timestamps: Dict[int, Set[str]] = defaultdict(set)
    method_timestamps: Dict[Tuple[str, Optional[int]], Set[str]] = defaultdict(set)
    method_features: Dict[Tuple[str, Optional[int]], Set[Tuple[int, int]]] = defaultdict(set)

    missing_timestamp_roots: List[Dict[str, Any]] = []
    missing_feature_results: List[Dict[str, Any]] = []

    feature_payload_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for ts in timestamps:
        ts_root = outputs_root / ts
        if not ts_root.exists():
            missing_timestamp_roots.append(
                {
                    "timestamp": ts,
                    "path": str(ts_root),
                    "warning": "Timestamp directory not found.",
                }
            )
            continue

        for layer_id, feature_id in pairs:
            feature_root = ts_root / f"layer-{int(layer_id)}" / f"feature-{int(feature_id)}"
            result_files = _find_result_files(feature_root, str(args.result_filename))
            if not result_files:
                missing_feature_results.append(
                    {
                        "timestamp": ts,
                        "layer_id": int(layer_id),
                        "feature_id": int(feature_id),
                        "path": str(feature_root),
                        "warning": "No result files found.",
                    }
                )
                continue

            for result_file in result_files:
                result_payload = _load_json(result_file)
                model_id, width = _resolve_model_width_from_result(
                    result_payload,
                    fallback_model=str(args.model_id),
                    fallback_width=str(args.width),
                )
                source = _build_source(layer_id=layer_id, width=width)
                cache_key = (str(model_id), str(source), int(feature_id))

                if cache_key not in feature_payload_cache:
                    neuronpedia_payload = fetch_feature_json(
                        model_id=str(model_id),
                        source=str(source),
                        feature_id=str(feature_id),
                        api_key=args.neuronpedia_api_key,
                        timeout=int(args.timeout),
                    )
                    feature_payload_cache[cache_key] = {
                        "token_set": _build_token_set(neuronpedia_payload),
                        "activations": [
                            item
                            for item in neuronpedia_payload.get("activations", [])
                            if isinstance(item, dict)
                        ],
                    }

                token_set = feature_payload_cache[cache_key]["token_set"]
                activations = feature_payload_cache[cache_key]["activations"]

                method_info = _resolve_method_info(result_payload, result_file)
                method_key = (
                    str(method_info["intervention_scope"]),
                    method_info["intervention_steps"],
                )

                file_counter = _compute_file_counter(
                    result_payload=result_payload,
                    token_set=token_set,
                    activations=activations,
                    count_token_source=str(args.count_token_source),
                )

                _add_counter(overall_counter, file_counter)
                _add_counter(by_feature_counter[(int(layer_id), int(feature_id))], file_counter)
                _add_counter(by_layer_counter[int(layer_id)], file_counter)
                _add_counter(by_method_counter[method_key], file_counter)

                feature_timestamps[(int(layer_id), int(feature_id))].add(ts)
                feature_methods[(int(layer_id), int(feature_id))].add(str(method_info["method_label"]))
                layer_timestamps[int(layer_id)].add(ts)
                method_timestamps[method_key].add(ts)
                method_features[method_key].add((int(layer_id), int(feature_id)))

    by_feature: List[Dict[str, Any]] = []
    for (layer_id, feature_id), counter in sorted(by_feature_counter.items()):
        row = {
            "layer_id": int(layer_id),
            "feature_id": int(feature_id),
            "timestamps": sorted(feature_timestamps[(layer_id, feature_id)]),
            "method_labels": sorted(feature_methods[(layer_id, feature_id)]),
        }
        row.update(_finalize_counter(counter))
        by_feature.append(row)

    by_layer: List[Dict[str, Any]] = []
    for layer_id, counter in sorted(by_layer_counter.items()):
        row = {
            "layer_id": int(layer_id),
            "timestamps": sorted(layer_timestamps[layer_id]),
        }
        row.update(_finalize_counter(counter))
        by_layer.append(row)

    by_method: List[Dict[str, Any]] = []
    for (scope, steps), counter in sorted(
        by_method_counter.items(),
        key=lambda item: (item[0][0], -1 if item[0][1] is None else int(item[0][1])),
    ):
        label = f"scope={scope}|steps={steps if steps is not None else 'unknown'}"
        row = {
            "method_label": label,
            "intervention_scope": str(scope),
            "intervention_steps": steps,
            "timestamps": sorted(method_timestamps[(scope, steps)]),
            "feature_count": len(method_features[(scope, steps)]),
        }
        row.update(_finalize_counter(counter))
        by_method.append(row)

    output_payload = {
        "metadata": {
            "outputs_root": str(outputs_root.resolve()),
            "timestamps": timestamps,
            "target_pairs": [
                {
                    "layer_id": int(layer_id),
                    "feature_id": int(feature_id),
                }
                for layer_id, feature_id in pairs
            ],
            "result_filename": str(args.result_filename),
            "count_token_source": str(args.count_token_source),
        },
        "overall": _finalize_counter(overall_counter),
        "by_feature": by_feature,
        "by_layer": by_layer,
        "by_method": by_method,
        "missing_timestamp_roots": missing_timestamp_roots,
        "missing_feature_results": missing_feature_results,
    }

    print(json.dumps(output_payload, ensure_ascii=False, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
