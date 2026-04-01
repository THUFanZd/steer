from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

EPS = 1e-8
NEURONPEDIA_BASE_URL = "https://www.neuronpedia.org"


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
        for item in args.target_pairs:
            pairs.append(_parse_pair_text(item))

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each (layer, feature), read run_steer_batch outputs under outputs/<timestamp>/..., "
            "exclude scale=0 interventions, and compute hit_ratio where the token immediately after "
            "the truncation point belongs to Neuronpedia pos_str/neg_str."
        )
    )
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--target-pairs", type=str, nargs="*", default=None)
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


def _resolve_model_width_from_result(result_payload: Dict[str, Any], fallback_model: str, fallback_width: str) -> Tuple[str, str]:
    metadata = result_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return fallback_model, fallback_width
    model_id = str(metadata.get("model_id") or fallback_model)
    width = str(metadata.get("width") or fallback_width)
    return model_id, width


def _feature_stats(
    *,
    layer_id: int,
    feature_id: int,
    result_files: Sequence[Path],
    token_set: Set[str],
    activations: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    numerator = 0
    denominator = 0
    skipped_zero_scale = 0
    skipped_missing_activation = 0
    skipped_missing_next_token = 0
    skipped_bad_intervention = 0

    for path in result_files:
        payload = _load_json(path)
        samples = payload.get("samples", [])
        if not isinstance(samples, list):
            continue

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
                    skipped_bad_intervention += 1
                    continue

                scale = _to_float(intervention.get("scale"), default=float("nan"))
                if math.isnan(scale):
                    skipped_bad_intervention += 1
                    continue
                if abs(scale) <= EPS:
                    skipped_zero_scale += 1
                    continue

                denominator += 1
                if sample_activation is None:
                    skipped_missing_activation += 1
                    continue
                if not next_token:
                    skipped_missing_next_token += 1
                    continue
                if next_token in token_set:
                    numerator += 1

    ratio = (float(numerator) / float(denominator)) if denominator > 0 else 0.0
    return {
        "layer_id": int(layer_id),
        "feature_id": int(feature_id),
        "result_file_count": len(result_files),
        "hit_count": int(numerator),
        "eligible_intervention_count": int(denominator),
        "hit_ratio": ratio,
        "skipped_zero_scale_count": int(skipped_zero_scale),
        "skipped_missing_activation_count": int(skipped_missing_activation),
        "skipped_missing_next_token_count": int(skipped_missing_next_token),
        "skipped_bad_intervention_count": int(skipped_bad_intervention),
    }


def main() -> None:
    args = _parse_args()
    pairs = _collect_target_pairs(args)

    ts_root = Path(args.outputs_root) / str(args.timestamp)
    if not ts_root.exists():
        raise FileNotFoundError(f"Timestamp directory not found: {ts_root}")

    per_feature: List[Dict[str, Any]] = []
    all_hit = 0
    all_eligible = 0

    for layer_id, feature_id in pairs:
        feature_root = ts_root / f"layer-{int(layer_id)}" / f"feature-{int(feature_id)}"
        result_files = _find_result_files(feature_root, str(args.result_filename))
        if not result_files:
            per_feature.append(
                {
                    "layer_id": int(layer_id),
                    "feature_id": int(feature_id),
                    "result_file_count": 0,
                    "hit_count": 0,
                    "eligible_intervention_count": 0,
                    "hit_ratio": 0.0,
                    "warning": f"No result files found under {feature_root}",
                }
            )
            continue

        sample_payload = _load_json(result_files[0])
        model_id, width = _resolve_model_width_from_result(
            sample_payload,
            fallback_model=str(args.model_id),
            fallback_width=str(args.width),
        )
        source = _build_source(layer_id=layer_id, width=width)
        neuronpedia_payload = fetch_feature_json(
            model_id=str(model_id),
            source=source,
            feature_id=str(feature_id),
            api_key=args.neuronpedia_api_key,
            timeout=int(args.timeout),
        )
        token_set = _build_token_set(neuronpedia_payload)
        activations_raw = neuronpedia_payload.get("activations", [])
        activations = [item for item in activations_raw if isinstance(item, dict)]

        stats = _feature_stats(
            layer_id=layer_id,
            feature_id=feature_id,
            result_files=result_files,
            token_set=token_set,
            activations=activations,
        )
        stats["model_id"] = str(model_id)
        stats["source"] = str(source)
        stats["width"] = str(width)
        stats["token_set_size"] = len(token_set)
        stats["activation_count"] = len(activations)

        all_hit += int(stats["hit_count"])
        all_eligible += int(stats["eligible_intervention_count"])
        per_feature.append(stats)

    overall_ratio = (float(all_hit) / float(all_eligible)) if all_eligible > 0 else 0.0
    output_payload = {
        "metadata": {
            "outputs_root": str(Path(args.outputs_root).resolve()),
            "timestamp": str(args.timestamp),
            "target_pair_count": len(pairs),
            "result_filename": str(args.result_filename),
        },
        "overall": {
            "hit_count": int(all_hit),
            "eligible_intervention_count": int(all_eligible),
            "hit_ratio": overall_ratio,
        },
        "features": per_feature,
    }

    print(json.dumps(output_payload, ensure_ascii=False, indent=2))

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
