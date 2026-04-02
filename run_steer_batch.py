from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from tqdm import tqdm

try:
    from function import DEFAULT_CANONICAL_MAP_PATH, extract_average_l0_from_canonical_map
except ModuleNotFoundError as exc:
    if exc.name != "openai":
        raise
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules.setdefault("openai", openai_stub)
    from function import DEFAULT_CANONICAL_MAP_PATH, extract_average_l0_from_canonical_map

DEFAULT_STEPS = (1, 5, 999)
DEFAULT_SCOPES = (
    "all_tokens",
    "last_token_only",
    "all_original_tokens",
    "last_original_token_only",
)
DEFAULT_STRENGTH_SCALES = ("-1", "-3", "1", "-5", "3") #"-1/3", "0.5",
PAIR_PATTERNS = (
    re.compile(r"^\s*(\d+)\s*[,\t ]\s*(\d+)\s*$"),
    re.compile(r"^\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*$"),
    re.compile(r"layer[-_ ]?(\d+)\D+feature[-_ ]?(\d+)", re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for steer_from_neuronpedia.py. "
            "Reads (layer_id, feature_id) pairs, expands steps/scopes/scales, "
            "and either prints or executes the resulting commands."
        )
    )
    parser.add_argument("--pairs-file", type=Path, default=Path("pairs.txt"))
    parser.add_argument(
        "--steps-file",
        type=Path,
        default=None,
        help="Optional file containing intervention steps (ints). One or more values per line.",
    )
    parser.add_argument(
        "--strength-scales-file",
        type=Path,
        default=None,
        help="Optional file containing strength scales. One or more values per line.",
    )
    parser.add_argument("--python-exe", type=str, default="python")  # sys.executable or 
    parser.add_argument("--script-path", type=Path, default=Path("steer_from_neuronpedia.py"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run folder under output-root. Defaults to current timestamp.",
    )
    parser.add_argument("--llm-name", type=str, default="/data/MODEL/Gemma-2-2b")
    parser.add_argument("--sae-root", type=Path, default=Path("/data/MODEL/gemma-scope-2b-pt-res"))
    parser.add_argument("--canonical-map-path", type=Path, default=DEFAULT_CANONICAL_MAP_PATH)
    parser.add_argument("--width", type=str, default="16k")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k-examples", type=int, default=3)
    parser.add_argument("--steps", nargs="*", type=int, default=list(DEFAULT_STEPS))
    parser.add_argument("--scopes", nargs="*", default=list(DEFAULT_SCOPES))
    parser.add_argument("--strength-scales", nargs="*", default=list(DEFAULT_STRENGTH_SCALES))
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_pairs_file(path: Path) -> List[Tuple[int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"pairs file not found: {path}")

    pairs: List[Tuple[int, int]] = []
    seen = set()
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parsed = None
        for pattern in PAIR_PATTERNS:
            match = pattern.search(line)
            if match:
                parsed = (int(match.group(1)), int(match.group(2)))
                break

        if parsed is None:
            raise ValueError(f"Could not parse line {lineno} in {path}: {raw_line!r}")

        if parsed not in seen:
            seen.add(parsed)
            pairs.append(parsed)

    if not pairs:
        raise ValueError(f"No valid (layer_id, feature_id) pairs found in {path}")
    return pairs


def parse_list_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"list file not found: {path}")

    items: List[str] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[,\s]+", line):
            text = token.strip()
            if text:
                items.append(text)
    if not items:
        raise ValueError(f"No valid values found in {path}")
    return items


def normalize_scopes(scopes: Sequence[str]) -> List[str]:
    allowed = set(DEFAULT_SCOPES)
    normalized: List[str] = []
    for scope in scopes:
        s = str(scope).strip()
        if not s:
            continue
        if s not in allowed:
            raise ValueError(
                f"Unsupported scope: {s}. Allowed scopes: {', '.join(DEFAULT_SCOPES)}"
            )
        normalized.append(s)
    if not normalized:
        raise ValueError("No valid scopes provided.")
    return normalized


def normalize_strength_scales(scales: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for item in scales:
        text = str(item).strip()
        if not text:
            continue
        for piece in text.split(","):
            token = piece.strip()
            if token:
                normalized.append(token)
    if not normalized:
        raise ValueError("No valid strength scales provided.")
    return normalized


def normalize_steps(steps: Sequence[object]) -> List[int]:
    normalized: List[int] = []
    for item in steps:
        text = str(item).strip()
        if not text:
            continue
        for piece in text.split(","):
            token = piece.strip()
            if not token:
                continue
            try:
                normalized.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Invalid step value: {token!r}") from exc
    normalized = list(dict.fromkeys(normalized))
    if not normalized:
        raise ValueError("No valid steps provided.")
    return normalized


def resolve_average_l0(layer_id: int, width: str, canonical_map_path: Path) -> str:
    value = extract_average_l0_from_canonical_map(
        canonical_map_path=canonical_map_path,
        layer_id=str(layer_id),
        width=str(width),
    )
    if value is None:
        raise ValueError(
            f"Could not resolve average_l0 for layer={layer_id}, width={width} "
            f"from canonical map: {canonical_map_path}"
        )
    return value


def build_local_sae_path(sae_root: Path, layer_id: int, width: str, average_l0: str) -> Path:
    return sae_root / f"layer_{layer_id}" / f"width_{width}" / f"average_l0_{average_l0}"


def build_command(
    *,
    python_exe: str,
    script_path: Path,
    layer_id: int,
    feature_id: int,
    llm_name: str,
    sae_path: Path,
    output_root: Path,
    scope: str,
    step: int,
    device: str,
    top_k_examples: int,
    strength_scales: Sequence[str],
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [
        python_exe,
        str(script_path),
        "--layer-id",
        str(layer_id),
        "--feature-id",
        str(feature_id),
        "--llm-name",
        str(llm_name),
        "--sae-path",
        str(sae_path),
        "--output-root",
        str(output_root),
        "--intervention-scope",
        str(scope),
        "--intervention-steps",
        str(step),
        "--device",
        str(device),
        "--top-k-examples",
        str(top_k_examples),
        "--strength-scales",
        *[str(x) for x in strength_scales],
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def iter_jobs(args: argparse.Namespace, run_output_root: Path) -> Iterable[Tuple[str, List[str]]]:
    pairs = parse_pairs_file(args.pairs_file)
    scopes = normalize_scopes(args.scopes)
    if args.steps_file is not None:
        steps = normalize_steps(parse_list_file(args.steps_file))
    else:
        steps = normalize_steps(args.steps)
    if args.strength_scales_file is not None:
        strength_scales = normalize_strength_scales(parse_list_file(args.strength_scales_file))
    else:
        strength_scales = normalize_strength_scales(args.strength_scales)

    average_l0_cache = {}
    for layer_id, feature_id in pairs:
        if layer_id not in average_l0_cache:
            average_l0_cache[layer_id] = resolve_average_l0(
                layer_id=layer_id,
                width=args.width,
                canonical_map_path=args.canonical_map_path,
            )
        average_l0 = average_l0_cache[layer_id]
        sae_path = build_local_sae_path(args.sae_root, layer_id, args.width, average_l0)

        for step in steps:
            for scope in scopes:
                job_name = (
                    f"layer={layer_id} feature={feature_id} "
                    f"step={step} scope={scope} average_l0={average_l0} "
                    f"run_output_root={run_output_root}"
                )
                yield job_name, build_command(
                    python_exe=args.python_exe,
                    script_path=args.script_path,
                    layer_id=layer_id,
                    feature_id=feature_id,
                    llm_name=args.llm_name,
                    sae_path=sae_path,
                    output_root=run_output_root,
                    scope=scope,
                    step=step,
                    device=args.device,
                    top_k_examples=args.top_k_examples,
                    strength_scales=strength_scales,
                    extra_args=args.extra_args,
                )


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id).strip() if args.run_id is not None else ""
    if not run_id:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_output_root = args.output_root / run_id
    jobs = list(iter_jobs(args, run_output_root=run_output_root))

    print(f"Loaded {len(jobs)} commands from {args.pairs_file}")
    print(f"Run output root: {run_output_root}")
    if args.dry_run:
        for idx, (job_name, cmd) in enumerate(jobs, start=1):
            print(f"[{idx}/{len(jobs)}] {job_name}")
            print(shlex.join(cmd))
        return 0

    failures = 0
    for job_name, cmd in tqdm(jobs, desc="Running commands"):
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures += 1
            print(f"Command failed ({result.returncode}): {job_name}", file=sys.stderr)
            print(shlex.join(cmd), file=sys.stderr)
            if args.fail_fast:
                return result.returncode

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
