from __future__ import annotations

from typing import Tuple


# steer_from_neuronpedia.py defaults
STEER_DEFAULT_MODEL_ID = "gemma-2-2b"
STEER_DEFAULT_LLM_NAME = "google/gemma-2-2b"
STEER_DEFAULT_WIDTH = "16k"
STEER_DEFAULT_SAE_RELEASE = "gemma-scope-2b-pt-res"
STEER_DEFAULT_DEVICE = "cpu"
STEER_DEFAULT_TEMPERATURE = 0.0
STEER_DEFAULT_MAX_NEW_TOKENS = 80
STEER_DEFAULT_INTERVENTION_SCOPE = "last_token_only"
STEER_DEFAULT_INTERVENTION_STEPS = 1
STEER_DEFAULT_TOP_K_EXAMPLES = 3
STEER_DEFAULT_MAX_PREFIX_TOKENS = 20
STEER_DEFAULT_STRENGTH_SCALES: Tuple[float, ...] = (0.0, 2.0 / 3.0, 1.5)
STEER_DEFAULT_TIMEOUT = 30
STEER_DEFAULT_OUTPUT_ROOT = "outputs"
STEER_DEFAULT_OUTPUT_FILENAME = "steer_from_neuronpedia.json"
STEER_DEFAULT_LOGIT_ANALYSIS_MAX_STEPS = 3
STEER_DEFAULT_LOGIT_ANALYSIS_TOP_K = 10
STEER_DEFAULT_LOGIT_ANALYSIS_REFERENCE = "clean"


# run_steer_batch.py defaults
BATCH_DEFAULT_STEPS: Tuple[int, ...] = (1, 5, 999)
BATCH_DEFAULT_SCOPES: Tuple[str, ...] = (
    "all_tokens",
    "last_token_only",
    "all_original_tokens",
    "last_original_token_only",
)
BATCH_DEFAULT_STRENGTH_SCALES: Tuple[str, ...] = ("-1", "-3", "1", "-5", "3")
BATCH_DEFAULT_PYTHON_EXE = "python"
BATCH_DEFAULT_SCRIPT_PATH = "steer_from_neuronpedia.py"
BATCH_DEFAULT_OUTPUT_ROOT = "outputs"
BATCH_DEFAULT_LLM_NAME = "/data/MODEL/Gemma-2-2b"
BATCH_DEFAULT_SAE_ROOT = "/data/MODEL/gemma-scope-2b-pt-res"
BATCH_DEFAULT_WIDTH = "16k"
BATCH_DEFAULT_DEVICE = "cpu"
BATCH_DEFAULT_TOP_K_EXAMPLES = 3
