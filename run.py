# delayed_run.py
import os
import time
import subprocess
from datetime import datetime, timedelta

DELAY_SECONDS = 0

def main():
    run_time = datetime.now() + timedelta(seconds=DELAY_SECONDS)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] start sleeping.")
    print(f"Will run at: {run_time:%Y-%m-%d %H:%M:%S}", flush=True)

    time.sleep(DELAY_SECONDS)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"

    cmd = [
        "python", "run_steer_batch.py",
        "--device", "cuda",
        "--pairs-file", "pairs.txt",
        "--llm-name", "/data/MODEL/Meta-Llama-3.1-8B",
        "--sae-path", "/data/MODEL/OpenSAE-LLaMA-3.1-Layer_20",
        "--scopes",
        "last_original_token_only",
        "last_token_only",
        "natural_support_mask",
        "--steps-file", "steps.txt",
        "--strength-scales-file", "scales.txt",
    ]

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] running command:", flush=True)
    print(" ".join(cmd), flush=True)

    result = subprocess.run(cmd, env=env)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] finished with return code: {result.returncode}", flush=True)

if __name__ == "__main__":
    main()