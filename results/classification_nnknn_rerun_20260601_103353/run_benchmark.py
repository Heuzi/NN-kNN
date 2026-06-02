import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from model.classification_workflow import (
    make_classification_cfg,
    run_repeated_classification_model_benchmarks,
)

run_dir = Path(os.environ["NNKNN_RERUN_DIR"])
manifest_path = run_dir / "manifest.json"
manifest = {
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "datasets": ["iris", "zebra", "zebra_special"],
    "methods": ["nnknn"],
    "mode": "kfold",
    "num_runs": 10,
    "base_seed": 42,
    "config": {
        "task_type": "classification",
        "training_epochs": 50,
        "batch_size": 32,
        "patience": 20,
        "tau": 0.5,
        "case_normalizer": "softmax",
        "normalize_over_cases": True,
        "case_score_mode": "bias_minus_distance",
    },
}

def write_manifest():
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

write_manifest()
try:
    cfg = make_classification_cfg({
        "training_epochs": 50,
        "batch_size": 32,
        "patience": 20,
        "tau": 0.5,
        "case_normalizer": "softmax",
        "normalize_over_cases": True,
        "case_score_mode": "bias_minus_distance",
        "checkpoint_path": str(run_dir / "checkpoint.pth"),
    })
    summary, runs, _ = run_repeated_classification_model_benchmarks(
        ["iris", "zebra", "zebra_special"],
        cfg,
        methods=["nnknn"],
        num_runs=10,
        mode="kfold",
        base_seed=42,
    )
    summary.to_csv(run_dir / "summary.csv", index=False)
    runs.to_csv(run_dir / "runs.csv", index=False)
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest()
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(summary.to_string(index=False), flush=True)
except Exception as exc:
    manifest["status"] = "failed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = repr(exc)
    write_manifest()
    traceback.print_exc()
    raise
