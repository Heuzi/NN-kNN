from pathlib import Path
import json
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.table1_nnknn_kfold import run_table1_kfold

DATASET_NAMES = [
    "califonia_housing",
    "diabets",
    "abalone",
    "body_fat",
    "airfoil",
    "car",
    "student_performance",
    "yacht",
    "energy_efficiency",
    "bike_sharing",
    "wine",
]

BASE_SEED = 43
OUTDIR = Path(__file__).resolve().parent

print("Starting Table 1 5-fold sweep", flush=True)
print("Datasets:", DATASET_NAMES, flush=True)
print("Base seed:", BASE_SEED, flush=True)
summary_df, runs_df, _ = run_table1_kfold(
    dataset_names=DATASET_NAMES,
    num_folds=5,
    base_seed=BASE_SEED,
)
summary_df.to_csv(OUTDIR / "summary_long.csv", index=False)
runs_df.to_csv(OUTDIR / "runs_long.csv", index=False)
pivot_df = summary_df.pivot(index="dataset", columns="entry_label", values="rmse_raw_table").reset_index()
pivot_df.to_csv(OUTDIR / "table1_like.csv", index=False)
(OUTDIR / "done.json").write_text(
    json.dumps(
        {
            "outdir": str(OUTDIR),
            "base_seed": BASE_SEED,
            "summary_rows": int(len(summary_df)),
            "run_rows": int(len(runs_df)),
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("Finished Table 1 5-fold sweep", flush=True)
