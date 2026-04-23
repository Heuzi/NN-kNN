from __future__ import annotations

from datetime import datetime
from pathlib import Path
import copy
import json
import sys
from typing import Any, Mapping

import pandas as pd
import torch
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.nn_cdh import NNCDHAdapter
from model.nnknn_model import GlocalFeatureWeight, NN_KNN_Model, device
from model.regression_workflow import (
    _checkpoint_path_for_run,
    configure_regression_cfg_for_state,
    evaluate_nnknn_pre_post_adaptation_state,
    load_regression_dataset_state,
    run_repeated_regression_model_benchmarks,
    split_regression_state,
    standardize_regression_state,
)
from tools.table1_nnknn_kfold import (
    TABLE1_BASELINE_METHODS,
    TABLE1_NNKNN_FAMILIES,
    _normalize_table1_baseline_runs,
    _normalize_table1_baseline_summary,
    _normalize_table1_nnknn_runs,
    _normalize_table1_nnknn_summary,
    _paper_result_name,
    build_table1_baseline_method_cfgs,
    make_table1_family_cfg,
    run_table1_nnknn_kfold,
    summarize_table1_nnknn_runs,
)


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
NUM_FOLDS = 5
OUTDIR = Path(__file__).resolve().parent

# Only reuse checkpoints from this failed rerun, not older seed-42 files with the same names.
FAILED_RUN_STARTED_AT = datetime(2026, 4, 21, 11, 48, 0)


def _clone_state_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in bundle.items()
    }


def _checkpoint_for_state(
    split_state: Mapping[str, Any],
    cfg_run: Mapping[str, Any],
    family_id: str,
    fold_idx: int,
) -> Path:
    return Path(
        _checkpoint_path_for_run(
            str(cfg_run.get("checkpoint_path", "nnknn_regression_best.pth")),
            dataset_name=str(split_state.get("display_name", "dataset")),
            run_label=f"{family_id}_fold_{fold_idx}",
        )
    )


def _checkpoint_is_from_failed_run(path: Path) -> bool:
    return path.exists() and datetime.fromtimestamp(path.stat().st_mtime) >= FAILED_RUN_STARTED_AT


def _dataset_has_complete_checkpoints(dataset_name: str) -> bool:
    dataset_state = load_regression_dataset_state(dataset_name)
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=BASE_SEED)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dataset_state["X"]), start=1):
        split_state = split_regression_state(
            _clone_state_bundle(dataset_state),
            seed=BASE_SEED,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        split_state = standardize_regression_state(split_state, enabled=True)
        for family in TABLE1_NNKNN_FAMILIES:
            family_cfg = make_table1_family_cfg(
                case_normalizer=family["case_normalizer"],
                use_locality=family["use_locality"],
            )
            cfg_run = configure_regression_cfg_for_state(family_cfg, split_state)
            if not _checkpoint_is_from_failed_run(
                _checkpoint_for_state(split_state, cfg_run, family["family_id"], fold_idx)
            ):
                return False
    return True


def _load_nnknn_model_from_checkpoint(
    split_state: Mapping[str, Any],
    cfg_run: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    feature_extractor = None
    feature_dim = split_state["X_train"].shape[-1]
    glocal_weightor = GlocalFeatureWeight(feature_dim, cfg_run["glocal_fw_set_num"]).to(device)
    adapter = NNCDHAdapter(feature_dim, label_dim=1).to(device) if cfg_run.get("use_nn_cdh", False) else None
    labels = split_state["y_train_norm"].float().unsqueeze(1)

    model = NN_KNN_Model(
        split_state["X_train"],
        labels,
        feature_extractor=feature_extractor,
        glocal_weightor=glocal_weightor,
        nn_cdh=adapter,
        **dict(cfg_run),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    workflow_state = dict(split_state)
    workflow_state.update(
        {
            "cfg": copy.deepcopy(dict(cfg_run)),
            "cfg_run": copy.deepcopy(dict(cfg_run)),
            "feature_extractor": feature_extractor,
            "glocal_weightor": glocal_weightor,
            "model": model,
            "resume_checkpoint_path": str(checkpoint_path),
        }
    )
    return workflow_state


def evaluate_checkpointed_nnknn_dataset(dataset_name: str) -> pd.DataFrame:
    print(f"[resume][nnknn] Reusing checkpoints for dataset: {dataset_name}", flush=True)
    dataset_state = load_regression_dataset_state(dataset_name)
    splitter = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=BASE_SEED)
    runs: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dataset_state["X"]), start=1):
        print(f"[resume][nnknn] {dataset_state['display_name']} fold {fold_idx}/{NUM_FOLDS}", flush=True)
        split_state = split_regression_state(
            _clone_state_bundle(dataset_state),
            seed=BASE_SEED,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        split_state = standardize_regression_state(split_state, enabled=True)
        split_state.update(
            {
                "dataset": split_state["display_name"],
                "mode": "kfold",
                "run_seed": BASE_SEED + fold_idx - 1,
                "dataset_seed": BASE_SEED,
                "split_seed": BASE_SEED,
                "run_index": fold_idx,
                "fold": fold_idx,
            }
        )

        for family in TABLE1_NNKNN_FAMILIES:
            print(
                f"[resume][nnknn] {dataset_state['display_name']} fold {fold_idx}/{NUM_FOLDS} "
                f"family={family['family_id']} checkpoint",
                flush=True,
            )
            family_cfg = make_table1_family_cfg(
                case_normalizer=family["case_normalizer"],
                use_locality=family["use_locality"],
            )
            cfg_run = configure_regression_cfg_for_state(family_cfg, split_state)
            checkpoint_path = _checkpoint_for_state(split_state, cfg_run, family["family_id"], fold_idx)
            workflow_state = _load_nnknn_model_from_checkpoint(split_state, cfg_run, checkpoint_path)
            workflow_state = evaluate_nnknn_pre_post_adaptation_state(
                workflow_state,
                batch_size=512,
                show_plots=False,
                print_metrics=False,
            )

            for adapted in (False, True):
                result_name = _paper_result_name(
                    use_locality=family["use_locality"],
                    adapted=adapted,
                )
                runs.append(
                    {
                        "dataset": dataset_state["display_name"],
                        "mode": "kfold",
                        "fold": fold_idx,
                        "run_index": fold_idx,
                        "family_id": family["family_id"],
                        "family_label": family["family_label"],
                        "result_type": result_name,
                        "result_label": result_name,
                        "rmse_raw": workflow_state["rmse_post_raw"] if adapted else workflow_state["rmse_pre_raw"],
                        "rmse_model_space": (
                            workflow_state["rmse_post_model_space"]
                            if adapted
                            else workflow_state["rmse_pre_model_space"]
                        ),
                        "standardized_targets": workflow_state["standardized_targets"],
                        "resume_source": "checkpoint",
                    }
                )

    return pd.DataFrame(runs)


def main() -> None:
    print("Starting resumed Table 1 5-fold sweep", flush=True)
    print("Datasets:", DATASET_NAMES, flush=True)
    print("Base seed:", BASE_SEED, flush=True)

    checkpointed_datasets = []
    missing_datasets = []
    for dataset_name in DATASET_NAMES:
        if _dataset_has_complete_checkpoints(dataset_name):
            checkpointed_datasets.append(dataset_name)
        else:
            missing_datasets.append(dataset_name)

    print("Checkpoint-backed NN-kNN datasets:", checkpointed_datasets, flush=True)
    print("NN-kNN datasets to train:", missing_datasets, flush=True)

    checkpointed_runs = [
        evaluate_checkpointed_nnknn_dataset(dataset_name)
        for dataset_name in checkpointed_datasets
    ]

    trained_summary_df = pd.DataFrame()
    trained_runs_df = pd.DataFrame()
    if missing_datasets:
        print("[resume][nnknn] Training missing NN-kNN datasets", flush=True)
        trained_summary_df, trained_runs_df, _ = run_table1_nnknn_kfold(
            missing_datasets,
            num_folds=NUM_FOLDS,
            base_seed=BASE_SEED,
        )
        if not trained_runs_df.empty:
            trained_runs_df = trained_runs_df.copy()
            trained_runs_df["resume_source"] = "trained"

    nnknn_runs_df = pd.concat(
        [*checkpointed_runs, trained_runs_df],
        ignore_index=True,
        sort=False,
    )
    nnknn_summary_df = summarize_table1_nnknn_runs(nnknn_runs_df)

    print("[resume] Starting baseline sweep for all datasets", flush=True)
    baseline_summary_df, baseline_runs_df, _ = run_repeated_regression_model_benchmarks(
        dataset_names=DATASET_NAMES,
        nnknn_cfg=None,
        methods=TABLE1_BASELINE_METHODS,
        method_cfgs=build_table1_baseline_method_cfgs(),
        num_runs=NUM_FOLDS,
        mode="kfold",
        base_seed=BASE_SEED,
        standardize="auto",
    )

    summary_df = pd.concat(
        [
            _normalize_table1_baseline_summary(baseline_summary_df),
            _normalize_table1_nnknn_summary(nnknn_summary_df),
        ],
        ignore_index=True,
        sort=False,
    )
    runs_df = pd.concat(
        [
            _normalize_table1_baseline_runs(baseline_runs_df),
            _normalize_table1_nnknn_runs(nnknn_runs_df),
        ],
        ignore_index=True,
        sort=False,
    )

    summary_df.to_csv(OUTDIR / "summary_long.csv", index=False)
    runs_df.to_csv(OUTDIR / "runs_long.csv", index=False)
    pivot_df = summary_df.pivot(index="dataset", columns="entry_label", values="rmse_raw_table").reset_index()
    pivot_df.to_csv(OUTDIR / "table1_like.csv", index=False)
    (OUTDIR / "resume_manifest.json").write_text(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "base_seed": BASE_SEED,
                "checkpointed_nnknn_datasets": checkpointed_datasets,
                "trained_nnknn_datasets": missing_datasets,
                "summary_rows": int(len(summary_df)),
                "run_rows": int(len(runs_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUTDIR / "done.json").write_text(
        json.dumps(
            {
                "outdir": str(OUTDIR),
                "base_seed": BASE_SEED,
                "summary_rows": int(len(summary_df)),
                "run_rows": int(len(runs_df)),
                "resumed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Finished resumed Table 1 5-fold sweep", flush=True)


if __name__ == "__main__":
    main()
