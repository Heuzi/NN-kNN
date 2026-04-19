from __future__ import annotations

import copy
from typing import Any, Mapping

import pandas as pd
import torch
from sklearn.model_selection import KFold

from model.regression_workflow import (
    configure_regression_cfg_for_state,
    evaluate_nnknn_pre_post_adaptation_state,
    load_regression_dataset_state,
    make_regression_cfg,
    run_repeated_regression_model_benchmarks,
    split_regression_state,
    standardize_regression_state,
    train_nnknn_regression_state,
)


def build_table1_paper_base_cfg(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared NN-kNN Table 1 config described in the paper.

    Notes:
    - This is for the tabular Table 1 setting, not the UTKFace setup.
    - Parameters explicitly stated in the paper are set here.
    - Parameters not stated in the paper keep the repository defaults.
    """
    cfg = make_regression_cfg(
        {
            "task_type": "regression",
            "case_score_mode": "bias_minus_distance",
            "softmax_over_cases": True,
            "tau": 1.0,
            "regression_locality": False,
            "lambda_base": 1.0,
            "locality_alpha": 2.0,
            "lambda_expdist": 0.1,
            "eps_sigma_multiplier": 0.1,
            "training_epochs": 7000,
            "batch_size": 64,
            "feature_extractor_lr": 1e-3,
            "glocal_weightor_lr": 1e-3,
            "case_net_lr": 3e-4,
            "checkpoint_path": "nnknn_regression_best.pth",
            "patience": 80,
            "post_mlp_enabled": False,
            "explanation_mode": True,
        }
    )
    if overrides:
        cfg.update(copy.deepcopy(dict(overrides)))
    return cfg


def make_table1_family_cfg(
    *,
    case_normalizer: str,
    use_locality: bool,
    base_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one of the four Table 1 NN-kNN configuration families.

    Each family is trained with NN-CDH enabled so the resulting workflow state
    can report both:
    - pre-adaptation RMSE (`pure` / retrieval-only)
    - post-adaptation RMSE (`adaptation`)

    This matches the paper's family structure:
    softmax, softmax+locality, sparsemax, sparsemax+locality.
    """
    cfg = build_table1_paper_base_cfg(base_cfg)
    cfg.update(
        {
            "case_normalizer": case_normalizer,
            "regression_locality": bool(use_locality),
            "use_nn_cdh": True,
            "nn_cdh_pretrain": True,
            "cdh_aggregate": True,
        }
    )
    return cfg


TABLE1_NNKNN_FAMILIES: list[dict[str, Any]] = [
    {
        "family_id": "softmax",
        "family_label": "NN-kNN (softmax)",
        "case_normalizer": "softmax",
        "use_locality": False,
    },
    {
        "family_id": "softmax_locality",
        "family_label": "NN-kNN (softmax) + locality",
        "case_normalizer": "softmax",
        "use_locality": True,
    },
    {
        "family_id": "sparsemax",
        "family_label": "NN-kNN (sparsemax)",
        "case_normalizer": "sparsemax",
        "use_locality": False,
    },
    {
        "family_id": "sparsemax_locality",
        "family_label": "NN-kNN (sparsemax) + locality",
        "case_normalizer": "sparsemax",
        "use_locality": True,
    },
]

TABLE1_BASELINE_METHODS: list[str] = ["knn_x", "mlkr_knn", "mlp"]


def _clone_state_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
        for key, value in bundle.items()
    }


def _paper_result_name(*, use_locality: bool, adapted: bool) -> str:
    if use_locality:
        return "locality + adaptation" if adapted else "locality"
    return "adaptation" if adapted else "pure"


def build_table1_baseline_method_cfgs(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the baseline configs used by the notebook's repeated benchmark cell."""
    cfgs: dict[str, dict[str, Any]] = {
        "knn_x": {"k": 5, "batch_size": 512, "device": torch.device("cpu")},
        "mlkr_knn": {"k": 25, "n_components": None, "max_iter": 1000},
        "mlp": {
            "hidden": (64, 32),
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "val_batch_size": 512,
            "epochs": 200,
            "patience": 20,
            "grad_clip_norm": 1.0,
            "device": torch.device("cpu"),
        },
    }
    for method_name, method_overrides in (overrides or {}).items():
        cfgs.setdefault(method_name, {})
        cfgs[method_name].update(copy.deepcopy(dict(method_overrides)))
    return cfgs


def _normalize_table1_nnknn_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    normalized = summary_df.copy()
    normalized["source"] = "nnknn"
    normalized["entry_id"] = normalized["family_id"] + ":" + normalized["result_type"]
    normalized["entry_label"] = normalized["family_label"] + " - " + normalized["result_label"]
    return normalized


def _normalize_table1_nnknn_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    normalized = runs_df.copy()
    normalized["source"] = "nnknn"
    normalized["entry_id"] = normalized["family_id"] + ":" + normalized["result_type"]
    normalized["entry_label"] = normalized["family_label"] + " - " + normalized["result_label"]
    return normalized


def _normalize_table1_baseline_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in summary_df.to_dict("records"):
        method = row["method"]
        method_label = row.get("method_label") or method
        records.append(
            {
                "dataset": row["dataset"],
                "family_id": method,
                "family_label": method_label,
                "result_type": "baseline",
                "result_label": method_label,
                "num_folds": row["num_runs"],
                "rmse_raw_mean": row.get("rmse_raw_mean"),
                "rmse_raw_std": row.get("rmse_raw_std"),
                "rmse_raw_table": row.get("rmse_raw_table"),
                "mode": row.get("mode"),
                "source": "baseline",
                "entry_id": method,
                "entry_label": method_label,
            }
        )
    return pd.DataFrame(records)


def _normalize_table1_baseline_runs(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for row in runs_df.to_dict("records"):
        method = row["method"]
        method_label = row.get("method_label") or method
        records.append(
            {
                "dataset": row["dataset"],
                "mode": row["mode"],
                "fold": row["fold"],
                "run_index": row["run_index"],
                "family_id": method,
                "family_label": method_label,
                "result_type": "baseline",
                "result_label": method_label,
                "rmse_raw": row.get("rmse_raw"),
                "rmse_model_space": row.get("rmse_model_space"),
                "standardized_targets": row.get("standardized_targets"),
                "source": "baseline",
                "entry_id": method,
                "entry_label": method_label,
            }
        )
    return pd.DataFrame(records)


def summarize_table1_nnknn_runs(
    runs_df: pd.DataFrame,
    *,
    digits: int = 4,
) -> pd.DataFrame:
    """Aggregate fold-level Table 1 NN-kNN results into paper-style summaries."""
    if runs_df.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    grouped = runs_df.groupby(
        ["dataset", "family_id", "family_label", "result_type", "result_label"],
        dropna=False,
        sort=False,
    )

    for (dataset, family_id, family_label, result_type, result_label), group in grouped:
        rmse_values = group["rmse_raw"].dropna()
        if rmse_values.empty:
            continue

        rmse_mean = float(rmse_values.mean())
        rmse_std = float(rmse_values.std(ddof=1)) if len(rmse_values) > 1 else 0.0
        records.append(
            {
                "dataset": dataset,
                "family_id": family_id,
                "family_label": family_label,
                "result_type": result_type,
                "result_label": result_label,
                "num_folds": int(len(group)),
                "rmse_raw_mean": rmse_mean,
                "rmse_raw_std": rmse_std,
                "rmse_raw_table": f"{rmse_mean:.{digits}f} +/- {rmse_std:.{digits}f}",
            }
        )

    return pd.DataFrame(records)


def run_table1_nnknn_kfold(
    dataset_names: str | list[str],
    *,
    num_folds: int = 5,
    base_seed: int = 42,
    base_cfg: Mapping[str, Any] | None = None,
    feature_extractor: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Run the Table 1 NN-kNN families with k-fold CV.

    Output structure:
    - `summary_df`: one row per dataset/family/result_type
    - `runs_df`: one row per dataset/family/fold/result_type
    - `artifacts[dataset][family_id]`: trained workflow states per fold
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    runs: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, list[dict[str, Any]]]] = {}
    total_datasets = len(dataset_names)

    for dataset_idx, dataset_name in enumerate(dataset_names, start=1):
        print(
            f"[table1][nnknn] Dataset {dataset_idx}/{total_datasets}: {dataset_name}",
            flush=True,
        )
        dataset_state = load_regression_dataset_state(dataset_name)
        artifacts[dataset_state["display_name"]] = {
            family["family_id"]: [] for family in TABLE1_NNKNN_FAMILIES
        }

        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=base_seed)
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(dataset_state["X"]), start=1):
            print(
                f"[table1][nnknn] {dataset_state['display_name']} fold {fold_idx}/{num_folds}",
                flush=True,
            )
            split_state = split_regression_state(
                _clone_state_bundle(dataset_state),
                seed=base_seed,
                train_idx=train_idx,
                val_idx=val_idx,
            )
            split_state = standardize_regression_state(split_state, enabled=True)
            split_state.update(
                {
                    "dataset": split_state["display_name"],
                    "mode": "kfold",
                    "run_seed": base_seed + fold_idx - 1,
                    "dataset_seed": base_seed,
                    "split_seed": base_seed,
                    "run_index": fold_idx,
                    "fold": fold_idx,
                }
            )

            for family in TABLE1_NNKNN_FAMILIES:
                print(
                    f"[table1][nnknn] {dataset_state['display_name']} fold {fold_idx}/{num_folds} family={family['family_id']}",
                    flush=True,
                )
                family_cfg = make_table1_family_cfg(
                    case_normalizer=family["case_normalizer"],
                    use_locality=family["use_locality"],
                    base_cfg=base_cfg,
                )
                cfg_run = configure_regression_cfg_for_state(family_cfg, split_state)
                workflow_state = train_nnknn_regression_state(
                    split_state,
                    cfg_run,
                    feature_extractor=feature_extractor,
                    checkpoint_label=f"{family['family_id']}_fold_{fold_idx}",
                )
                workflow_state = evaluate_nnknn_pre_post_adaptation_state(
                    workflow_state,
                    batch_size=512,
                    show_plots=False,
                    print_metrics=False,
                )
                workflow_state.update(
                    {
                        "family_id": family["family_id"],
                        "family_label": family["family_label"],
                    }
                )
                artifacts[dataset_state["display_name"]][family["family_id"]].append(workflow_state)

                runs.append(
                    {
                        "dataset": dataset_state["display_name"],
                        "mode": "kfold",
                        "fold": fold_idx,
                        "run_index": fold_idx,
                        "family_id": family["family_id"],
                        "family_label": family["family_label"],
                        "result_type": _paper_result_name(
                            use_locality=family["use_locality"],
                            adapted=False,
                        ),
                        "result_label": _paper_result_name(
                            use_locality=family["use_locality"],
                            adapted=False,
                        ),
                        "rmse_raw": workflow_state["rmse_pre_raw"],
                        "rmse_model_space": workflow_state["rmse_pre_model_space"],
                        "standardized_targets": workflow_state["standardized_targets"],
                    }
                )
                runs.append(
                    {
                        "dataset": dataset_state["display_name"],
                        "mode": "kfold",
                        "fold": fold_idx,
                        "run_index": fold_idx,
                        "family_id": family["family_id"],
                        "family_label": family["family_label"],
                        "result_type": _paper_result_name(
                            use_locality=family["use_locality"],
                            adapted=True,
                        ),
                        "result_label": _paper_result_name(
                            use_locality=family["use_locality"],
                            adapted=True,
                        ),
                        "rmse_raw": workflow_state["rmse_post_raw"],
                        "rmse_model_space": workflow_state["rmse_post_model_space"],
                        "standardized_targets": workflow_state["standardized_targets"],
                    }
                )

    runs_df = pd.DataFrame(runs)
    summary_df = summarize_table1_nnknn_runs(runs_df)
    return summary_df, runs_df, artifacts


def run_table1_kfold(
    dataset_names: str | list[str],
    *,
    num_folds: int = 5,
    base_seed: int = 42,
    nnknn_base_cfg: Mapping[str, Any] | None = None,
    baseline_method_cfgs: Mapping[str, Mapping[str, Any]] | None = None,
    feature_extractor: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the full Table 1 suite for k-fold CV, excluding UTKFace if omitted upstream.

    Returns:
    - `summary_df`: normalized combined summary for NN-kNN families and baselines.
    - `runs_df`: normalized fold-level results for NN-kNN families and baselines.
    - `artifacts`: nested dict containing raw NN-kNN and baseline artifacts.
    """
    print("[table1] Starting NN-kNN family sweep", flush=True)
    nnknn_summary_df, nnknn_runs_df, nnknn_artifacts = run_table1_nnknn_kfold(
        dataset_names,
        num_folds=num_folds,
        base_seed=base_seed,
        base_cfg=nnknn_base_cfg,
        feature_extractor=feature_extractor,
    )

    print("[table1] Starting baseline sweep", flush=True)
    baseline_summary_df, baseline_runs_df, baseline_artifacts = run_repeated_regression_model_benchmarks(
        dataset_names=dataset_names,
        nnknn_cfg=None,
        feature_extractor=feature_extractor,
        methods=TABLE1_BASELINE_METHODS,
        method_cfgs=build_table1_baseline_method_cfgs(baseline_method_cfgs),
        num_runs=num_folds,
        mode="kfold",
        base_seed=base_seed,
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

    print("[table1] Finished full Table 1 sweep", flush=True)
    return summary_df, runs_df, {"baselines": baseline_artifacts, "nnknn": nnknn_artifacts}
