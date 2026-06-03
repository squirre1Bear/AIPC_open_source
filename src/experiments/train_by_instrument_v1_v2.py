# python src/experiments/train_by_instrument_v1_v2.py \
#     --data-root /root/autodl-tmp/datasets/aipc \
#     --out-root ~/aipc/models/exp_by_instrument \
#     --steps v1 group-oof v2 \
#     --num-threads 20 \
#     --extra-v1-args "--skip-existing --fdr-min-delta 20" \
#     --early-stopping-rounds 100 \
#     --v1-num-boost-round 2000
#     --extra-v2-args "--fdr-min-delta 20"
#     --v2-num-boost-round 2000

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


INSTRUMENTS = ["mzml", "tims", "wiff"]
DEFAULT_CPU_THREADS = 12

V1_ROWS = {
    "mzml": {"train_max_rows": 10_000_000, "valid_max_rows": 2_000_000, "max_rows_per_file": 150_000},
    "tims": {"train_max_rows": 8_000_000, "valid_max_rows": 1_500_000, "max_rows_per_file": 150_000},
    "wiff": {"train_max_rows": 12_000_000, "valid_max_rows": 2_000_000, "max_rows_per_file": 150_000},
}

V2_ROWS = {
    "mzml": {"train_max_rows": 6_000_000, "valid_max_rows": 1_500_000, "max_rows_per_file": 120_000},
    "tims": {"train_max_rows": 5_000_000, "valid_max_rows": 1_200_000, "max_rows_per_file": 120_000},
    "wiff": {"train_max_rows": 8_000_000, "valid_max_rows": 1_500_000, "max_rows_per_file": 120_000},
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def script_path(relative_path: str) -> Path:
    return repo_root() / relative_path


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def out_root(args: argparse.Namespace) -> Path:
    return resolve_repo_path(args.out_root)


def pred_out_path(args: argparse.Namespace) -> Path:
    return resolve_repo_path(args.pred_out_path)


def append_common_flags(cmd: list[str], flags: list[str]) -> list[str]:
    if isinstance(flags, str):
        flags = shlex.split(flags)
    if flags:
        cmd.extend(flags)
    return cmd


def run_command(cmd: list[str], dry_run: bool, cpu_threads: int = DEFAULT_CPU_THREADS) -> None:
    print("\n" + " ".join(cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cpu_threads = env.get("AIPC_CPU_THREADS", str(max(1, int(cpu_threads))))
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TBB_NUM_THREADS",
    ]:
        env[key] = cpu_threads
    env.setdefault("POLARS_MAX_THREADS", cpu_threads)
    subprocess.run(cmd, check=True, env=env)


def build_v1_command(args: argparse.Namespace, instrument: str) -> list[str]:
    rows = V1_ROWS[instrument]
    return append_common_flags(
        [
            sys.executable,
            str(script_path("src/model/train_lightGBM_v1_folds_withoutRT.py")),
            "--data-root",
            args.data_root,
            "--out-dir",
            str(out_root(args) / "v1" / instrument),
            "--only-instrument",
            instrument,
            "--n-folds",
            str(args.n_folds),
            "--balance-by-rows",
            "--reuse-folds",
            "--train-max-rows",
            str(rows["train_max_rows"]),
            "--valid-max-rows",
            str(rows["valid_max_rows"]),
            "--max-rows-per-file",
            str(rows["max_rows_per_file"]),
            "--num-boost-round",
            str(args.v1_num_boost_round),
            "--early-stopping-rounds",
            str(args.early_stopping_rounds),
            "--semi-target-fdr",
            str(args.semi_target_fdr),
            "--semi-neg-pos-ratio",
            str(args.semi_neg_pos_ratio),
            "--semi-hard-decoy-frac",
            str(args.semi_hard_decoy_frac),
            "--semi-high-score-decoy-frac",
            str(args.semi_high_score_decoy_frac),
            "--num-threads",
            str(args.num_threads),
        ],
        args.extra_v1_args,
    )


def build_group_oof_command(args: argparse.Namespace, instrument: str) -> list[str]:
    return [
        sys.executable,
        str(script_path("src/preprocess/add_group_feature_oof.py")),
        "--data-root",
        args.data_root,
        "--fold-model-dir",
        str(out_root(args) / "v1" / instrument),
        "--splits",
        "train",
        "valid",
        "--workers",
        str(args.group_workers),
        "--only-instrument",
        instrument,
        "--mask-rt-qvalue-anomaly",
        "--force",
    ]


def build_group_test_command(args: argparse.Namespace, instrument: str) -> list[str]:
    return [
        sys.executable,
        str(script_path("src/preprocess/add_group_feature_test.py")),
        "--parquet-dir",
        args.test_parquet_dir,
        "--fold-model-dir",
        str(out_root(args) / "v1" / instrument),
        "--only-instrument",
        instrument,
        "--mask-rt-qvalue-anomaly",
        "--force",
    ]


def build_v2_command(args: argparse.Namespace, instrument: str) -> list[str]:
    rows = V2_ROWS[instrument]
    return append_common_flags(
        [
            sys.executable,
            str(script_path("src/model/train_lightGBM_v2_groupaware_withoutRT.py")),
            "--data-root",
            args.data_root,
            "--v1-model-dir",
            str(out_root(args) / "v1" / instrument),
            "--out-dir",
            str(out_root(args) / "v2" / instrument),
            "--only-instrument",
            instrument,
            "--mode",
            "rank",
            "--rank-objective",
            args.rank_objective,
            "--label-gain",
            args.label_gain,
            "--eval-at",
            args.eval_at,
            "--train-max-rows",
            str(rows["train_max_rows"]),
            "--valid-max-rows",
            str(rows["valid_max_rows"]),
            "--max-rows-per-file",
            str(rows["max_rows_per_file"]),
            "--num-boost-round",
            str(args.v2_num_boost_round),
            "--early-stopping-rounds",
            str(args.early_stopping_rounds),
            "--num-threads",
            str(args.num_threads),
        ],
        args.extra_v2_args,
    )


def build_predict_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(script_path("src/submit/predict.py")),
        str(out_root(args) / "v2"),
        "--parquet_dir",
        args.test_parquet_dir,
        "--out_path",
        str(pred_out_path(args)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental per-instrument v1 -> group features -> v2 Ranker pipeline."
    )
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")
    parser.add_argument("--out-root", type=str, default="~/aipc/models/exp_by_instrument")
    parser.add_argument(
        "--test-parquet-dir",
        type=str,
        default="/root/autodl-tmp/datasets/aipc/processed/bas_merged",
    )
    parser.add_argument("--pred-out-path", type=str, default="~/aipc/submissions/exp_by_instrument")
    parser.add_argument("--steps", nargs="+", choices=["v1", "group-oof", "v2", "group-test", "predict"], default=["v1", "group-oof", "v2"])
    parser.add_argument("--instruments", nargs="+", choices=INSTRUMENTS, default=INSTRUMENTS)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--semi-target-fdr", type=float, default=0.01)
    parser.add_argument("--semi-neg-pos-ratio", type=float, default=2.0)
    parser.add_argument("--semi-hard-decoy-frac", type=float, default=0.5)
    parser.add_argument("--semi-high-score-decoy-frac", type=float, default=0.3)
    parser.add_argument("--rank-objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank")
    parser.add_argument("--label-gain", type=str, default="0,1,4")
    parser.add_argument("--eval-at", type=str, default="1,3,5,10")
    parser.add_argument("--v1-num-boost-round", type=int, default=3000)
    parser.add_argument("--v2-num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--group-workers", type=int, default=12)
    parser.add_argument("--extra-v1-args", type=str, default="")
    parser.add_argument("--extra-v2-args", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_out_root = out_root(args)
    if not args.dry_run:
        (resolved_out_root / "v1").mkdir(parents=True, exist_ok=True)
        (resolved_out_root / "v2").mkdir(parents=True, exist_ok=True)

    print("========== per-instrument experiment ==========")
    print("data_root:", args.data_root)
    print("out_root:", resolved_out_root)
    print("steps:", args.steps)
    print("instruments:", args.instruments)
    print("dry_run:", args.dry_run)

    command_builders = {
        "v1": build_v1_command,
        "group-oof": build_group_oof_command,
        "v2": build_v2_command,
        "group-test": build_group_test_command,
    }
    for step in args.steps:
        if step == "predict":
            run_command(build_predict_command(args), dry_run=args.dry_run, cpu_threads=args.num_threads)
            continue
        for instrument in args.instruments:
            run_command(
                command_builders[step](args, instrument),
                dry_run=args.dry_run,
                cpu_threads=args.num_threads,
            )


if __name__ == "__main__":
    main()
