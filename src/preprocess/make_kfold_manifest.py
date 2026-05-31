"""
python src/preprocess/make_kfold_manifest.py \
  --data-root /root/autodl-tmp/datasets/aipc \
  --n-folds 5 \
  --out-dir /root/autodl-tmp/datasets/aipc/eval_folds

Build a non-destructive K-fold file manifest for AIPC PSM rescoring.

Why this exists:
- The old split_train_valid.py physically moves parquet files into train/valid.
- For trustworthy offline validation, we want reproducible file-level folds without
  moving data again.
- Each fold contains mzML/TIMS/WIFF files and is balanced by row count within each
  instrument as much as possible.

Example:
python src/preprocess/make_kfold_manifest.py \
  --data-root /root/autodl-tmp/datasets/aipc \
  --n-folds 5 \
  --seed 20260529 \
  --out-dir /root/autodl-tmp/datasets/aipc/eval_folds
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import polars as pl
from tqdm import tqdm


INSTRUMENTS = ["mzml", "tims", "wiff"]

SOURCE_DIR_NAMES = {
    "mzml": "mzml_merged",
    "tims": "tims_merged",
    "wiff": "wiff_merged",
}

REQUIRED_BASE_COLS = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "precursor_sequence",
    "peptide_key",
    "precursor_mz",
    "rt",
    "predicted_rt",
    "delta_rt",
    "charge",
    "ion_mobility",
    "has_ion_mobility",
    "label",
]

REQUIRED_FEATURE_FLAGS = ["aux_feature_done", "fragment_feature_done"]


def stable_hash(text: str, seed: int) -> int:
    key = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16)


def is_candidate_parquet(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".parquet"
        and ".tmp" not in name
        and not name.endswith(".bak")
        and not name.endswith(".bak_fragment")
        and not name.endswith(".tmp_group.parquet")
    )


def discover_from_processed(data_root: Path) -> List[Dict]:
    processed_root = data_root / "processed"
    out = []
    for instrument in INSTRUMENTS:
        d = processed_root / SOURCE_DIR_NAMES[instrument]
        if not d.exists():
            print(f"processed source not found, skip: {d}")
            continue
        files = [p for p in sorted(d.glob("*.parquet")) if is_candidate_parquet(p)]
        print(f"processed/{SOURCE_DIR_NAMES[instrument]}: {len(files)} files")
        out.extend({"path": p, "instrument_hint": instrument, "source_split": "processed"} for p in files)
    return out


def discover_from_processed_split(data_root: Path) -> List[Dict]:
    split_root = data_root / "processed_split"
    out = []
    for source_split in ["train", "valid"]:
        for instrument in INSTRUMENTS:
            d = split_root / source_split / instrument
            if not d.exists():
                print(f"processed_split source not found, skip: {d}")
                continue
            files = [p for p in sorted(d.glob("*.parquet")) if is_candidate_parquet(p)]
            print(f"processed_split/{source_split}/{instrument}: {len(files)} files")
            out.extend({"path": p, "instrument_hint": instrument, "source_split": source_split} for p in files)
    return out


def discover_files(data_root: Path, source: str) -> List[Dict]:
    if source == "processed":
        return discover_from_processed(data_root)
    if source == "processed_split":
        return discover_from_processed_split(data_root)
    if source != "auto":
        raise ValueError(f"unknown source: {source}")

    split_items = discover_from_processed_split(data_root)
    if split_items:
        print("Using processed_split because it contains parquet files.")
        return split_items

    print("processed_split is empty; falling back to processed/*_merged.")
    return discover_from_processed(data_root)


def get_one_file_info(item: Dict, require_group_feature: bool) -> Dict:
    path = Path(item["path"])
    instrument_hint = item["instrument_hint"]

    schema_cols = pl.scan_parquet(path).collect_schema().names()
    schema = set(schema_cols)

    required = list(REQUIRED_BASE_COLS) + list(REQUIRED_FEATURE_FLAGS)
    if require_group_feature:
        required.append("group_feature_done")

    missing = [c for c in required if c not in schema]
    if missing:
        return {
            "path": str(path),
            "file_name": path.name,
            "instrument": instrument_hint,
            "source_split": item.get("source_split", ""),
            "ok": False,
            "skip_reason": f"missing columns/flags: {missing}",
        }

    status = (
        pl.scan_parquet(path)
        .select(
            [
                pl.len().alias("total_rows"),
                pl.col("label").cast(pl.Int64).sum().alias("target_rows"),
                pl.col("instrument").drop_nulls().first().alias("instrument_from_file"),
                pl.col("file_id").drop_nulls().first().alias("file_id"),
            ]
        )
        .collect()
    ).row(0, named=True)

    total_rows = int(status["total_rows"])
    target_rows = int(status["target_rows"])
    decoy_rows = total_rows - target_rows
    instrument = status.get("instrument_from_file") or instrument_hint

    return {
        "path": str(path),
        "file_name": path.name,
        "file_id": str(status.get("file_id") or path.stem),
        "instrument": str(instrument),
        "source_split": item.get("source_split", ""),
        "total_rows": total_rows,
        "target_rows": target_rows,
        "decoy_rows": decoy_rows,
        "target_rate": target_rows / total_rows if total_rows > 0 else 0.0,
        "size_bytes": int(path.stat().st_size),
        "ok": True,
        "skip_reason": "",
    }


def assign_balanced_folds(file_infos: List[Dict], n_folds: int, seed: int) -> List[Dict]:
    """Assign folds independently per instrument, balancing total row count."""
    by_instrument: Dict[str, List[Dict]] = defaultdict(list)
    for info in file_infos:
        by_instrument[info["instrument"]].append(dict(info))

    assigned: List[Dict] = []
    for instrument, items in sorted(by_instrument.items()):
        # Largest files first for row balance; hash is deterministic tie-breaker.
        items = sorted(
            items,
            key=lambda x: (
                -int(x["total_rows"]),
                stable_hash(f"{x['instrument']}|{x['file_name']}|{x['total_rows']}", seed),
            ),
        )

        fold_rows = [0 for _ in range(n_folds)]
        fold_files = [0 for _ in range(n_folds)]

        for x in items:
            # Choose the currently lightest fold for this instrument.
            fold = min(range(n_folds), key=lambda k: (fold_rows[k], fold_files[k], k))
            x["fold"] = int(fold)
            assigned.append(x)
            fold_rows[fold] += int(x["total_rows"])
            fold_files[fold] += 1

        print(f"\n{instrument} fold row balance:")
        for k in range(n_folds):
            print(f"  fold {k}: files={fold_files[k]}, rows={fold_rows[k]}")

    return assigned


def write_outputs(assigned: List[Dict], bad_infos: List[Dict], out_dir: Path, n_folds: int, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pl.DataFrame(assigned).sort(["fold", "instrument", "file_name"])
    manifest_path = out_dir / "fold_manifest.csv"
    manifest_df.write_csv(manifest_path)

    summary_df = (
        manifest_df.group_by(["fold", "instrument"])
        .agg(
            [
                pl.len().alias("num_files"),
                pl.col("total_rows").sum().alias("total_rows"),
                pl.col("target_rows").sum().alias("target_rows"),
                pl.col("decoy_rows").sum().alias("decoy_rows"),
            ]
        )
        .with_columns(
            [
                (pl.col("target_rows") / pl.col("total_rows")).alias("target_rate"),
            ]
        )
        .sort(["fold", "instrument"])
    )
    summary_path = out_dir / "fold_summary.csv"
    summary_df.write_csv(summary_path)

    fold_global = (
        manifest_df.group_by("fold")
        .agg(
            [
                pl.len().alias("num_files"),
                pl.col("total_rows").sum().alias("total_rows"),
                pl.col("target_rows").sum().alias("target_rows"),
                pl.col("decoy_rows").sum().alias("decoy_rows"),
            ]
        )
        .with_columns((pl.col("target_rows") / pl.col("total_rows")).alias("target_rate"))
        .sort("fold")
    )
    fold_global.write_csv(out_dir / "fold_summary_global.csv")

    if bad_infos:
        pl.DataFrame(bad_infos).write_csv(out_dir / "skipped_files.csv")

    meta = {
        "n_folds": n_folds,
        "seed": args.seed,
        "source": args.source,
        "data_root": str(Path(args.data_root)),
        "require_group_feature": bool(args.require_group_feature),
        "num_ok_files": len(assigned),
        "num_skipped_files": len(bad_infos),
    }
    with open(out_dir / "fold_manifest_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nWrote:")
    print(f"  {manifest_path}")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'fold_summary_global.csv'}")
    if bad_infos:
        print(f"  {out_dir / 'skipped_files.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source",
        choices=["auto", "processed", "processed_split"],
        default="auto",
        help="processed_split is preferred when you already ran split_train_valid.py.",
    )
    parser.add_argument(
        "--require-group-feature",
        action="store_true",
        help="Use this when the folds are intended for v2 full validation and every file must already contain group_feature_done.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug only.")
    args = parser.parse_args()

    if args.n_folds < 2:
        raise ValueError("--n-folds must be >= 2")

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "eval_folds"

    print("========== Discover files ==========")
    items = discover_files(data_root, args.source)
    if args.max_files is not None:
        items = items[: args.max_files]
    if not items:
        raise RuntimeError("No parquet files found. Check --data-root and --source.")

    print("\n========== Scan file metadata ==========")
    ok_infos: List[Dict] = []
    bad_infos: List[Dict] = []
    for item in tqdm(items):
        try:
            info = get_one_file_info(item, require_group_feature=args.require_group_feature)
        except Exception as e:
            info = {
                "path": str(item.get("path", "")),
                "instrument": item.get("instrument_hint", ""),
                "source_split": item.get("source_split", ""),
                "ok": False,
                "skip_reason": str(e),
            }
        if info.get("ok"):
            ok_infos.append(info)
        else:
            bad_infos.append(info)

    print(f"ok files: {len(ok_infos)}")
    print(f"skipped files: {len(bad_infos)}")
    if not ok_infos:
        raise RuntimeError("No ready files found for K-fold manifest.")

    print("\n========== Assign folds ==========")
    assigned = assign_balanced_folds(ok_infos, n_folds=args.n_folds, seed=args.seed)

    print("\n========== Write manifest ==========")
    write_outputs(assigned, bad_infos, out_dir, args.n_folds, args)


if __name__ == "__main__":
    main()
