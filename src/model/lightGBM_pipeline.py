# 用于测试跑通流程
from pathlib import Path
import argparse
import hashlib
import joblib

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import lightgbm as lgb
from lightgbm import LGBMClassifier


# ============================================================
# 1. 基础特征列
# ============================================================

FEATURE_COLUMNS = [
    "instrument_id",

    "charge",
    "precursor_mz",
    "rt",
    "predicted_rt",
    "delta_rt",
    "abs_delta_rt",

    "ion_mobility",
    "has_ion_mobility",

    "peptide_length",
    "num_mods",

    "num_peaks",
    "mz_min",
    "mz_max",
    "mz_range",

    "intensity_max",
    "intensity_sum",
    "intensity_mean",
    "intensity_max_fraction",
]


# ============================================================
# 2. 工具函数
# ============================================================

def stable_hash(text: str) -> int:
    """
    用稳定 hash 做训练/验证切分。
    不要用 Python 自带 hash，因为它每次运行可能不一样。
    """
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def infer_instrument_from_path(path: Path) -> str:
    """
    根据文件名推断仪器类型。
    训练 processed 文件一般已经有 instrument 列；
    测试 bas_data 里通常没有 instrument 列，所以需要根据文件名补。
    """
    name = path.name.lower()

    if name.startswith("mzml"):
        return "mzml"
    if name.startswith("tims"):
        return "tims"
    if name.startswith("wiff"):
        return "wiff"

    # 兜底
    if "mzml" in name:
        return "mzml"
    if "tims" in name:
        return "tims"
    if "wiff" in name:
        return "wiff"

    return "unknown"


def collect_processed_files(root: Path):
    """
    收集你已经处理好的三仪器 parquet 文件。
    """
    dirs = [
        root / "processed" / "mzml_merged",
        root / "processed" / "tims_merged",
        root / "processed" / "wiff_merged",
    ]

    files = []

    for d in dirs:
        if not d.exists():
            print(f"警告：目录不存在，跳过：{d}")
            continue

        part_files = sorted(d.glob("*.parquet"))
        print(f"{d} 文件数：{len(part_files)}")
        files.extend(part_files)

    return files


def split_train_valid_files(files, valid_ratio=0.2):
    """
    按文件切分训练集和验证集。
    不要按行随机切分，因为同一个 file_id 里的谱图候选可能高度相关。
    """
    train_files = []
    valid_files = []

    mod = int(round(1 / valid_ratio))

    for p in files:
        h = stable_hash(str(p))
        if h % mod == 0:
            valid_files.append(p)
        else:
            train_files.append(p)

    return train_files, valid_files


# ============================================================
# 3. 特征生成
# ============================================================

def add_missing_columns(df: pl.DataFrame, source_path: Path, is_train: bool) -> pl.DataFrame:
    """
    统一训练集和测试集字段。
    这里不做复杂映射，只补必要字段。
    """

    # 测试集可能叫 delta_rt_model，你说测试时会统一为 delta_rt。
    # 这里也做一层保护：如果只有 delta_rt_model，就改为 delta_rt。
    if "delta_rt" not in df.columns and "delta_rt_model" in df.columns:
        df = df.rename({"delta_rt_model": "delta_rt"})

    # 如果没有 instrument，根据文件名补
    if "instrument" not in df.columns:
        instrument = infer_instrument_from_path(source_path)
        df = df.with_columns(pl.lit(instrument).alias("instrument"))

    # 如果没有 has_ion_mobility，根据 instrument 补
    if "has_ion_mobility" not in df.columns:
        df = df.with_columns(
            (pl.col("instrument") == "tims").cast(pl.Int8).alias("has_ion_mobility")
        )

    # 如果没有 ion_mobility，补 0
    if "ion_mobility" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("ion_mobility"))

    # 如果没有 peptide_key，先用 precursor_sequence
    if "peptide_key" not in df.columns:
        df = df.with_columns(pl.col("precursor_sequence").alias("peptide_key"))

    # 如果没有 group_key，使用 instrument + 文件名 + scan_number
    if "group_key" not in df.columns:
        df = df.with_columns(
            (
                pl.col("instrument")
                + pl.lit("_")
                + pl.lit(source_path.stem)
                + pl.lit("_")
                + pl.col("scan_number").cast(pl.Utf8)
            ).alias("group_key")
        )

    # 训练集必须有 label
    if is_train and "label" not in df.columns:
        raise ValueError(f"训练文件缺少 label 列：{source_path}")

    return df


def make_features(df: pl.DataFrame, source_path: Path, is_train: bool) -> pl.DataFrame:
    """
    从一份 parquet 数据中生成基础模型特征。

    注意：
    这里暂时只做基础特征，不做 b/y 理论离子匹配。
    目标是先跑通训练、验证、预测、提交流程。
    """

    df = add_missing_columns(df, source_path, is_train=is_train)

    # 去掉修饰括号内容后，估计肽段长度。
    # 例如 M[Oxidation]PEPTIDE -> MPEPTIDE
    clean_seq = (
        pl.col("precursor_sequence")
        .cast(pl.Utf8)
        .str.replace_all(r"\[[^\]]*\]", "")
        .str.replace_all(r"\([^\)]*\)", "")
    )

    # 基础字段和谱图统计
    df = df.with_columns([
        # 仪器编码
        pl.when(pl.col("instrument") == "mzml").then(0)
        .when(pl.col("instrument") == "tims").then(1)
        .when(pl.col("instrument") == "wiff").then(2)
        .otherwise(3)
        .cast(pl.Int32)
        .alias("instrument_id"),

        # delta_rt 绝对值
        pl.col("delta_rt").cast(pl.Float64).abs().alias("abs_delta_rt"),

        # 肽段长度：只数标准氨基酸字母
        clean_seq.str.count_matches(r"[ACDEFGHIKLMNPQRSTVWY]").alias("peptide_length"),

        # 修饰数量：简单统计括号数量
        pl.col("precursor_sequence").cast(pl.Utf8).str.count_matches(r"[\[\(]").alias("num_mods"),

        # 谱峰数量
        pl.col("mz_array").list.len().cast(pl.Float64).alias("num_peaks"),

        # m/z 范围
        pl.col("mz_array").list.min().cast(pl.Float64).alias("mz_min"),
        pl.col("mz_array").list.max().cast(pl.Float64).alias("mz_max"),

        # 强度统计
        pl.col("intensity_array").list.max().cast(pl.Float64).alias("intensity_max"),
        pl.col("intensity_array").list.sum().cast(pl.Float64).alias("intensity_sum"),
    ])

    df = df.with_columns([
        (pl.col("mz_max") - pl.col("mz_min")).alias("mz_range"),

        (
            pl.col("intensity_sum") / (pl.col("num_peaks") + 1e-6)
        ).alias("intensity_mean"),

        (
            pl.col("intensity_max") / (pl.col("intensity_sum") + 1e-6)
        ).alias("intensity_max_fraction"),
    ])

    # 训练标签统一成 0/1
    if is_train:
        label_str = pl.col("label").cast(pl.Utf8).str.to_lowercase()

        df = df.with_columns(
            pl.when(label_str.is_in(["1", "true", "target", "t"]))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("label_int")
        )

    # 选出最终需要的列
    keep_cols = FEATURE_COLUMNS + [
        "file_id",
        "instrument",
        "scan_number",
        "group_key",
        "peptide_key",
    ]

    if is_train:
        keep_cols.append("label_int")

    if not is_train:
        keep_cols.append("index")

    # 有些测试文件可能没有 file_id，这里补一下
    if "file_id" not in df.columns:
        df = df.with_columns(pl.lit(source_path.stem).alias("file_id"))

    return df.select(keep_cols)


def sample_balanced_by_label(df: pl.DataFrame, rows_per_class: int, seed: int):
    """
    从单个 parquet 里采样 target 和 decoy。
    这样可以避免某些大文件压倒其他文件，也避免类别严重不平衡。
    """
    if rows_per_class <= 0:
        return df

    parts = []

    for label_value in [0, 1]:
        sub = df.filter(pl.col("label_int") == label_value)

        if sub.height == 0:
            continue

        n = min(rows_per_class, sub.height)
        parts.append(sub.sample(n=n, seed=seed))

    if len(parts) == 0:
        return df.head(0)

    return pl.concat(parts)


def polars_to_xy(pdf: pd.DataFrame):
    """
    pandas DataFrame -> X, y
    """
    X = (
        pdf[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    y = pdf["label_int"].astype(np.int8).values

    return X, y


def polars_to_x(pdf: pd.DataFrame):
    """
    pandas DataFrame -> X
    """
    X = (
        pdf[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )

    return X


# ============================================================
# 4. 读取训练 / 验证数据
# ============================================================

def load_training_table(
    files,
    rows_per_class_per_file=1000,
    max_files=None,
    seed=42,
):
    """
    分文件读取 processed parquet，生成训练表。
    """

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    frames = []

    for p in tqdm(files, desc="读取训练/验证特征"):
        try:
            raw = pl.read_parquet(p)
            feat = make_features(raw, p, is_train=True)
            feat = sample_balanced_by_label(feat, rows_per_class_per_file, seed=seed)

            if feat.height == 0:
                continue

            frames.append(feat.to_pandas())

        except Exception as e:
            print(f"跳过文件：{p}")
            print(f"错误信息：{e}")

    if len(frames) == 0:
        raise RuntimeError("没有成功读取任何训练数据。")

    df = pd.concat(frames, ignore_index=True)
    return df


# ============================================================
# 5. 简单 FDR 验证
# ============================================================

def evaluate_simple_fdr(valid_df: pd.DataFrame, score: np.ndarray, fdr_threshold=0.01):
    """
    一个简化版 target-decoy FDR 验证。

    注意：
    这不是官方完整评测，只用于本地判断 score 方向是否正确。
    """
    df = valid_df[["label_int", "peptide_key", "instrument"]].copy()
    df["score"] = score

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    is_target = (df["label_int"].values == 1)
    is_decoy = ~is_target

    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)

    fdr = cum_decoy / np.maximum(cum_target, 1)

    # q-value：从后往前取最小 FDR
    qvalue = np.minimum.accumulate(fdr[::-1])[::-1]
    df["qvalue"] = qvalue

    accepted = df[(df["label_int"] == 1) & (df["qvalue"] <= fdr_threshold)]

    unique_peptides = accepted["peptide_key"].nunique()
    accepted_psms = len(accepted)

    print("\n========== 简单 FDR 验证 ==========")
    print(f"FDR 阈值: {fdr_threshold}")
    print(f"通过阈值的 target PSM 数: {accepted_psms}")
    print(f"通过阈值的 unique peptide 数: {unique_peptides}")

    print("\n按仪器统计 unique peptide:")
    for inst, sub in accepted.groupby("instrument"):
        print(f"{inst}: {sub['peptide_key'].nunique()}")

    return {
        "accepted_psms": accepted_psms,
        "unique_peptides": unique_peptides,
    }


# ============================================================
# 6. 训练模型
# ============================================================

def train(args):
    root = Path(args.root)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_files = collect_processed_files(root)

    train_files, valid_files = split_train_valid_files(
        all_files,
        valid_ratio=args.valid_ratio,
    )

    print(f"\n训练文件数: {len(train_files)}")
    print(f"验证文件数: {len(valid_files)}")

    train_df = load_training_table(
        train_files,
        rows_per_class_per_file=args.rows_per_class_per_file,
        max_files=args.max_train_files,
        seed=args.seed,
    )

    valid_df = load_training_table(
        valid_files,
        rows_per_class_per_file=args.rows_per_class_per_file,
        max_files=args.max_valid_files,
        seed=args.seed,
    )

    print("\n训练样本数:", len(train_df))
    print("验证样本数:", len(valid_df))

    print("\n训练 label 分布:")
    print(train_df["label_int"].value_counts())

    print("\n验证 label 分布:")
    print(valid_df["label_int"].value_counts())

    X_train, y_train = polars_to_xy(train_df)
    X_valid, y_valid = polars_to_xy(valid_df)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )

    print("\n开始训练 LightGBM...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    )

    valid_score = model.predict_proba(X_valid)[:, 1]

    evaluate_simple_fdr(valid_df, valid_score, fdr_threshold=0.01)

    model_path = model_dir / "baseline_lgbm.pkl"

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
        },
        model_path,
    )

    print(f"\n模型已保存：{model_path}")

    # 保存验证集预测，方便你后面排查
    valid_out = valid_df[[
        "file_id",
        "instrument",
        "scan_number",
        "group_key",
        "peptide_key",
        "label_int",
    ]].copy()

    valid_out["score"] = valid_score

    valid_out_path = model_dir / "valid_pred.parquet"
    valid_out.to_parquet(valid_out_path, index=False)

    print(f"验证集预测已保存：{valid_out_path}")


# ============================================================
# 7. 测试集推理
# ============================================================

def predict(args):
    model_obj = joblib.load(args.model_path)
    model = model_obj["model"]

    root = Path(args.root)
    test_dir = root / "bas_data"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(test_dir.glob("*.parquet"))

    print(f"找到测试 parquet 文件数：{len(test_files)}")

    all_pred_path = out_dir / "all_pred.tsv"

    total_rows = 0

    with open(all_pred_path, "w", encoding="utf-8") as f:
        f.write("index\tscore\n")

        for p in tqdm(test_files, desc="测试集推理"):
            try:
                raw = pl.read_parquet(p)

                feat = make_features(raw, p, is_train=False)

                index = feat["index"].to_numpy()

                pdf = feat.to_pandas()
                X = polars_to_x(pdf)

                score = model.predict_proba(X)[:, 1]

                out = pd.DataFrame({
                    "index": index,
                    "score": score,
                })

                # 每个测试文件单独保存一份，方便排查
                part_path = out_dir / f"{p.stem}_pred.csv"
                out.to_csv(part_path, index=False, header=False)

                # 追加写入 all_pred.tsv
                out.to_csv(
                    f,
                    sep="\t",
                    index=False,
                    header=False,
                    mode="a",
                )

                total_rows += len(out)

            except Exception as e:
                print(f"\n处理测试文件失败：{p}")
                print(f"错误信息：{e}")

    print(f"\n预测完成：{all_pred_path}")
    print(f"总预测行数：{total_rows}")

    if total_rows != 10_768_114:
        print("警告：预测行数不是 10,768,114，请检查是否有测试文件处理失败。")
    else:
        print("行数正确。")


# ============================================================
# 8. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--root", type=str, default=r"E:\AIPC_dataset")
    train_parser.add_argument("--model_dir", type=str, default=r"E:\AIPC_dataset\models\baseline_lgbm")
    train_parser.add_argument("--valid_ratio", type=float, default=0.2)
    train_parser.add_argument("--rows_per_class_per_file", type=int, default=1000)
    train_parser.add_argument("--max_train_files", type=int, default=0)
    train_parser.add_argument("--max_valid_files", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)

    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--root", type=str, default=r"E:\AIPC_dataset")
    pred_parser.add_argument("--model_path", type=str, default=r"E:\AIPC_dataset\models\baseline_lgbm\baseline_lgbm.pkl")
    pred_parser.add_argument("--out_dir", type=str, default=r"E:\AIPC_dataset\pred\baseline_lgbm")

    args = parser.parse_args()

    # argparse 默认读到的是 0，这里转成 None，表示不限制文件数量
    if hasattr(args, "max_train_files") and args.max_train_files == 0:
        args.max_train_files = None

    if hasattr(args, "max_valid_files") and args.max_valid_files == 0:
        args.max_valid_files = None

    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)


if __name__ == "__main__":
    main()

# python src\model\lightGBM_pipeline.py train   --root "E:\AIPC_dataset"   --model_dir "E:\AIPC_dataset\models\baseline_lgbm_test"   --rows_per_class_per_file 500 --max_train_files 200 --max_valid_files 100
