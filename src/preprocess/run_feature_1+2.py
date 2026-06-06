from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


# ============================================================
# 路径配置
# ============================================================

PREPROCESS_DIR = Path(r"D:\Python_Projects\pfind_AIPC\src\preprocess")

SCRIPT_1 = PREPROCESS_DIR / "add_feature_1.py"
SCRIPT_2 = PREPROCESS_DIR / "add_feature_2.py"

LOG_DIR = PREPROCESS_DIR / "pipeline_logs"

# 如果第一步有失败，是否还继续跑第二步
# 建议 False：因为第二步自己会跳过 aux_feature_done 不存在的文件
STOP_IF_STAGE1_FAILED = False

# 是否在两个脚本跑完后，额外扫描 parquet 里的 parse_ok / fragment_parse_ok
ENABLE_PARQUET_QUALITY_CHECK = True

DATA_DIRS = [
    Path(r"/root/autodl-tmp/datasets/aipc/processed/mzml_merged"),
    Path(r"/root/autodl-tmp/datasets/aipc/processed/tims_merged"),
    Path(r"/root/autodl-tmp/datasets/aipc/processed/wiff_merged"),
]
# DATA_DIRS = [
#     Path(r"E:\AIPC_dataset\processed\mzml_merged"),
#     Path(r"E:\AIPC_dataset\processed\tims_merged"),
#     Path(r"E:\AIPC_dataset\processed\wiff_merged"),
# ]


# ============================================================
# 错误记录结构
# ============================================================

@dataclass
class Issue:
    stage: str
    issue_type: str
    path: str
    message: str
    line_no: int | None = None


# ============================================================
# 日志解析
# ============================================================

FAIL_PATH_RE = re.compile(r"处理失败[:：]\s*(?P<path>.+?)\s*$")
NO_AUX_RE = re.compile(r"基础辅助特征尚未完成，跳过[:：]\s*(?P<path>.+?)\s*$")
ERROR_INFO_RE = re.compile(r"错误信息[:：]\s*(?P<msg>.+?)\s*$")
RETURN_CODE_RE = re.compile(r"子进程退出码[:：]\s*(?P<code>.+?)\s*$")
CURRENT_FILE_RE = re.compile(r"(?:正在处理|开始处理)[:：]?\s*(?P<path>[A-Za-z]:\\.+?\.parquet)\s*$")
PEPTIDE_EMPTY_RE = re.compile(r"peptide_feature长度为0，有误")


def parse_log(stage: str, log_text: str) -> list[Issue]:
    issues: list[Issue] = []
    current_path = ""

    last_failed_issue_index: int | None = None

    for i, line in enumerate(log_text.splitlines(), start=1):
        line = line.strip()

        m = CURRENT_FILE_RE.search(line)
        if m:
            current_path = m.group("path")

        m = FAIL_PATH_RE.search(line)
        if m:
            issue = Issue(
                stage=stage,
                issue_type="failed_file",
                path=m.group("path"),
                message=line,
                line_no=i,
            )
            issues.append(issue)
            last_failed_issue_index = len(issues) - 1
            continue

        m = NO_AUX_RE.search(line)
        if m:
            issues.append(
                Issue(
                    stage=stage,
                    issue_type="skipped_no_aux_feature",
                    path=m.group("path"),
                    message=line,
                    line_no=i,
                )
            )
            continue

        m = ERROR_INFO_RE.search(line)
        if m and last_failed_issue_index is not None:
            issues[last_failed_issue_index].message += " | " + line
            continue

        m = RETURN_CODE_RE.search(line)
        if m and last_failed_issue_index is not None:
            issues[last_failed_issue_index].message += " | " + line
            continue

        if PEPTIDE_EMPTY_RE.search(line):
            issues.append(
                Issue(
                    stage=stage,
                    issue_type="empty_peptide_feature",
                    path=current_path,
                    message=line,
                    line_no=i,
                )
            )

    return issues


# ============================================================
# 运行子脚本
# ============================================================

def run_stage(stage: str, script_path: Path) -> tuple[int, Path, list[Issue]]:
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在：{script_path}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}_{stage}.log"

    env = os.environ.copy()
    env.setdefault("POLARS_MAX_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("=" * 100)
    print(f"开始运行阶段：{stage}")
    print(f"脚本：{script_path}")
    print(f"日志：{log_path}")
    print("=" * 100)

    lines: list[str] = []

    with subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    ) as proc:
        assert proc.stdout is not None

        for line in proc.stdout:
            print(line, end="")
            lines.append(line)

        return_code = proc.wait()

    log_text = "".join(lines)
    log_path.write_text(log_text, encoding="utf-8", errors="replace")

    issues = parse_log(stage, log_text)

    print()
    print(f"阶段完成：{stage}")
    print(f"退出码：{return_code}")
    print(f"解析到问题数：{len(issues)}")
    print()

    return return_code, log_path, issues


# ============================================================
# 可选：扫描 parquet 里的质量标记
# ============================================================

def iter_parquet_files():
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue

        for path in sorted(data_dir.rglob("*.parquet")):
            name = path.name

            if ".tmp" in name:
                continue
            if name.endswith(".bak") or name.endswith(".bak_fragment"):
                continue

            yield path


def parquet_quality_check() -> list[dict]:
    try:
        import polars as pl
    except Exception as e:
        print(f"无法导入 polars，跳过 parquet 质量检查：{e}")
        return []

    rows: list[dict] = []

    for path in iter_parquet_files():
        row = {
            "path": str(path),
            "has_aux_feature_done": None,
            "has_fragment_feature_done": None,
            "row_count": None,
            "parse_ok_0_rows": None,
            "fragment_parse_ok_0_rows": None,
            "unknown_mod_rows": None,
            "error": "",
        }

        try:
            lf = pl.scan_parquet(path)
            cols = lf.collect_schema().names()

            row["has_aux_feature_done"] = "aux_feature_done" in cols
            row["has_fragment_feature_done"] = "fragment_feature_done" in cols

            exprs = [pl.len().alias("row_count")]

            if "parse_ok" in cols:
                exprs.append((pl.col("parse_ok") == 0).sum().alias("parse_ok_0_rows"))

            if "fragment_parse_ok" in cols:
                exprs.append(
                    (pl.col("fragment_parse_ok") == 0)
                    .sum()
                    .alias("fragment_parse_ok_0_rows")
                )

            if "unknown_mod_count" in cols:
                exprs.append(
                    (pl.col("unknown_mod_count") > 0)
                    .sum()
                    .alias("unknown_mod_rows")
                )

            stat = lf.select(exprs).collect().row(0, named=True)

            for key, value in stat.items():
                row[key] = value

        except Exception as e:
            row["error"] = repr(e)

        rows.append(row)

    return rows


# ============================================================
# 输出汇总
# ============================================================

def write_issues_csv(issues: list[Issue], path: Path):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stage", "issue_type", "path", "message", "line_no"],
        )
        writer.writeheader()

        for issue in issues:
            writer.writerow(asdict(issue))


def write_dicts_csv(rows: list[dict], path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_issues: list[Issue] = []
    stage_logs: dict[str, str] = {}
    stage_return_codes: dict[str, int] = {}

    # -----------------------------
    # 第一阶段：add_feature_1
    # -----------------------------
    rc1, log1, issues1 = run_stage("add_feature_1", SCRIPT_1)
    all_issues.extend(issues1)
    stage_logs["add_feature_1"] = str(log1)
    stage_return_codes["add_feature_1"] = rc1

    if STOP_IF_STAGE1_FAILED and (rc1 != 0 or len(issues1) > 0):
        print("第一阶段发现失败，已停止第二阶段。")
    else:
        # -----------------------------
        # 第二阶段：add_feature_2
        # -----------------------------
        rc2, log2, issues2 = run_stage("add_feature_2", SCRIPT_2)
        all_issues.extend(issues2)
        stage_logs["add_feature_2"] = str(log2)
        stage_return_codes["add_feature_2"] = rc2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    issues_csv = LOG_DIR / f"{timestamp}_pipeline_issues.csv"
    issues_json = LOG_DIR / f"{timestamp}_pipeline_summary.json"

    write_issues_csv(all_issues, issues_csv)

    quality_csv = None
    quality_rows = []

    if ENABLE_PARQUET_QUALITY_CHECK:
        print("=" * 100)
        print("开始扫描 parquet 质量标记")
        print("=" * 100)

        quality_rows = parquet_quality_check()
        quality_csv = LOG_DIR / f"{timestamp}_parquet_quality_summary.csv"
        write_dicts_csv(quality_rows, quality_csv)

    summary = {
        "created_at": timestamp,
        "stage_logs": stage_logs,
        "stage_return_codes": stage_return_codes,
        "issue_count": len(all_issues),
        "issues_csv": str(issues_csv),
        "quality_csv": str(quality_csv) if quality_csv else None,
        "issues": [asdict(x) for x in all_issues],
    }

    issues_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("Pipeline 完成")
    print("=" * 100)
    print(f"问题汇总 CSV：{issues_csv}")
    print(f"问题汇总 JSON：{issues_json}")

    if quality_csv:
        print(f"parquet 质量检查 CSV：{quality_csv}")

    if all_issues:
        print()
        print("发现以下问题：")
        for issue in all_issues:
            print(f"[{issue.stage}] [{issue.issue_type}] {issue.path}")
            print(f"  {issue.message}")

        sys.exit(1)

    print("没有从日志中解析到失败文件。")


if __name__ == "__main__":
    main()


# 运行
# python D:\Python_Projects\pfind_AIPC\src\preprocess\run_feature_1+2.py

# 会有输出
# D:\Python_Projects\pfind_AIPC\src\preprocess\pipeline_logs\*_add_feature_1.log
# D:\Python_Projects\pfind_AIPC\src\preprocess\pipeline_logs\*_add_feature_2.log
# D:\Python_Projects\pfind_AIPC\src\preprocess\pipeline_logs\*_pipeline_issues.csv
# D:\Python_Projects\pfind_AIPC\src\preprocess\pipeline_logs\*_pipeline_summary.json
# D:\Python_Projects\pfind_AIPC\src\preprocess\pipeline_logs\*_parquet_quality_summary.csv
