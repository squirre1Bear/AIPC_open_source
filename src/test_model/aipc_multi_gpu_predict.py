import argparse
import heapq
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import pyarrow.parquet as pq
import torch


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def normalize_output_stem(file_path: str) -> str:
    stem = os.path.basename(file_path)[: -len(".parquet")]
    if stem.endswith("_benchmark"):
        stem = stem[: -len("_benchmark")]
    return stem


def parse_gpus(gpus_arg: str) -> List[str]:
    if gpus_arg.strip().lower() == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, so auto GPU selection cannot be used.")
        return [str(i) for i in range(torch.cuda.device_count())]

    gpus = [item.strip() for item in gpus_arg.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPU ids were parsed. Use --gpus auto or a comma-separated list like 0,1,2,3.")
    return gpus


def list_parquet_files(parquet_dir: str) -> List[str]:
    files = sorted(
        os.path.join(parquet_dir, name)
        for name in os.listdir(parquet_dir)
        if name.endswith(".parquet")
    )
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    return files


def completed_result_path(out_path: str, parquet_file: str) -> str:
    stem = normalize_output_stem(parquet_file)
    return os.path.join(out_path, f"{stem}_benchmark_result.tsv")


def file_num_rows(file_path: str) -> int:
    return int(pq.ParquetFile(file_path).metadata.num_rows)


@dataclass(order=True)
class WorkerBucket:
    total_rows: int
    worker_idx: int
    gpu_id: str = field(compare=False)
    files: List[str] = field(default_factory=list, compare=False)


def assign_files_to_workers(files: Sequence[str], gpu_ids: Sequence[str]) -> List[WorkerBucket]:
    buckets = [WorkerBucket(total_rows=0, worker_idx=i, gpu_id=gpu) for i, gpu in enumerate(gpu_ids)]
    heap = buckets[:]
    heapq.heapify(heap)

    weighted_files: List[Tuple[int, str]] = []
    for file_path in files:
        weighted_files.append((file_num_rows(file_path), file_path))
    weighted_files.sort(key=lambda item: item[0], reverse=True)

    for rows, file_path in weighted_files:
        bucket = heapq.heappop(heap)
        bucket.files.append(file_path)
        bucket.total_rows += rows
        heapq.heappush(heap, bucket)

    return sorted(heap, key=lambda item: item.worker_idx)


def write_file_list(file_list_path: str, files: Sequence[str]) -> None:
    with open(file_list_path, "w", encoding="utf-8") as f:
        for item in files:
            f.write(item + "\n")


def run_worker(
    worker: WorkerBucket,
    project_root: str,
    python_executable: str,
    model_path: str,
    parquet_dir: str,
    config_path: str,
    out_path: str,
    predict_batch_size: int,
    parquet_batch_rows: int,
    file_list_path: str,
    log_path: str,
) -> subprocess.Popen:
    cmd = [
        python_executable,
        "-m",
        "src.test_model.aipc_test_baseline",
        "--model_path",
        model_path,
        "--parquet_dir",
        parquet_dir,
        "--config",
        config_path,
        "--out_path",
        out_path,
        "--predict_batch_size",
        str(predict_batch_size),
        "--parquet_batch_rows",
        str(parquet_batch_rows),
        "--file_list",
        file_list_path,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = worker.gpu_id
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._codex_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def wait_processes(processes: List[Tuple[WorkerBucket, subprocess.Popen, str]]) -> None:
    failures = []
    while processes:
        remaining = []
        for worker, proc, log_path in processes:
            code = proc.poll()
            if code is None:
                remaining.append((worker, proc, log_path))
                continue

            proc._codex_log_handle.close()  # type: ignore[attr-defined]
            if code == 0:
                logger.info(
                    "Worker %d on GPU %s finished successfully. Files=%d, estimated_rows=%d, log=%s",
                    worker.worker_idx,
                    worker.gpu_id,
                    len(worker.files),
                    worker.total_rows,
                    log_path,
                )
            else:
                failures.append((worker, code, log_path))
        if remaining:
            time.sleep(5)
        processes = remaining

    if failures:
        messages = [
            f"worker={worker.worker_idx}, gpu={worker.gpu_id}, exit_code={code}, log={log_path}"
            for worker, code, log_path in failures
        ]
        raise RuntimeError("Some inference workers failed:\n" + "\n".join(messages))


def run_merge(project_root: str, python_executable: str, out_path: str, track_type: str, submit_path: str) -> None:
    cmd = [
        python_executable,
        "src/test_model/aipc_generate_submit.py",
        "--dir",
        out_path,
        "--output",
        submit_path,
        "--type",
        track_type,
    ]
    subprocess.run(cmd, cwd=project_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AIPC baseline inference in parallel across multiple GPUs.")
    parser.add_argument("--project_root", default=".", help="Project root that contains src/")
    parser.add_argument("--python_executable", default=sys.executable, help="Python interpreter used to launch workers")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--parquet_batch_rows", type=int, default=4096)
    parser.add_argument("--gpus", default="auto", help="GPU ids, for example 0,1,2,3. Use auto to detect visible GPUs.")
    parser.add_argument("--type", default="basic", choices=["basic", "advanced"], help="Competition track type for submit merge")
    parser.add_argument("--skip_merge", action="store_true", help="Only run inference, do not merge final submit TSV")
    parser.add_argument("--force_rerun", action="store_true", help="Re-run files even if benchmark_result.tsv already exists")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    shard_dir = os.path.join(args.out_path, "_worker_shards")
    log_dir = os.path.join(args.out_path, "_worker_logs")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    gpu_ids = parse_gpus(args.gpus)
    files = list_parquet_files(args.parquet_dir)

    pending_files = []
    skipped = 0
    for file_path in files:
        result_path = completed_result_path(args.out_path, file_path)
        if not args.force_rerun and os.path.exists(result_path):
            skipped += 1
            continue
        pending_files.append(file_path)

    logger.info("Detected %d parquet files. Pending=%d, already_done=%d", len(files), len(pending_files), skipped)

    if pending_files:
        workers = assign_files_to_workers(pending_files, gpu_ids)
        processes: List[Tuple[WorkerBucket, subprocess.Popen, str]] = []

        for worker in workers:
            if not worker.files:
                logger.info("Worker %d on GPU %s has no files assigned.", worker.worker_idx, worker.gpu_id)
                continue

            file_list_path = os.path.join(shard_dir, f"worker_{worker.worker_idx}_files.txt")
            log_path = os.path.join(log_dir, f"worker_{worker.worker_idx}_gpu_{worker.gpu_id}.log")
            write_file_list(file_list_path, worker.files)
            logger.info(
                "Launching worker %d on GPU %s with %d files and estimated_rows=%d",
                worker.worker_idx,
                worker.gpu_id,
                len(worker.files),
                worker.total_rows,
            )
            process = run_worker(
                worker=worker,
                project_root=args.project_root,
                python_executable=args.python_executable,
                model_path=args.model_path,
                parquet_dir=args.parquet_dir,
                config_path=args.config,
                out_path=args.out_path,
                predict_batch_size=args.predict_batch_size,
                parquet_batch_rows=args.parquet_batch_rows,
                file_list_path=file_list_path,
                log_path=log_path,
            )
            processes.append((worker, process, log_path))

        wait_processes(processes)
    else:
        logger.info("No pending files. Skipping worker launch.")

    if args.skip_merge:
        logger.info("Skipping submit merge because --skip_merge was set.")
        return

    submit_name = "bas_submit.tsv" if args.type == "basic" else "adv_submit.tsv"
    submit_path = os.path.join(args.out_path, submit_name)
    run_merge(
        project_root=args.project_root,
        python_executable=args.python_executable,
        out_path=args.out_path,
        track_type=args.type,
        submit_path=submit_path,
    )
    logger.info("Submit file generated at %s", submit_path)


if __name__ == "__main__":
    main()

# python3 src/test_model/aipc_multi_gpu_predict.py \
#   --project_root /home/yhc/projects/AIPC/ \
#   --model_path /home/yhc/projects/pfind_AIPC/model/best_full_aux_20260407_v1/best.pt \
#   --parquet_dir ./data/bas_test_dataset \
#   --config ./model.yaml \
#   --out_path ./data/bas_test_score \
#   --predict_batch_size 512 \
#   --parquet_batch_rows 4096 \
#   --gpus auto \
#   --type basic


# python src/test_model/aipc_multi_gpu_predict.py ^
#   --project_root D:/Python_Projects/pfind_AIPC ^
#   --python_executable D:/Python_Projects/pfind_AIPC/.venv/Scripts/python.exe ^
#   --model_path D:/Python_Projects/pfind_AIPC/model/best_full_aux_20260407_v1/best.pt ^
#   --parquet_dir E:/AIPC_dataset/bas_test_dataset ^
#   --config D:/Python_Projects/pfind_AIPC/model.yaml ^
#   --out_path E:/AIPC_dataset/bas_test_score ^
#   --predict_batch_size 512 ^
#   --parquet_batch_rows 4096 ^
#   --gpus auto ^
#   --type basic