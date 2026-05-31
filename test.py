from pathlib import Path
import polars as pl

BAS_DIR = Path("/root/autodl-tmp/datasets/aipc/bas_data")

files = sorted(BAS_DIR.glob("*.parquet"))

print(f"找到 parquet 文件: {len(files)}")
print("=" * 80)

bad_files = []

for i, path in enumerate(files):
    size = path.stat().st_size

    try:
        # 检查 parquet 文件头尾
        with open(path, "rb") as f:
            head = f.read(4)

            if size >= 4:
                f.seek(-4, 2)
                tail = f.read(4)
            else:
                tail = b""

        magic_ok = head == b"PAR1" and tail == b"PAR1"

        # 尝试用 polars 读取 1 行
        pl.read_parquet(path, n_rows=1)

        if magic_ok:
            print(f"OK   [{i:02d}] {path.name}  size={size}")
        else:
            print(f"WARN [{i:02d}] {path.name}  size={size}  head={head} tail={tail}")

    except Exception as e:
        print(f"BAD  [{i:02d}] {path.name}  size={size}")
        print(f"     head={head if 'head' in locals() else None} tail={tail if 'tail' in locals() else None}")
        print(f"     error={type(e).__name__}: {e}")

        bad_files.append(path)

print("=" * 80)

if bad_files:
    print(f"发现损坏 parquet 文件: {len(bad_files)}")
    for p in bad_files:
        print(p)
else:
    print("所有 parquet 文件都可以正常读取")