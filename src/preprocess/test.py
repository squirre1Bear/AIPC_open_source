import polars as pl

df = pl.read_parquet(r"E:\AIPC_dataset\processed\mzml_merged\mzml_merged_0000.parquet")

print(df.select([
    pl.col("mz_array").list.len().max().alias("max_mz_len"),
    pl.col("intensity_array").list.len().max().alias("max_intensity_len"),
    pl.col("mz_array").is_null().sum().alias("null_mz_array"),
]))

print([
    c for c in [
        "abs_ppm_iso_-1",
        "abs_ppm_iso_0",
        "abs_ppm_iso_1",
        "abs_ppm_iso_2",
        "min_abs_precursor_ppm"
    ]
    if c not in df.columns
])