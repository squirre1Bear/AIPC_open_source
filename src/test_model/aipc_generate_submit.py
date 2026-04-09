import argparse
import glob
import os
import sys

import pandas as pd


class FilenameError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Check and merge benchmark TSV files")
    parser.add_argument("--dir", required=True, help="Directory containing TSV files")
    parser.add_argument("--output", required=True, help="Output merged TSV path")
    parser.add_argument("--type", required=True, help="basic or advanced")
    args = parser.parse_args()

    basic_file_list = glob.glob(os.path.join(args.dir, "bas*_benchmark_result.tsv"))
    advanced_file_list = glob.glob(os.path.join(args.dir, "adv*_benchmark_result.tsv"))

    print("== Step 1: check exists ==")
    if args.type == "basic":
        if len(basic_file_list) != 60:
            raise FileNotFoundError(
                f"Your pred files for the test dataset of basic track are not complete, we need 60, but you only have {len(basic_file_list)}"
            )
    elif args.type == "advanced":
        if len(advanced_file_list) != 50:
            raise FileNotFoundError(
                f"Your pred files for the test dataset of advanced track are not complete, we need 50, but you only have {len(advanced_file_list)}"
            )
    else:
        print("ERROR: your args type is wrong, should be basic or advanced")
        sys.exit()
    print("The number of pred files is correct.")

    print("== Step 2: check filename ==")
    if args.type == "basic":
        target_fn_set = {f"bas_a_testdata_{num}_benchmark_result.tsv" for num in range(30)}
        target_fn_set.update({f"bas_b_testdata_{num}_benchmark_result.tsv" for num in range(30)})
        actual = {os.path.basename(path) for path in basic_file_list}
        if actual != target_fn_set:
            raise FilenameError(
                "The name of your pred files should be bas_a_testdata_num_benchmark_result.tsv or bas_b_testdata_num_benchmark_result.tsv"
            )
        file_list = basic_file_list
    elif args.type == "advanced":
        target_fn_set = {f"adv_a_testdata_{num}_benchmark_result.tsv" for num in range(30)}
        target_fn_set.update({f"adv_b_testdata_{num}_benchmark_result.tsv" for num in range(20)})
        actual = {os.path.basename(path) for path in advanced_file_list}
        if actual != target_fn_set:
            raise FilenameError(
                "The name of your pred files should be adv_a_testdata_num_benchmark_result.tsv or adv_b_testdata_num_benchmark_result.tsv"
            )
        file_list = advanced_file_list
    else:
        print("ERROR: your args type is wrong, should be basic or advanced")
        sys.exit()
    print("The name of pred files is correct.")

    print("\n== Step 3: combine all tsv files ==")
    merged_list = []
    for file in file_list:
        df = pd.read_csv(file, sep="\t")
        fname = os.path.basename(file)[: -len("_benchmark_result.tsv")]
        df["filename"] = fname
        merged_list.append(df)

    merged_df = pd.concat(merged_list, ignore_index=True)
    merged_df.to_csv(args.output, sep="\t", index=False)
    print(f"combine finished: {args.output}")


if __name__ == "__main__":
    main()
