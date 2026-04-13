import os
import argparse
import pandas as pd
import sys
import glob


class FilenameError(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description="Check & merge TSV files")
    parser.add_argument("--dir", required=True, help="Directory containing TSV files")
    parser.add_argument("--output", required=True, help="Output merged TSV path")
    parser.add_argument("--type", required=True, help="basic or advanced")

    args = parser.parse_args()

    basic_file_list = glob.glob(os.path.join(args.dir, 'bas*_benchmark_result.tsv'))
    advancd_file_list = glob.glob(os.path.join(args.dir, 'adv*_benchmark_result.tsv'))

    # ========== 1. check exists ==========
    print("== Step 1: check exists ==")
    if args.type == 'basic':
        if len(basic_file_list) != 60:
            raise FileNotFoundError(
                f"Your pred files for the test dataset of basic track are not complete, we need 60, but you only have {len(basic_file_list)}")
    elif args.type == 'advanced':
        if len(advancd_file_list) != 50:
            raise FileNotFoundError(
                f"Your pred files for the test dataset of advanced track are not complete, we need 50, but you only have {len(advancd_file_list)}")
    else:
        print('ERROR: your args type is wrong, should be basic or advanced')
        sys.exit()
    print("The number of pred files is correct.")

    # ========== 2. check filename ==========
    print("== Step 2: check filename ==")
    if args.type == 'basic':
        target_fn_set = set([f'bas_a_testdata_{num}_benchmark_result.tsv' for num in range(30)] + [
            f'bas_b_testdata_{num}_benchmark_result.tsv' for num in range(30)])
        if set([i.split('/')[-1] for i in basic_file_list]) != target_fn_set:
            raise FilenameError(
                "The name of your pred files should be bas_a_testdata_num_benchmark_result.tsv or bas_b_testdata_num_benchmark_result.tsv")
    elif args.type == 'advanced':
        target_fn_set = set([f'adv_a_testdata_{num}_benchmark_result.tsv' for num in range(30)] + [
            f'adv_b_testdata_{num}_benchmark_result.tsv' for num in range(20)])
        if set([i.split('/')[-1] for i in advancd_file_list]) != target_fn_set:
            raise FilenameError(
                "The name of your pred files should be adv_a_testdata_num_benchmark_result.tsv or adv_b_testdata_num_benchmark_result.tsv")
    else:
        print('ERROR: your args type is wrong, should be basic or advanced')
        sys.exit()
    print("The name of pred files is correct.")

    # ========== 3. combine all tsv files ==========
    print("\n== Step 3: combine all tsv files ==")
    merged_list = []

    if args.type == 'basic':
        for file in basic_file_list:
            df = pd.read_csv(file, sep="\t")
            fname = file.split('/')[-1][:-len('_benchmark_result.tsv')]
            df["filename"] = fname
            merged_list.append(df)
    elif args.type == 'advanced':
        for file in advancd_file_list:
            df = pd.read_csv(file, sep="\t")
            fname = file.split('/')[-1][:-len('_benchmark_result.tsv')]
            df["filename"] = fname
            merged_list.append(df)

    merged_df = pd.concat(merged_list, ignore_index=True)

    merged_df.to_csv(args.output, sep="\t", index=False)
    print(f"combine finished：{args.output} ✔")


if __name__ == "__main__":
    main()

# python3 ./src/test_model/aipc_generate_submit.py --dir ./data/bas_test_score --output ./data/bas_test_score/bas_submit.tsv --type basic
# python /zhangxiaofan/DDA_BERT_deltaRT/aipc_generate_submit_251204.py --dir /zhangxiaofan/DDA_BERT_deltaRT/pred/test_251204/bas_result --output /zhangxiaofan/DDA_BERT_deltaRT/pred/test_251204/bas_submit.tsv --type basic