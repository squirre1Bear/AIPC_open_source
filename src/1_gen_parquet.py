import pandas as pd
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-raw', type=str, default="", help="")
parser.add_argument('-sage_sr', type=str, default="", help="")
parser.add_argument('-fp_sr', type=str, default="", help="")
parser.add_argument('-parquet_path', type=str, default="", help="")

args = parser.parse_args()

def clean_psm_func(peptide, residues_dict):
    for key, value in residues_dict.items():
        if value not in peptide:
            peptide = peptide.replace(key, value)
    return peptide

# 读取谱图数据
raw_df = pd.read_parquet(args.raw)
raw_df['scan'] = raw_df['scan'].astype(int)
# 读取sage数据
sage_df = pd.read_parquet(args.sage_sr)
sage_df['scan'] = sage_df['scan'].astype(int)
sage_df['psm_id'] = sage_df['scan'].astype(str) + '_' + sage_df['precursor_sequence']
sage_df_target = sage_df[sage_df['label']==1]
sage_df_decoy = sage_df[sage_df['label']==0]
# 读取fp数据
fp_df = pd.read_parquet(args.fp_sr)
fp_df['psm_id'] = fp_df['scan'].astype(str) + '_' + fp_df['detect_sequence']
fp_df = fp_df.drop(columns=['scan'], axis=1)

# 生成parquet文件
target_sr_df = sage_df_target.merge(fp_df, on='psm_id', how='inner')
decoy_num = len(target_sr_df)
decoy_df_sorted = sage_df_decoy.sort_values(by='sage_discriminant_score',ascending=False).reset_index(drop=True)
if len(decoy_df_sorted) <= decoy_num:
    decoy_df_need = decoy_df_sorted
else:
    decoy_df_high_score = decoy_df_sorted.iloc[:int(decoy_num/2)]
    decoy_df_low_score = decoy_df_sorted.iloc[int(decoy_num/2):].sample(n=int(decoy_num/2),random_state=42)
    decoy_df_need = pd.concat([decoy_df_high_score,decoy_df_low_score], axis=0, ignore_index=True)
sr_df_need = pd.concat([target_sr_df, decoy_df_need], axis=0, ignore_index=True)
parquet_df = sr_df_need.merge(raw_df, on='scan', how='inner')

assert len(parquet_df) == len(sr_df_need), f"parquet df is {len(parquet_df)}, sr df need is {len(sr_df_need)}"
print(f'before the cut, psm num is {len(parquet_df)}')

parquet_df['cleaned_sequence'] = parquet_df['precursor_sequence'].str.replace('n[42]', '').str.replace('N[.98]', 'N').str.replace('Q[.98]', 'Q').str.replace('M[15.99]', 'M').str.replace('C[57.02]', 'C')
parquet_df['sequence_len'] = parquet_df['cleaned_sequence'].apply(len)
parquet_df = parquet_df[(parquet_df['sequence_len']<=50)&(parquet_df['sequence_len']>=7)]
parquet_df = parquet_df[(parquet_df['charge']<=5)&(parquet_df['charge']>=2)]

print(f'after the cut, psm num is {len(parquet_df)}')
parquet_df = parquet_df[['scan','precursor_mz','charge','rt','mz_array','intensity_array','precursor_sequence','label','predicted_rt', 'delta_rt','sage_discriminant_score','spectrum_q']]
parquet_df['weight'] = 1


# 保存
parquet_df.to_parquet(args.parquet_path)