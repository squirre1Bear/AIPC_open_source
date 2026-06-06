  本项目用于 AIPC（人工智能蛋白质组学系列竞赛：利用人工智能揭示蛋白质组学的“暗物质”）提交结果生
  成。

  核心思路：利用训练集中已经出现过的高置信肽段序列构建 peptide lexicon，将训练集中的
  target / decoy 统计信息编码成一个查表分数，再与已有 benchmark 模型分数融合，用于提升 1% FDR 下
  鉴定到的唯一 clean peptide 数量。

  ## 模型介绍

  模型由两部分组成：

  1. 基础模型分数
     使用 lightGBM 对测试集进行初打分，得到benchmark_score。

  2. 全训练集 peptide lexicon 分数
     从训练集构建 peptide 词表，统计每个 peptide key 在 target 和 decoy 中出现的次数，并计算：

  lex_logodds = log((target_count + 1) / (decoy_count + 1))

  最终提交分数为：

  final_score = benchmark_score + weight * lex_logodds

  其中 weight 通过本地 offline leaderboard 网格搜索确定。

  ## Peptide 编码方式

  项目中主要验证了两种 peptide key 编码。

  ### clean 编码

  对 precursor_sequence 做基础清洗：

  n[42]      -> 空
  N[.98]    -> N
  Q[.98]    -> Q
  M[15.99]  -> M
  C[57.02]  -> C

  该方式保留大部分原始 peptide 表达，只处理常见固定/可解释修饰。

  ### stripped 编码

  在 clean 编码基础上进一步处理：

  删除所有 [...] 修饰
  删除所有非大写字母字符

  相比于 clean 编码，stripped 目标是把不同修饰形式映射到同一个氨基酸主链序列，从而扩大测试集覆盖率。

  ## 目录结构

  src/
    eval/
      offline_leaderboard.py
      aipc_lexicon_probe_polars.py

    submit/
      make_lexicon_boost_submission.py

  eval/
    lexicon_full_clean_train_*/
    lexicon_full_stripped_train_*/
    lexicon_full_clean_valid_grid*/
    lexicon_full_stripped_valid_grid*/


  关键脚本说明：

  src/eval/offline_leaderboard.py

  统一 offline leaderboard 评测入口。

  src/eval/aipc_lexicon_probe_polars.py

  用于构建/加载 lexicon，并在验证集上测试不同 lexicon 权重。

  src/submit/make_lexicon_boost_submission.py

  使用 benchmark 分数和 lexicon 分数生成最终提交文件。

  ## 数据准备

  默认数据目录：

  /root/autodl-tmp/datasets/aipc

  训练集 split 目录：

  /root/autodl-tmp/datasets/aipc/processed_split/train

  测试集目录：

  /root/autodl-tmp/datasets/aipc/processed/bas_merged

  benchmark 提交目录示例：

  /root/aipc/submissions/0605/seq_support_strong_cpu16_best_20260605_204201

  ## 构建 Lexicon

  如果已经有 lexicon parquet，可以直接跳过构建阶段。

  clean lexicon 示例：

  python src/eval/aipc_lexicon_probe_polars.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --valid-pred-root /root/aipc/models/exp_by_instrument/v1 \
    --out-dir /root/aipc/eval/lexicon_full_clean_valid_grid \
    --key-mode clean \
    --weights 0.25,0.5,1,2,4,8,12,16,24,32,48,64 \
    --workers 8

  stripped lexicon 示例：

  python src/eval/aipc_lexicon_probe_polars.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --valid-pred-root /root/aipc/models/exp_by_instrument/v1 \
    --out-dir /root/aipc/eval/lexicon_full_stripped_valid_grid \
    --key-mode stripped \
    --weights 64,128,256,512,1024,2048,4096,8192,16000,32768 \
    --workers 8

  ## 权重验证

  如果 lexicon 已经构建完成，可以通过 --lexicon 参数直接加载。

  clean 示例：

  python src/eval/aipc_lexicon_probe_polars.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --valid-pred-root /root/aipc/models/exp_by_instrument/v1 \
    --out-dir /root/aipc/eval/lexicon_full_clean_valid_grid_ext \
    --key-mode clean \
    --lexicon /root/aipc/eval/lexicon_full_clean_train_20260606_1052/
    lexicon_clean_full_train.parquet \
    --weights 2, 4, 8, 16, 32, 64,128,256,512,1024,2048,4096,8192,16000,32768,65536 \
    --workers 8

  stripped 示例：

  taskset -c 0-7 python src/eval/aipc_lexicon_probe_polars.py \
    --train-root /root/autodl-tmp/datasets/aipc/processed_split/train \
    --valid-pred-root /root/aipc/models/exp_by_instrument/v1 \
    --out-dir /root/aipc/eval/lexicon_full_stripped_valid_grid_ext \
    --key-mode stripped \
    --lexicon /root/aipc/eval/lexicon_full_stripped_train_20260606/
    lexicon_stripped_full_train.parquet \
    --weights 64,128,256,512,1024,2048,4096,8192,16000,32768,65536 \
    --workers 8

  验证完成后查看：

  leaderboard.csv
  details.json
  metadata.json

  其中主要关注：

  official_like_unique_clean_peptide_at_1pct

  ## 生成提交文件

  clean 示例：

   python src/submit/make_lexicon_boost_submission.py \
    --test-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
    --benchmark-dir /root/aipc/submissions/0605/seq_support_strong_cpu16_best_20260605_204201 \
    --lexicon /root/aipc/eval/lexicon_full_clean_train_20260606_1052/
    lexicon_clean_full_train.parquet \
    --out-dir /root/aipc/submissions/0606/lexicon_full_clean_benchmark_add_w64 \
    --weight 64 \
    --key-mode clean \
    --score-mode add \
    --base-source benchmark \
    --workers 8 \
    --expected-rows 10768114

  stripped 示例：

   python src/submit/make_lexicon_boost_submission.py \
    --test-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
    --benchmark-dir /root/aipc/submissions/0605/seq_support_strong_cpu16_best_20260605_204201 \
    --lexicon /root/aipc/eval/lexicon_full_stripped_train_20260606/
    lexicon_stripped_full_train.parquet \
    --out-dir /root/aipc/submissions/0606/lexicon_full_stripped_benchmark_add_w32768 \
    --weight 32768 \
    --key-mode stripped \
    --score-mode add \
    --base-source benchmark \
    --workers 8 \
    --expected-rows 10768114